import torch.optim as optim
import time
import argparse
import datetime
import os
import random
import sys
import hydra
import torch
from tqdm import tqdm
from utils.writer import Writer
from omegaconf import open_dict
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
import csv
from datasets.dataset import DataloaderMode
from utils.utils import get_logger, is_logging_process, print_config,set_random_seed
from dataloader.dataloader import create_dataloader

from models.MAE_fine_OCTA import mae_vit_large_patch16_dec512d8b, apply_dual_lora_to_block, set_lora_trainable_state

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def setup(cfg, rank):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    torch.cuda.empty_cache()

    os.environ["MASTER_ADDR"] = cfg.dist.master_addr
    os.environ["MASTER_PORT"] = cfg.dist.master_port
    timeout_sec = 1800
    if cfg.dist.timeout is not None:
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        timeout_sec = cfg.dist.timeout
    timeout = datetime.timedelta(seconds=timeout_sec)

    # initialize the process group
    dist.init_process_group(
        cfg.dist.mode,
        rank=rank,
        world_size=cfg.dist.gpus,
        timeout=timeout,
    )

def cleanup():
    dist.destroy_process_group()

def distributed_run(fn, cfg):
    mp.spawn(fn, args=(cfg, ), nprocs=cfg.dist.gpus, join=True)


def train_mae_reconstruction(model, train_loader, device, save_path, epochs=60, lr=1e-4, mask_ratio=0.75):
    """
    MAE Data Adaptation using masked image modeling, only decoder and Domain LoRA are trainable
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.to(device)

    print(f"\n--- Begining MAE Data Adaptation ---")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Number of Parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters (Data LoRA + Decoder): {trainable_params / 1e6:.2f} M")
    print(f"Learning Rate: {lr}, Mask Ratio: {mask_ratio}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print("--------------------------------------------------\n")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch_idx, (imgs, _) in enumerate(pbar):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(imgs, mask_ratio=mask_ratio)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        elapsed_time = time.time() - start_time

        # 打印 epoch 结束日志
        print(f"\n[Epoch {epoch}/{epochs}] 平均训练损失 (MAE): {avg_loss:.4f} | 耗时: {elapsed_time / 60:.2f} 分钟")

    print(f"\n--- MAE has finished Data Adaptation! ---")

    # save model weights
    chkpt_point_save_path = os.path.join(save_path, "MAE_domain_adaptation_weights.pth")
    torch.save(model.state_dict(), chkpt_point_save_path)
    print(f"Model's weights has save in : {chkpt_point_save_path}")

def test_mae_reconstruction(model, test_loader, device, save_path, mask_ratio=0.75):
    print(f"\n--- Testing MAE Data Adaptation ---")
    start_time = time.time()
    csv_file = os.path.join(save_path, "test_result.csv")
    results = []
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "loss"])

    with torch.no_grad():
        model.eval()
        total_loss = 0
        pbar = tqdm(test_loader, desc=f"Testing ...")
        for batch_idx, (imgs, file_name) in enumerate(pbar):
            imgs = imgs.to(device)
            loss, _, _ = model(imgs, mask_ratio=mask_ratio)
            total_loss += loss.item()
            file_name_s = file_name[0]
            results.append([file_name_s, loss.item()])
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            if (batch_idx + 1) % 1000 == 0:
                try:
                    with open(csv_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(results)
                    results = []
                except Exception as e:
                    print(f"Fail to save ：{str(e)}")

        avg_loss = total_loss / len(test_loader)
        elapsed_time = time.time() - start_time
        print(f"\nTesting MSE Loss: {avg_loss:.4f} | Using Time: {elapsed_time / 60:.2f} min")
    print(f"\n--- MAE Data Adaptation Finished. ---")


def train_loop(model, rank, cfg):
    logger = get_logger(cfg, os.path.basename(__file__))
    if cfg.device == "cuda" and cfg.dist.gpus != 0:
        cfg.device = rank
        # turn off background generator when distributed run is on
        cfg.data.use_background_generator = False
        setup(cfg, rank)
        torch.cuda.set_device(cfg.device)
        writer = None

    # setup writer
    if is_logging_process():
        os.makedirs(cfg.log.chkpt_dir, exist_ok=True)
        writer = Writer(cfg, "wandb")
        if cfg.data.data_root_dir == "" or cfg.data.data_root_dir_3d == "":
            logger.error("train or test data directory cannot be empty.")
            raise Exception("Please specify directories of data")
        logger.info("Set up train process")
        logger.info(
            "BackgroundGenerator is turned off when Distributed running is on")

    # This is just to avoid accident
    cfg.inference_mode = False
    cfg.testing_noise = False

    # Sync dist processes (because of download MNIST Dataset)
    if cfg.dist.gpus != 0:
        dist.barrier()

    if is_logging_process():
        logger.info("Making train dataloader...")
    train_loader = create_dataloader(cfg, DataloaderMode.train, rank)
    if is_logging_process():
        logger.info("Making validation dataloader...")
    val_loader = create_dataloader(cfg, DataloaderMode.validation, rank)
    test_loader = create_dataloader(cfg, DataloaderMode.test, rank)

    # Data Adaptatin
    # Replace your save path of data adaptation weights
    save_path = "../DA_check_point"
    train_mae_reconstruction(model, train_loader, device='cuda', save_path=save_path)

    test_mae_reconstruction(model, test_loader, device='cuda', save_path=save_path)



def get_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',type=str, default='default.yaml')
    args = parser.parse_args()
    return args.experiment


@hydra.main(version_base="1.1", config_path="config", config_name="default")
def main(hydra_cfg):
    hydra_cfg.device = hydra_cfg.device.lower()
    # change to your project path
    os.chdir("./TAPE")
    with open_dict(hydra_cfg):
        hydra_cfg.job_logging_cfg = HydraConfig.get().job_logging
    
    print_config(hydra_cfg,get_logger(hydra_cfg, os.path.basename(__file__), disable_console=True))

    if hydra_cfg.random_seed is None:
        hydra_cfg.random_seed = random.randint(1, 10000)
    set_random_seed(hydra_cfg.random_seed)

    mae_model = mae_vit_large_patch16_dec512d8b(img_size=224, in_chans=3)
    # Load pretrained weights, use MAE or RETFound
    chkpt_dir = "pretrained_weights_download_path/RETFound_oct_weights.pth"
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = mae_model.load_state_dict(checkpoint['model'], strict=False)
    print(f"Loaded checkpoint keys: {msg}")
    
    # Notice that this function must be called after loading pretrained-weights
    apply_dual_lora_to_block(mae_model, mae_model.lora_rank, mae_model.lora_alpha)

    # frozen all parameters
    for param in mae_model.parameters():
        param.requires_grad = False

    # Trainable Data LoRA and frozen Task LoRA
    set_lora_trainable_state(mae_model, 'MAE', True) 
    set_lora_trainable_state(mae_model, 'SEG', False)

    # Trainable original decoder
    for param in mae_model.decoder_embed.parameters(): param.requires_grad = True
    for block in mae_model.decoder_blocks:
        for param in block.parameters(): param.requires_grad = True
    for param in mae_model.decoder_norm.parameters(): param.requires_grad = True
    for param in mae_model.decoder_pred.parameters(): param.requires_grad = True

    mae_model = mae_model.cuda()

    if hydra_cfg.device == "cpu" or hydra_cfg.dist.gpus == 0:
        hydra_cfg.dist.gpus = 0
        train_loop(mae_model, 0, hydra_cfg)
    


if __name__ == "__main__":
    experiment = get_experiment()
    print('experiment: ', experiment)
    sys.argv.append(f'exp_name={experiment}')
    main()
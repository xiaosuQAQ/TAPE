import argparse
import datetime
import os
import random
import sys
import traceback

import hydra
import torch
from tqdm import tqdm
from utils.writer import Writer
from omegaconf import open_dict
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from datasets.dataset import DataloaderMode
from dataloader.dataloader import create_dataloader
from utils.utils import (get_logger, is_logging_process,
                         print_config, set_random_seed,
                         pixel_accuracy_no_background, mean_accuracy_no_background,
                         mean_dice_no_background, mean_iou_no_background)
import csv
import numpy as np
import torch.nn.functional as F
from model.MAE_fine_OCTA import mae_vit_large_patch16_dec512d8b
from model.MaeSegmodel import MAEEncoder, DualLoRALayer,\
    MAESegmentationModel,MAEEncoderWithGating, \
    apply_dual_lora_to_block, set_lora_trainable_state


def load_folder_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        folder_list = [line.strip() for line in f.readlines()]
    return folder_list

def setup(cfg, rank):
    # if your GPU is not from nvidia then please comment out this
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
    mp.spawn(fn, args=(cfg,), nprocs=cfg.dist.gpus, join=True)

def train_loop(rank, cfg):
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
        # set log/checkpoint dir
        os.makedirs(cfg.log.chkpt_dir, exist_ok=True)
        # set writer (tensorboard / wandb)
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

    # TODO: move all the argument inside the model class
    logger.info(cfg.model_obj)

    mae_model = mae_vit_large_patch16_dec512d8b(img_size=224, in_chans=3)
    # Load pretrained weights of original MAE or RETFound
    # 1. Replace to your download path
    chkpt_dir = "pretrained_weights_download_path/RETFound_oct_weights.pth"
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = mae_model.load_state_dict(checkpoint['model'], strict=False)
    print(f"Loaded checkpoint keys: {msg}")

    # 2. Add Domain LoRA and Task LoRA
    apply_dual_lora_to_block(mae_model, mae_model.lora_rank, mae_model.lora_alpha)

    # 3. Load weights from Domain LoRA in Stage I
    # Replace with your path
    chkpt_dir_OCTA_lora = "DA_check_point/mae_pretrain_weights_octa_ffn_61.pth"
    checkpoint_OCTA_lora = torch.load(chkpt_dir_OCTA_lora, map_location='cpu')

    msg = mae_model.load_state_dict(checkpoint_OCTA_lora, strict=False)
    print(f"Loaded Domain LoRA checkpoint keys: {msg}")

    # 4. Frozen all parameters
    for param in mae_model.parameters():
        param.requires_grad = False

    encoder = MAEEncoderWithGating(mae_model)
    # 5. Create segment model with foundation model's encoder
    net_arch = MAESegmentationModel(
        mae_encoder=encoder,
        num_classes=6,
        img_size=224,
        patch_size=16,
    ).to(cfg.device)

    # 6. Frozen Domain LoRA and trainable Task LoRA
    set_lora_trainable_state(net_arch, 'MAE', False)
    set_lora_trainable_state(net_arch, 'SEG', True)

    trainable_params = sum(p.numel() for p in net_arch.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in net_arch.parameters())
    trainable_param_names = [name for name, param in net_arch.named_parameters() if param.requires_grad]

    for name in trainable_param_names:
        print(f"'{name}',")

    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    logger.info(cfg.loss_obj)
    loss_f = hydra.utils.instantiate(cfg.loss_obj, cfg)

    logger.info(cfg.handler_obj)
    model = hydra.utils.instantiate(cfg.handler_obj, cfg, net_arch, loss_f, writer, rank)

    if cfg.load.resume_state_path is not None:
        model.load_training_state()
    elif cfg.load.network_chkpt_path is not None:
        model.load_network()
    else:
        if is_logging_process():
            logger.info("Starting new training run.")

    try:
        if cfg.dist.gpus == 0 or cfg.data.divide_dataset_per_gpu:
            epoch_step = 1
        else:
            epoch_step = cfg.dist.gpus
        # Train and validate
        for epoch in tqdm(range(model.epoch + 1, cfg.train.num_epoch,
                                epoch_step),
                          desc="Epoch",
                          unit='epoch'):
            model.epoch = epoch
            model.train_model(train_loader)
            model.save_training_state()
            model.validate_model(val_loader)
            model.log_ped_case(test_loader)
            if cfg.log.save_model_each_epoch:
                model.save_training_state()

        if is_logging_process():
            logger.info("End of Train")


        # diseases samples
        folder_list = load_folder_list('/inspire/hdd/global_user/lizizhen-240108540152/sxf/dataset/subfolders2.txt')
        folder_set = set(folder_list)
        # Replace with your save path
        save_path = 'test_results'
        unet = model.net
        unet = unet.to('cuda').eval()
        results = []
        save_batch = 1000   # Trigger writing after processing 1k pieces of data
        epoch_csv_dir = os.path.join(save_path, f"epoch{epoch}")
        os.makedirs(epoch_csv_dir, exist_ok=True)
        csv_file = os.path.join(epoch_csv_dir, "results.csv")
        if not os.path.exists(csv_file):
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["file_name", "PA", "mPA", "mIoU", "mDice"])

        with torch.no_grad():
            for input_oct, input_octa, label, _, _, idx, file_name in tqdm(test_loader):
                mean = unet(input_oct.to('cuda'), input_octa.to('cuda'))
                probs = F.softmax(mean, dim=1)
                pred = torch.argmax(probs, dim=1).cpu().numpy().squeeze()
                label = label.squeeze().numpy()
                # calculate scores
                pa = pixel_accuracy_no_background(pred, label)
                macc = mean_accuracy_no_background(pred, label, num_classes=6)
                miou = mean_iou_no_background(pred, label, num_classes=6)
                mdice = mean_dice_no_background(pred, label, num_classes=6)

                idx = idx.numpy()
                idx_s = idx[0]
                file_name_s = file_name[0]
                results.append([file_name_s, pa, macc, miou, mdice])

                if file_name_s in folder_set:
                    target_dir = os.path.join(save_path, str(file_name_s))
                    os.makedirs(target_dir, exist_ok=True)
                    np.save(os.path.join(target_dir, "pred_mask_{}.npy".format(file_name_s)), pred)
                    np.save(os.path.join(target_dir, "label_{}.npy".format(file_name_s)), label)

                if (idx_s + 1) % save_batch == 0:
                    try:
                        with open(csv_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerows(results)
                        results = []
                    except Exception as e:
                        print(f"Fail to save: {str(e)}")
            del mean, input
            torch.cuda.empty_cache()
        if len(results) != 0:
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(results)
        print("--------------------- Saved test results. ---------------------")
    except Exception as e:
        if is_logging_process():
            logger.error(traceback.format_exc())
            print(traceback.format_exc())
        else:
            traceback.print_exc()
    finally:
        if cfg.dist.gpus != 0:
            cleanup()


def get_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='default.yaml')
    args = parser.parse_args()
    return args.experiment


@hydra.main(version_base="1.1", config_path="config", config_name="default")
def main(hydra_cfg):
    hydra_cfg.device = hydra_cfg.device.lower()
    os.chdir("./TAPE")
    with open_dict(hydra_cfg):
        hydra_cfg.job_logging_cfg = HydraConfig.get().job_logging

    print_config(hydra_cfg, get_logger(hydra_cfg, os.path.basename(__file__),
                                       disable_console=True))
    if hydra_cfg.random_seed is None:
        hydra_cfg.random_seed = random.randint(1, 10000)
    set_random_seed(hydra_cfg.random_seed)

    if hydra_cfg.dist.gpus < 0:
        hydra_cfg.dist.gpus = torch.cuda.device_count()
    if hydra_cfg.device == "cpu" or hydra_cfg.dist.gpus == 0:
        hydra_cfg.dist.gpus = 0
        train_loop(0, hydra_cfg)
    else:
        hydra_cfg.work_dir = hydra_cfg.work_dir
        distributed_run(train_loop, hydra_cfg)


if __name__ == "__main__":
    experiment = get_experiment()
    print('experiment: ', experiment)
    sys.argv.append(f'exp_name={experiment}')
    main()



import os

import numpy as np
import torch
from einops import rearrange
from jbag.checkpoint import load_checkpoint, save_checkpoint
from jbag.config import get_config
from jbag.io import read_txt_2_list
from jbag.losses import DiceLoss
from jbag.lr_schedulers import poly_lr
from jbag.lr_utils import get_lr
from jbag.metrics import dsc
from jbag.samplers import GridSampler
from jbag.transforms import ZscoreNormalization, ToType, AddChannel
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm, trange

from MetricSummary import MetricSummary
from cfgs.args import parser
from dataset.get_boundary import get_boundary
from dataset.dataset import ImageDataset
from dataset.transforms import GetBoundary
from materials.meas_stds_const import BCA_MEAN, BCA_STD
from models import GANet


def main():
    def train_batch(input_batch):
        image = input_batch['data']
        tissue_gt = input_batch[tissue_label]
        region_gt = input_batch[region_label]
        tissue, region = model(image.to(device))
        tissue, region = torch.softmax(tissue, dim=1), torch.softmax(region, dim=1)
        tissue_loss = dice_criterion(tissue, tissue_gt.to(device))
        region_loss = dice_criterion(region, region_gt.to(device))
        return tissue_loss, region_loss

    @torch.no_grad()
    def infer_3d_volume(val_batch):
        val_image = val_batch['data']
        val_image = rearrange(val_image, 'b h w d -> (b d) h w')
        batch_size = cfg.val_batch_size
        batch_size = batch_size if batch_size < val_image.shape[1] else val_image.shape[1]
        patch_size = (batch_size, val_image.size(1), val_image.size(2))
        sampler = GridSampler(val_image, patch_size)

        output = []
        for patch in sampler:
            patch = patch.unsqueeze(dim=1)
            output_patch, _ = model(patch.to(device))
            output_patch = torch.argmax(output_patch, dim=1).to(torch.uint8).cpu()
            output.append(output_patch)
        output = sampler.restore(output)

        return output

    # Model
    model = GANet().to(device)
    optimizer = SGD(params=model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-3)

    grad_scaler = GradScaler()

    dice_criterion = DiceLoss(to_one_hot_y=True)

    tr_transforms = Compose([
        ToType(keys="data", dtype=np.float32),
        ZscoreNormalization(keys="data", mean_value=BCA_MEAN, std_value=BCA_STD, mean_key="mean", std_key="std"),
        AddChannel(keys=["data"], dim=0)
    ])
    training_image_dir = cfg.slice_sample_dir.image

    tissue_label = cfg.tissue_label
    region_label = cfg.region_label

    training_label_dict = {tissue_label: cfg.slice_sample_dir[tissue_label],
                           region_label: cfg.slice_sample_dir[region_label]}

    training_dataset = ImageDataset(data_indices=read_txt_2_list(cfg.training_slice_txt),
                                    raw_data_dir=training_image_dir,
                                    label_dir_dict=training_label_dict,
                                    transforms=tr_transforms)

    training_data_loader = DataLoader(dataset=training_dataset,
                                      batch_size=cfg.batch_size,
                                      shuffle=True,
                                      num_workers=2,
                                      pin_memory=True)

    # val dataset and data loader
    boundary_dict = get_boundary(cfg.boundary_file)
    val_transforms = Compose([
        ToType(keys="data", dtype=np.float32),
        ZscoreNormalization(keys="data", mean_value=BCA_MEAN, std_value=BCA_STD, mean_key="mean", std_key="std"),
        GetBoundary(keys=["data", tissue_label], boundary_dict=boundary_dict)
    ])
    val_samples = read_txt_2_list(cfg.val_ct_txt)
    volume_image_dir = cfg.volume_sample_dir.image
    val_label_dict = {tissue_label: cfg.volume_sample_dir[tissue_label]}
    val_dataset = ImageDataset(data_indices=val_samples,
                               raw_data_dir=volume_image_dir,
                               label_dir_dict=val_label_dict,
                               add_postfix=True,
                               transforms=val_transforms)
    val_data_loader = DataLoader(val_dataset, 1)

    best_val_dice = 0
    iteration = 0

    snapshot_dir = cfg.snapshot
    os.makedirs(snapshot_dir, exist_ok=True)
    checkpoint_dir = os.path.join(snapshot_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    vis_log_path = os.path.join(snapshot_dir, 'log')
    vis_log = SummaryWriter(vis_log_path)

    initial_epoch = cfg.initial_epoch if "initial_epoch" in cfg else 0
    # Load checkpoint
    if "checkpoint" in cfg and cfg.checkpoint is not None:
        load_checkpoint(cfg.checkpoint, model)

    # Train
    dice_metric = MetricSummary()
    for epoch in trange(initial_epoch, cfg.epochs):
        model.train()
        poly_lr(optimizer=optimizer, initial_lr=cfg.lr, epoch=epoch, num_epochs=cfg.epochs)
        vis_log.add_scalar("lr", get_lr(optimizer), epoch)
        for batch in tqdm(training_data_loader):
            optimizer.zero_grad()
            with autocast():
                tissue_loss, region_loss = train_batch(batch)
                loss = tissue_loss + region_loss
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            vis_log.add_scalar('loss/tissue', tissue_loss.detach().cpu(), iteration)
            vis_log.add_scalar('loss/region', region_loss.detach().cpu(), iteration)
            iteration += 1

        # validate epoch
        val_interval = cfg.val_interval if "val_interval" in cfg else 1
        if (epoch != 0 and epoch % val_interval == 0) or (epoch == cfg.epochs - 1):
            model.eval()
            val_dice_scores = []
            for val_batch in tqdm(val_data_loader):
                with autocast():
                    val_result = infer_3d_volume(val_batch)
                val_result = rearrange(val_result, "d h w -> 1 h w d")
                target = val_batch[tissue_label]
                val_dice_score = dsc(val_result, target)
                val_dice_scores.append(val_dice_score)

            val_dice_score = torch.stack(val_dice_scores).mean()

            vis_log.add_scalar('val dice', val_dice_score, epoch)
            if val_dice_score >= best_val_dice:
                best_val_dice = val_dice_score
                save_checkpoint(os.path.join(snapshot_dir, 'best_val_checkpoint.pt'), model)

                vis_log.add_scalar('snapshot/epoch', epoch, epoch)
                vis_log.add_scalar('snapshot/dice', best_val_dice, epoch)


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = get_config(args.cfg)
    if args.gpus:
        cfg.gpus = args.gpus
    if 'gpus' in cfg:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cfg.gpus)
    device = 'cpu'
    if torch.cuda.is_available() and 'gpus' in cfg:
        device = torch.device('cuda')
        if 'cudnn' in cfg:
            cudnn.benchmark = cfg.cudnn.benchmark
            cudnn.deterministic = cfg.cudnn.deterministic
    main()

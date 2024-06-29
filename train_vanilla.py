import os

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
from einops import rearrange
from jbag.checkpoint import load_checkpoint, save_checkpoint
from jbag.config import get_config
from jbag.io import read_txt_2_list
from jbag.models import UNetPlusPlus
from jbag.samplers import GridSampler
from jbag.transforms import ZscoreNormalization, ToType, AddChannel
from medpy.metric import dc
from monai.losses import DiceLoss
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose
from tqdm import tqdm, trange

from cfgs.args import parser
from dataset.image_dataset import JSONImageDataset
from materials.meas_stds_const import WDS_MEAN, WDS_STD
from jbag import logger
from jbag import MetricSummary

def main():
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
            output_patch = model(patch.to(device))
            output_patch = torch.argmax(output_patch, dim=1).to(torch.uint8).cpu()
            output.append(output_patch)
        output = sampler.restore(output)

        return output

    # Model
    model = UNetPlusPlus(in_channels=1, out_channels=cfg.n_classes).to(device)
    if is_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])
    grad_scaler = GradScaler()

    optimizer = SGD(params=model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-3)
    poly_lr = PolynomialLR(optimizer, total_iters=cfg.epochs, power=0.9)

    # training dataset and data loader
    label = cfg.label
    training_samples = read_txt_2_list(cfg.training_slice_txt)
    training_image_dir = cfg.slice_sample_dir.image
    training_label_dict = {label: cfg.slice_sample_dir[label]}

    tr_transforms = Compose([
        ToType(keys='data', dtype=np.float32),
        ZscoreNormalization(keys='data', mean_value=WDS_MEAN, std_value=WDS_STD, mean_key='mean', std_key='std'),
        AddChannel(keys=['data', label], dim=0)
    ])

    training_dataset = JSONImageDataset(sample_list=training_samples,
                                        sample_dir=training_image_dir,
                                        label_dict=training_label_dict,
                                        transforms=tr_transforms)

    if is_distributed:
        training_data_sampler = DistributedSampler(training_dataset)
        shuffle = False
    else:
        training_data_sampler = None
        shuffle = True

    training_data_loader = DataLoader(dataset=training_dataset,
                                      batch_size=cfg.batch_size,
                                      shuffle=shuffle,
                                      num_workers=2,
                                      pin_memory=True,
                                      sampler=training_data_sampler)

    # val dataset and data loader
    val_samples = read_txt_2_list(cfg.val_ct_txt)
    volume_image_dir = cfg.volume_sample_dir.image
    val_label_dict = {label: cfg.volume_sample_dir[label]}

    # boundary_dict = get_boundary(cfg.boundary_file)
    val_transforms = Compose([
        ToType(keys='data', dtype=np.float32),
        ZscoreNormalization(keys='data', mean_value=WDS_MEAN, std_value=WDS_STD, mean_key='mean', std_key='std'),
        # GetBoundary(keys=['data', label], boundary_dict=boundary_dict)
    ])
    val_dataset = JSONImageDataset(sample_list=val_samples,
                                   sample_dir=volume_image_dir,
                                   label_dict=val_label_dict,
                                   add_postfix=True,
                                   transforms=val_transforms)

    val_data_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    val_data_loader = DataLoader(val_dataset, 1, sampler=val_data_sampler)

    best_val_dice = 0
    iteration = 0

    if is_master:
        snapshot_dir = cfg.snapshot
        os.makedirs(snapshot_dir, exist_ok=True)
        checkpoint_dir = os.path.join(snapshot_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        vis_log_path = os.path.join(snapshot_dir, 'log')
        vis_log = SummaryWriter(vis_log_path)

    # Load checkpoint
    if 'checkpoint' in cfg and cfg.checkpoint:
        load_checkpoint(cfg.checkpoint, model)

    # Train
    dice_criterion = DiceLoss(to_onehot_y=True, softmax=True)
    dice_metric = MetricSummary(metric_fn=dc)
    val_interval = cfg.val_interval if 'val_interval' in cfg else 1
    epoch_bar = trange(0, cfg.epochs, postfix={'rank': world_rank})
    for epoch in epoch_bar:
        epoch_bar.postfix = f'epoch: {epoch}'
        model.train()

        batch_bar = tqdm(training_data_loader, postfix={'rank': world_rank})
        for batch in batch_bar:
            optimizer.zero_grad()
            image = batch['data']
            gt_label = batch[label]
            with autocast():
                output = model(image.to(device))
                loss = dice_criterion(output, gt_label.to(device))
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            loss_scalar = loss.detach().cpu()
            if is_master:
                vis_log.add_scalar('loss', loss_scalar, iteration)
            iteration += 1
            batch_bar.postfix = f'loss: {loss_scalar:.5f} epoch: {epoch}  rank: {world_rank}'

        poly_lr.step()
        if is_master:
            vis_log.add_scalar('lr', poly_lr.get_last_lr(), epoch)

        # validate epoch
        if epoch % val_interval == 0:
            model.eval()
            dice_metric.reset()
            for val_batch in tqdm(val_data_loader):
                with autocast():
                    val_result = infer_3d_volume(val_batch)
                val_result = rearrange(val_result, 'd h w -> 1 h w d')
                target = val_batch[label]

                val_result = val_result.squeeze(0).cpu().numpy()
                target = target.squeeze(0).cpu().numpy()

                dice_metric.update(val_result, target)

            mean_val_dice_score = dice_metric.mean()
            if is_distributed:
                mean_val_dice_score = torch.tensor(mean_val_dice_score, device=device)
                dist.all_reduce(mean_val_dice_score, op=ReduceOp.AVG)
            if is_master:
                vis_log.add_scalar('val dice', mean_val_dice_score, epoch)
                if mean_val_dice_score >= best_val_dice:
                    best_val_dice = mean_val_dice_score
                    save_checkpoint(os.path.join(snapshot_dir, 'best_val_checkpoint.pt'), model)

                    vis_log.add_scalar('snapshot/epoch', epoch, epoch)
                    vis_log.add_scalar('snapshot/dice', best_val_dice, epoch)
        dist.barrier()


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = get_config(args.cfg)

    if args.gpus:
        cfg.gpus = args.gpus
    if 'gpus' in cfg:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cfg.gpus)
    if 'cudnn' in cfg:
        cudnn.benchmark = cfg.cudnn.benchmark
        cudnn.deterministic = cfg.cudnn.deterministic

    is_distributed = cfg.is_distributed if 'is_distributed' in cfg else False
    if is_distributed:
        if len(cfg.gpus) == 1:
            logger.warning('*****************************************')
            logger.warning('Program is running on distributed mode but with ONLY 1 GPU acquired.')
            logger.warning('*****************************************')

        local_rank = int(os.environ['LOCAL_RANK'])
        world_rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl', rank=world_rank, world_size=world_size)

        device = torch.device(f'cuda:{str(local_rank)}')
        logger.info('*****************************************')
        logger.info(f'Program is running on distributed mode with WORLD SIZE of {world_size}.')
        logger.info(f'Current process LOCAL RANK IS: {local_rank}, GLOBAL RANK IS: {world_rank}.')
        logger.info('*****************************************')
    else:
        device = torch.device('cuda')
        world_rank = 0
    is_master = world_rank == 0
    main()

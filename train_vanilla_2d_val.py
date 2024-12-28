import os

import numpy as np
import torch
import torch.distributed as dist
from jbag import MetricSummary
from jbag import logger
from jbag.config import load_config
from jbag.io import read_json
from jbag.model_weights import load_weights, save_weights
from jbag.models.deep_supervision import DeepSupervisionLossWrapper, get_deep_supervision_loss_weights, \
    set_deep_supervision, get_deep_supervision_scales
from jbag.transforms import ToType, AddChannel, ToTensor
from jbag.transforms.brightness import MultiplicativeBrightnessTransform
from jbag.transforms.contrast import ContrastTransform
from jbag.transforms.downsample import DownsampleTransform
from jbag.transforms.gamma import GammaTransform
from jbag.transforms.gaussian_blur import GaussianBlurTransform
from jbag.transforms.gaussian_noise import GaussianNoiseTransform
from jbag.transforms.normalization import ZscoreNormalization
from jbag.transforms.spatial import SpatialTransform
from medpy.metric import dc
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.distributed import ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose
from tqdm import tqdm, trange

from cfgs.args import parser
from dataset.dataset import ImageDataset
from loss_criterions import get_loss_criterion
from models import model_zoo


def main():
    # Model
    network: torch.nn.Module = model_zoo[cfg.network.architecture](cfg.network)
    network.to(device)

    if is_ddp:
        network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
        network = DDP(network, device_ids=[local_rank])

    dataset_properties = read_json(cfg.dataset_property_file)
    val_test_transforms = Compose([
        ToType(keys='data', dtype=np.float32),
        ZscoreNormalization(keys='data', mean=dataset_properties['intensity_mean'],
                            std=dataset_properties['intensity_std'],
                            lower_bound=dataset_properties['intensity_0_5_percentile'],
                            upper_bound=dataset_properties['intensity_99_5_percentile']),
        AddChannel(keys=['data'], axis=0)
    ])

    log_dir = cfg.snapshot
    os.makedirs(log_dir, exist_ok=True)
    if is_master:
        vis_log_path = os.path.join(log_dir, 'log')
        vis_log = SummaryWriter(vis_log_path)

    def train():
        grad_scaler = GradScaler()
        optimizer = SGD(params=network.parameters(), lr=cfg.lr, momentum=0.99, weight_decay=3e-5)
        poly_lr = PolynomialLR(optimizer, total_iters=cfg.epochs, power=0.9)

        # Load pre-trained model
        if 'checkpoint' in cfg and cfg.checkpoint:
            load_weights(cfg.checkpoint, network)

        # dataloader
        training_samples = dataset_properties['training_dataset']
        image_dir = cfg.slice_sample_dir.image
        label_dir_dict = {cfg.label: cfg.slice_sample_dir[cfg.label]}

        tr_transforms = [
            ToType(keys='data', dtype=np.float32),
            ZscoreNormalization(keys='data', mean=dataset_properties['intensity_mean'],
                                std=dataset_properties['intensity_std'],
                                lower_bound=dataset_properties['intensity_0_5_percentile'],
                                upper_bound=dataset_properties['intensity_99_5_percentile']),
            AddChannel(keys=['data', cfg.label], axis=0),
            ToTensor(keys=['data', cfg.label]),
            SpatialTransform(keys=['data', cfg.label], apply_probability=1,
                             patch_size=cfg.training_data_augments.spatial_transform.patch_size,
                             patch_center_dist_from_border=cfg.training_data_augments.spatial_transform.patch_center_dist_from_border,
                             random_crop=cfg.training_data_augments.spatial_transform.random_crop,
                             interpolation_modes=['bilinear', 'nearest'],
                             p_rotation=cfg.training_data_augments.spatial_transform.p_rotation,
                             rotation_angle_range=cfg.training_data_augments.spatial_transform.rotation,
                             p_scaling=cfg.training_data_augments.spatial_transform.p_scaling,
                             scaling_range=cfg.training_data_augments.spatial_transform.scaling,
                             p_synchronize_scaling_across_axes=cfg.training_data_augments.spatial_transform.p_synchronize_scaling_across_axes
                             ),
            GaussianNoiseTransform(keys=['data'],
                                   apply_probability=cfg.training_data_augments.gaussian_noise_transform.p,
                                   noise_variance=cfg.training_data_augments.gaussian_noise_transform.noise_variance,
                                   synchronize_channels=cfg.training_data_augments.gaussian_noise_transform.synchronize_channels,
                                   p_per_channel=cfg.training_data_augments.gaussian_noise_transform.p_per_channel
                                   ),
            GaussianBlurTransform(keys=['data'],
                                  apply_probability=cfg.training_data_augments.gaussian_blur_transform.p,
                                  blur_sigma=cfg.training_data_augments.gaussian_blur_transform.blur_sigma,
                                  synchronize_channels=cfg.training_data_augments.gaussian_blur_transform.synchronize_channels,
                                  synchronize_axes=cfg.training_data_augments.gaussian_blur_transform.synchronize_axes,
                                  p_per_channel=cfg.training_data_augments.gaussian_blur_transform.p_per_channel
                                  ),
            MultiplicativeBrightnessTransform(keys=['data'],
                                              apply_probability=cfg.training_data_augments.brightness_transform.p,
                                              multiplier_range=cfg.training_data_augments.brightness_transform.multiplier_range,
                                              synchronize_channels=cfg.training_data_augments.brightness_transform.synchronize_channels,
                                              p_per_channel=cfg.training_data_augments.brightness_transform.p_per_channel
                                              ),
            ContrastTransform(keys=['data'],
                              apply_probability=cfg.training_data_augments.contrast_transform.p,
                              contrast_range=cfg.training_data_augments.contrast_transform.contrast_range,
                              preserve_range=cfg.training_data_augments.contrast_transform.preserve_range,
                              synchronize_channels=cfg.training_data_augments.contrast_transform.synchronize_channels,
                              p_per_channel=cfg.training_data_augments.contrast_transform.p_per_channel
                              ),
            GammaTransform(keys=['data'],
                           apply_probability=cfg.training_data_augments.gamma_transform1.p,
                           gamma=cfg.training_data_augments.gamma_transform1.gamma,
                           p_invert_image=cfg.training_data_augments.gamma_transform1.p_invert_image,
                           synchronize_channels=cfg.training_data_augments.gamma_transform1.synchronize_channels,
                           p_per_channel=cfg.training_data_augments.gamma_transform1.p_per_channel,
                           p_retain_stats=cfg.training_data_augments.gamma_transform1.p_retain_stats
                           ),
            GammaTransform(keys=['data'],
                           apply_probability=cfg.training_data_augments.gamma_transform2.p,
                           gamma=cfg.training_data_augments.gamma_transform2.gamma,
                           p_invert_image=cfg.training_data_augments.gamma_transform2.p_invert_image,
                           synchronize_channels=cfg.training_data_augments.gamma_transform2.synchronize_channels,
                           p_per_channel=cfg.training_data_augments.gamma_transform2.p_per_channel,
                           p_retain_stats=cfg.training_data_augments.gamma_transform2.p_retain_stats
                           ),
        ]

        deep_supervision = 'deep_supervision' in cfg.network and cfg.network.deep_supervision
        if deep_supervision:
            tr_transforms.append(DownsampleTransform(keys=[cfg.label], scales=get_deep_supervision_scales(cfg)))

        tr_transforms = Compose(tr_transforms)

        training_dataset = ImageDataset(data_indices=training_samples,
                                        raw_data_dir=image_dir,
                                        label_dir_dict=label_dir_dict,
                                        add_postfix=False,
                                        transforms=tr_transforms)

        if is_ddp:
            training_data_sampler = DistributedSampler(training_dataset)
            shuffle = False
        else:
            training_data_sampler = None
            shuffle = True

        training_data_loader = DataLoader(dataset=training_dataset,
                                          batch_size=cfg.batch_size,
                                          shuffle=shuffle,
                                          num_workers=4,
                                          pin_memory=True,
                                          sampler=training_data_sampler)

        training_data_iterator = iter(training_data_loader)

        # val dataset and data loader
        val_samples = dataset_properties['val_dataset']

        val_dataset = ImageDataset(data_indices=val_samples,
                                   raw_data_dir=image_dir,
                                   label_dir_dict=label_dir_dict,
                                   add_postfix=False,
                                   transforms=val_test_transforms)

        val_data_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None
        val_data_loader = DataLoader(val_dataset, cfg.val_batch_size, sampler=val_data_sampler, num_workers=4,
                                     pin_memory=True)

        best_val_dice = 0
        iteration = 0

        # Train
        loss_criterion: torch.nn.Module = get_loss_criterion(cfg.loss_criterion)(to_onehot_y=True, softmax=True)
        if deep_supervision:
            loss_weights = get_deep_supervision_loss_weights(cfg)
            loss_criterion = DeepSupervisionLossWrapper(loss_criterion, loss_weights)
        dice_metric = MetricSummary(metric_fn=dc)
        val_interval = cfg.val_interval if 'val_interval' in cfg else 1
        epoch_bar = trange(0, cfg.epochs, postfix={'rank': world_rank})
        loss_summary = MetricSummary()
        for epoch in epoch_bar:
            loss_summary.reset()
            epoch_bar.postfix = f'epoch: {epoch}'
            if is_master:
                vis_log.add_scalar('lr', poly_lr.get_last_lr()[0], epoch)

            network.train()
            # insure deep supervision
            if deep_supervision:
                set_deep_supervision(network, True)
            batch_bar = trange(0, cfg.n_iter_per_epoch, postfix={'rank': world_rank})
            for _ in batch_bar:
                try:
                    batch = next(training_data_iterator)
                except StopIteration:
                    training_data_iterator = iter(training_data_loader)
                    batch = next(training_data_iterator)

                optimizer.zero_grad()
                image = batch['data']
                gt_label = batch[cfg.label]
                with autocast():
                    output = network(image.to(device))
                    if deep_supervision:
                        gt_label = [each.to(device) for each in gt_label]
                    else:
                        gt_label = gt_label.to(device)
                    loss = loss_criterion(output, gt_label)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(network.parameters(), 12)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                loss_scalar = loss.detach().cpu()
                loss_summary.add_value(loss_scalar)
                if is_master:
                    vis_log.add_scalar('loss', loss_scalar, iteration)
                iteration += 1
                batch_bar.postfix = f'epoch: {epoch} rank: {world_rank} loss: {loss_summary.mean():.5f}'

            poly_lr.step()
            # validate epoch
            if epoch % val_interval == 0:
                network.eval()
                set_deep_supervision(network, False)
                dice_metric.reset()
                val_bar = tqdm(val_data_loader, postfix={'epoch': epoch, 'rank': world_rank})
                with torch.no_grad():
                    for batch in val_bar:
                        image = batch['data']
                        with autocast():
                            output = network(image.to(device))
                        output = torch.argmax(output, dim=1).cpu().numpy()
                        target = batch[cfg.label].numpy()
                        dice_metric(output, target)
                        val_bar.postfix = f'epoch: {epoch} rank: {world_rank} dice: {dice_metric.mean()}'

                mean_val_dice_score = dice_metric.mean()
                if is_ddp:
                    mean_val_dice_score = torch.tensor(mean_val_dice_score, device=device)
                    dist.all_reduce(mean_val_dice_score, op=ReduceOp.AVG)
                if is_master:
                    vis_log.add_scalar('val dice', mean_val_dice_score, epoch)
                    if mean_val_dice_score >= best_val_dice:
                        best_val_dice = mean_val_dice_score
                        save_weights(os.path.join(log_dir, 'best_val_checkpoint.pt'), network)

                        vis_log.add_scalar('snapshot/epoch', epoch, epoch)
                        vis_log.add_scalar('snapshot/dice', best_val_dice, epoch)
            if is_ddp:
                dist.barrier()

            if epoch != 0 and epoch % cfg.checkpoint_saved_interval == 0:
                saved_parameters = {'epoch': epoch, 'grad_scaler': grad_scaler.state_dict(),
                                    'lr_scheduler': poly_lr.state_dict()}
                if is_master:
                    saved_parameters['best_val_dice'] = best_val_dice
                save_weights(os.path.join(log_dir, 'checkpoint', f'checkpoint_{epoch}.pth'), network, optimizer,
                             **saved_parameters)

    if cfg.train:
        train()

    if is_master:
        vis_log.close()


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = load_config(args.cfg)

    if args.gpus:
        cfg.gpus = args.gpus
    if 'gpus' in cfg:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cfg.gpus)
    if 'cudnn' in cfg:
        cudnn.benchmark = cfg.cudnn.benchmark
        cudnn.deterministic = cfg.cudnn.deterministic

    is_ddp = cfg.is_ddp if 'is_ddp' in cfg else False
    if is_ddp:
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

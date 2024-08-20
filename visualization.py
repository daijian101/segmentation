import os

import torch
from cavass.ops import read_cavass_file
from cavass.window_transform import cavass_soft_tissue_window
from einops import rearrange
from jbag.image import overlay
from tensorboardX import SummaryWriter
from tqdm import tqdm


def get_tensorboard_3d_images(image, mask):
    overlaid_image = overlay(image, mask)
    overlaid_image = torch.from_numpy(overlaid_image)
    overlaid_image = rearrange(overlaid_image, 'h w d c -> d c h w')
    return overlaid_image


if __name__ == '__main__':
    bim_data_path = '/data1/dj/running/bca_Dphm_dice_focal_test/test_result'
    im0_data_path = '/data1/dj/data/bca/cavass_data/images'

    samples = [each for each in os.listdir(bim_data_path) if each.endswith('.BIM')]

    writer = SummaryWriter('/data1/dj/tmp/dphm')

    for each in tqdm(samples):
        subject_name = each[:-9]
        bim_file = os.path.join(bim_data_path, each)
        im0_path = os.path.join(im0_data_path, subject_name + '.IM0')

        image_data = read_cavass_file(im0_path)

        image_data = cavass_soft_tissue_window(image_data)
        mask_data = read_cavass_file(bim_file)

        writer.add_images(subject_name, get_tensorboard_3d_images(image_data, mask_data))

    writer.close()

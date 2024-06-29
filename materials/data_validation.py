import os
import random
from cavass.ops import read_cavass_file
from cavass.window_transform import cavass_soft_tissue_window
from jbag.image import overlay
from tensorboardX import SummaryWriter
import torch
from einops import rearrange
bim_data_path = '/home/deserts/sda3.7T/dj/data/bca/cavass_data/Dphm'
im0_data_path = '/home/deserts/sda3.7T/dj/data/bca/cavass_data/images'
samples = [each[:-4] for each in os.listdir(bim_data_path) if each.endswith('.BIM')]
random.shuffle(samples)


samples = samples[:10]
writer = SummaryWriter('/home/deserts/sda3.7T/dj/tmp')
for subject_name in samples:
    bim_file = os.path.join(bim_data_path, subject_name + '.BIM')
    im0_path = os.path.join(im0_data_path, subject_name + '.IM0')

    image_data = read_cavass_file(im0_path)

    image_data = cavass_soft_tissue_window(image_data)
    mask_data = read_cavass_file(bim_file)
    image = overlay(image_data, mask_data)
    image = torch.from_numpy(image)
    image = rearrange(image, 'h w d c -> d c h w')
    writer.add_images(subject_name, image)

writer.close()

import os

from jbag.image import overlay
from jbag.io import read_json

from cavass.window_transform import cavass_soft_tissue_window
import numpy as np
from tensorboardX import SummaryWriter

labels = ['SAT', 'VAT', 'SMT', 'Sk', 'Dphm']

image_path = '/data1/dj/data/bca/json/slices/images'

writer = SummaryWriter('/data1/dj/tmp/training')
for label in labels:
    fg_count = 0
    bg_count = 0
    label_path = os.path.join('/data1/dj/data/bca/json/slices/', label)
    all_files = os.listdir(label_path)
    for each_file in all_files:
        if bg_count > 10 and fg_count > 10:
            break
        mask = read_json(os.path.join(label_path, each_file))['data']
        if np.sum(mask) == 0:
            bg_count += 1
            if bg_count > 10:
                continue
            targ = f'{label}_bg_{bg_count}'

        else:
            fg_count += 1
            if fg_count > 10:
                continue
            targ = f'{label}_fg_{fg_count}'
        image_slice = read_json(os.path.join(image_path, each_file))['data']
        image = cavass_soft_tissue_window(image_slice)
        overlaid_image = overlay(image, mask)
        writer.add_image(targ, overlaid_image)

writer.close()
import os

import numpy as np
import torch
from cavass.ops import read_cavass_file
from cavass.window_transform import cavass_soft_tissue_window
from einops import rearrange
from jbag.config import get_config
from jbag.image import overlay
from jbag.io import read_txt_2_list
from tensorboardX import SummaryWriter
from tqdm import tqdm

from cfgs.args import parser


def main():
    im0_path = cfg.IM0_path
    if 'inference_samples' in cfg:
        if isinstance(cfg.inference_samples, str):
            samples = read_txt_2_list(cfg.inference_samples)
        else:
            samples = cfg.inference_samples
    else:
        samples = [each[:-4] for each in os.listdir(im0_path) if each.endswith('.IM0')]

    samples = np.random.choice(samples, 10).tolist()
    labels = cfg.inference_labels
    segmentation_path = cfg.result_cavass_path

    for sample in tqdm(samples):
        im0_file = os.path.join(im0_path, f'{sample}.IM0')
        image = read_cavass_file(im0_file)
        image = cavass_soft_tissue_window(image)
        for label in labels:
            label_path = os.path.join(segmentation_path, label, f'{sample}_{label}.BIM')
            label_data = read_cavass_file(label_path)
            overlaid_image = overlay(image, label_data)
            overlaid_image = torch.from_numpy(overlaid_image)
            overlaid_image = rearrange(overlaid_image, "h w d c -> d c h w")
            writer.add_images(f"{sample}/{label}", overlaid_image)

    pass


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = get_config(args.cfg)
    writer = SummaryWriter("/data/dj/tmp/validation")
    main()
    writer.close()

# Generate SR region and refine SAT with SR region.

import os

import numpy as np
from cavass.ops import read_cavass_file, save_cavass_file
from jbag.io import read_txt_2_list
from tqdm import tqdm

im0_path = '/data/dj/data/bca/cavass_data/images'

target_path = '/data/dj/data/bca/cavass_data/SR'

oam_path = '/data/dj/data/bca/cavass_data/OAM'
skn_path = '/data/dj/data/bca/cavass_data/Skn'
sat_path = '/data/dj/data/bca/cavass_data/SAT'

cts = read_txt_2_list("/data/dj/data/bca/dataset/all_SAT.txt")
for ct in tqdm(cts):

    sr_file = os.path.join(target_path, f"{ct}.BIM")
    if os.path.exists(sr_file):
        continue
    im0_file = os.path.join(im0_path, f"{ct}.IM0")
    skn = read_cavass_file(os.path.join(skn_path, f"{ct}.BIM")).astype(int)
    oam = read_cavass_file(os.path.join(oam_path, f"{ct}.BIM")).astype(int)

    sr = skn - oam
    sr = np.where(sr == 1, sr, 0)
    sr = sr.astype(bool)
    save_cavass_file(sr_file, sr, binary=True, reference_file=im0_file)

    # sat_file = os.path.join(sat_path, f"{ct}.BIM")
    # sat = read_cavass_file(sat_file)
    # sat = np.logical_and(sat, sr)
    # os.remove(sat_file)
    # save_cavass_file(sat_file, sat, binary=True, reference_file=im0_file)

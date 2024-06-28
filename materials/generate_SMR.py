# Generate SR region and refine SAT with SR region.

import os

import numpy as np
from cavass.ops import read_cavass_file, save_cavass_file
from jbag.io import read_txt_2_list, scp
from tqdm import tqdm

im0_path = '/data1/dj/data/bca/cavass_data/images'

target_path = '/data1/dj/data/bca/cavass_data/SMR'

oam_path = '/data1/dj/data/bca/cavass_data/OAM'
iam_path = '/data1/dj/data/bca/cavass_data/IAM'
smt_path = '/data1/dj/data/bca/cavass_data/SMT'

cts = read_txt_2_list('/data1/dj/data/bca/dataset/all_Sk.txt')
for ct in tqdm(cts):
    smr_file = os.path.join(target_path, f"{ct}.BIM")
    if os.path.exists(smr_file):
        continue
    im0_file = os.path.join(im0_path, f"{ct}.IM0")

    oam_file = os.path.join(oam_path, f"{ct}.BIM")
    iam_file = os.path.join(iam_path, f"{ct}.BIM")

    if not os.path.exists(oam_file):
        print(f"Transmit OAM {ct}")
        scp(dst_user="ubuntu", dst_host="10.21.22.70", dst_path=f"/data/dj/data/bca/cavass_data/OAM/{ct}.BIM",
            local_path=oam_path, dst_port=6202, receive=True)

    if not os.path.exists(iam_file):
        print(f"Transmit IAM {ct}")
        scp(dst_user="ubuntu", dst_host="10.21.22.70", dst_path=f"/data/dj/data/bca/cavass_data/IAM/{ct}.BIM",
            local_path=iam_path, dst_port=6202, receive=True)

    oam = read_cavass_file(os.path.join(oam_path, f"{ct}.BIM")).astype(int)
    iam = read_cavass_file(os.path.join(iam_path, f"{ct}.BIM")).astype(int)

    smr = oam - iam
    smr = np.where(smr == 1, smr, 0)
    smr = smr.astype(bool)
    save_cavass_file(smr_file, smr, binary=True, reference_file=im0_file)

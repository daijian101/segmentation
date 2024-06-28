import os
import shutil

from jbag.io import read_txt_2_list
from tqdm import tqdm

skn_path = '/home/ubuntu/sda/BCA/core/IM0'
wds_path = '/home/ubuntu/sda/BCA/W-DS1'
target_path = '/data/dj/data/bca/cavass_data'

labels = ['Dphm']
target_image_path = os.path.join(target_path, 'images')

os.makedirs(target_image_path, exist_ok=True)

for ct in tqdm(read_txt_2_list('/data/dj/data/bca/dataset/Dphm_val_cts.txt')):
    if ct.startswith('SKN'):
        source_image_file = f'{skn_path}/{ct}/CT.IM0'
    else:
        source_image_file = f"{wds_path}/{ct}/{ct}.IM0"

    target_image_file = os.path.join(target_image_path, f"{ct}.IM0")
    if not os.path.exists(target_image_file):
        shutil.copy(source_image_file, target_image_file)

    for label in labels:
        if ct.startswith("SKN"):
            source_label_file = f"{skn_path}/{ct}/{ct}_{label}.BIM"
        else:
            source_label_file = f"{wds_path}/{ct}/{ct}_{label}.BIM"

        target_label_path = os.path.join(target_path, label)
        os.makedirs(target_label_path, exist_ok=True)
        target_label_file = os.path.join(target_label_path, f"{ct}.BIM")
        if not os.path.exists(target_label_file):
            if os.path.exists(source_label_file):
                shutil.copy(source_label_file, target_label_file)
            else:
                print(f"File not found {source_label_file}")

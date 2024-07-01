import cavass
import os

from tqdm import tqdm

image_path = '/data1/dj/data/bca/cavass_data/images'
label_path = '/data1/dj/data/bca/cavass_data/Dphm'
nnunet_data_path = '/data1/dj/nnUNet_raw/Dataset001_Dphm'
os.makedirs(nnunet_data_path, exist_ok=True)

for each in tqdm(os.listdir(label_path)):
    if not each.endswith('.BIM'):
        continue

    ct_name = each[:-4]
    im0_file = os.path.join(image_path, ct_name + '.IM0')
    bim_file = os.path.join(label_path, ct_name + '.BIM')

    target_image_file = os.path.join(nnunet_data_path, '/data1/dj/data/bca/cavass_data/Dphm')


import os

from cavass.ops import read_cavass_file
from jbag.io import read_txt_2_list, scp
from jbag.io import save_json
from tqdm import tqdm

data_path = '/data1/dj/data/bca/'

image_root_path = os.path.join(data_path, 'cavass_data')
im0_path = os.path.join(image_root_path, 'images')

ct_saved_image_path = os.path.join(data_path, 'json/volume/images')
label_saved_root_path = os.path.join(data_path, 'json/volume')

label = 'SMT'

labels = [label]
cts = read_txt_2_list(os.path.join(data_path, f'dataset/{label}_val_cts.txt'))

for ct in tqdm(cts):
    ct_saved_file_path = os.path.join(ct_saved_image_path, f'{ct}.json')
    if not os.path.exists(ct_saved_file_path):
        im0_file = os.path.join(im0_path, f'{ct}.IM0')
        im0_data = read_cavass_file(im0_file)
        image_data = {'data': im0_data,
                      'subject': ct}
        save_json(ct_saved_file_path, image_data)

if labels:
    for ct in tqdm(cts):
        for label in labels:
            label_saved_file_path = os.path.join(label_saved_root_path, label, f'{ct}.json')
            if not os.path.exists(label_saved_file_path):
                label_file = os.path.join(image_root_path, label, f'{ct}.BIM')
                if not os.path.exists(label_file):
                    scp(dst_user='ubuntu', dst_host='10.21.22.70',
                        dst_path=f'/data/dj/data/bca/cavass_data/{label}/{ct}.BIM',
                        local_path=os.path.join(image_root_path, label), dst_port=6202, receive=True)

                label_data = read_cavass_file(label_file)
                data = {'data': label_data, 'subject': ct, 'class': label}
                save_json(label_saved_file_path, data)

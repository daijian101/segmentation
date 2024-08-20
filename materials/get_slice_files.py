import os
import sys

sys.path.append('../')

from cavass.ops import read_cavass_file, get_image_resolution
from jbag.io import read_txt_2_list, scp
from jbag.io import save_json
from tqdm import tqdm

image_root_path = '/data/dj/data/bca/cavass_data'

im0_path = os.path.join(image_root_path, 'images')
ct_saved_image_path = '/data/dj/data/bca/json/slices/images'
label_saved_root_path = '/data/dj/data/bca/json/slices'

label = 'Skn'
labels = [label]
# boundary_dict = get_boundary(boundary_file)

cts = read_txt_2_list(f'/data/dj/data/bca/dataset/{label}_training_cts.txt')
for ct in tqdm(cts):
    im0_file = os.path.join(im0_path, f'{ct}.IM0')
    ct_data = None

    # inferior, superior = boundary_dict[ct]

    _, _, max_slice = get_image_resolution(im0_file)

    # start = inferior - 20 if inferior - 20 >= 0 else 0
    # end = superior + 21 if superior + 21 < max_slice else max_slice

    start = 0
    end = max_slice

    for label in labels:
        label_file = os.path.join(image_root_path, label, f'{ct}.BIM')
        label_data = None
        for i in range(start, end):
            slice_file_name = f'{ct}_{i:0>3d}.json'
            label_slice_file_path = os.path.join(label_saved_root_path, label, slice_file_name)

            if not os.path.exists(label_slice_file_path):
                if label_data is None:
                    if not os.path.exists(label_file):
                        os.makedirs(os.path.join(image_root_path, label), exist_ok=True)
                        scp(dst_user='ubuntu', dst_host='10.21.22.70',
                            dst_path=f'/data/dj/data/bca/cavass_data/{label}/{ct}.BIM',
                            local_path=os.path.join(image_root_path, label), dst_port=6202, receive=True)

                    label_data = read_cavass_file(label_file)
                label_slice_data = label_data[..., i]
                data = {'data': label_slice_data, 'subject': ct, 'slice_number': i, 'class': label}
                save_json(label_slice_file_path, data)

    for i in range(start, end):
        slice_file_name = f'{ct}_{i:0>3d}.json'
        ct_slice_path = os.path.join(ct_saved_image_path, slice_file_name)
        if not os.path.exists(ct_slice_path):
            if ct_data is None:
                ct_data = read_cavass_file(im0_file)
            ct_slice_data = {'data': ct_data[..., i],
                             'subject': ct,
                             'slice_number': i}
            save_json(ct_slice_path, ct_slice_data)

import os

from cavass.ops import read_cavass_file
import numpy as np
from jbag.mp import fork
from jbag.io import save_json


def get_slices(image_file, label_file, test_index):
    image = read_cavass_file(image_file)
    label = read_cavass_file(label_file)
    roi_region = np.unique(np.nonzero(label)[2])
    for i in roi_region:
        image_slice = image[..., i]
        label_slice = label[..., i]
        image_slice_file = os.path.join(saved_path, 'images', f'{test_index}_{i:0>3d}.json')
        label_slice_file = os.path.join(saved_path, 'Skn', f'{test_index}_{i:0>3d}.json')
        save_json(image_slice_file, {'data': image_slice})
        save_json(label_slice_file, {'data': label_slice})


if __name__ == '__main__':
    data_path = '/data/dj/data/bca/cavass_data'
    saved_path = '/data/dj/data/bca/json/slices'
    image_path = os.path.join(data_path, 'images')
    label_path = os.path.join(data_path, 'Skn')
    params = []
    for each in os.listdir(label_path):
        if each.endswith('.BIM'):
            test_index = each[:-4]
            params.append((os.path.join(image_path, f'{test_index}.IM0'),
                           os.path.join(label_path, each),
                           test_index)
                          )
    fork(get_slices,8, params)

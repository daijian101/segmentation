import os

from cavass.ops import read_cavass_file
import numpy as np
from jbag.mp import fork
from jbag.io import save_json


def get_slices(image_file, label_file, study):
    image = read_cavass_file(image_file)
    label = read_cavass_file(label_file)
    roi_region = np.unique(np.nonzero(label)[2])
    for i in roi_region:
        image_slice = image[..., i]
        label_slice = label[..., i]
        image_slice_file = os.path.join(saved_path, 'images', f'{study}_{i:0>3d}.json')
        label_slice_file = os.path.join(saved_path, 'Skn', f'{study}_{i:0>3d}.json')
        save_json(image_slice_file, {'data': image_slice})
        save_json(label_slice_file, {'data': label_slice})


if __name__ == '__main__':
    data_path = '/data/dj/NSkn'
    saved_path = '/data/dj/data/bca/json/slices'
    image_path = os.path.join(data_path, 'images')
    label_path = os.path.join(data_path, 'Skn')
    params = []
    for each in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, each)):
            image_file = None
            label_file = None
            for i_each in os.listdir(os.path.join(data_path, each)):
                if i_each.endswith('.IM0'):
                    image_file = os.path.join(data_path, each, i_each)
                if i_each.endswith('.BIM'):
                    label_file = os.path.join(data_path, each, i_each)

            if image_file is not None and label_file is not None:

                params.append((image_file,
                               label_file,
                               each)
                             )
            else:
                print(f'Missing file, image file is {image_file}, label file is {label_file}')
    fork(get_slices,8, params)

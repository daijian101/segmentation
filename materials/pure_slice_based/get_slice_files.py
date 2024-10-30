import os
import sys

import numpy as np
from jbag.mp import fork

sys.path.append('../')

from cavass.ops import read_cavass_file
from jbag.io import save_json


def process_study(study, im0_file, label_files, reference_label):
    ct_data = None

    reference_file = label_files[reference_label]
    reference_data = read_cavass_file(reference_file)
    label_data = {reference_label: reference_data}
    z_slices = np.nonzero(reference_data)[2]
    min_slice, max_slice = np.min(z_slices), np.max(z_slices)
    for z in range(min_slice, max_slice + 1):
        image_slice_file = os.path.join(ct_saved_image_dir, f'{study}_{z:0>3d}.json')
        if not os.path.exists(image_slice_file):
            if ct_data is None:
                ct_data = read_cavass_file(im0_file)
            ct_slice = ct_data[..., z]
            save_json(image_slice_file, {'data': ct_slice, 'study': study, 'slice_index': z})
        for label, label_file in label_files.items():
            label_slice_file = os.path.join(label_saved_dir, label, f'{study}_{z:0>3d}.json')
            if not os.path.exists(label_slice_file):
                if label not in label_data:
                    label_data[label] = read_cavass_file(label_file)
                label_slice = label_data[label][..., z]
                save_json(label_slice_file, {'data': label_slice, 'study': study, 'slice_index': z})


if __name__ == '__main__':
    data_path = '/data1/dj/data/bca/'

    image_dir = os.path.join(data_path, 'cavass_data')

    im0_dir = os.path.join(image_dir, 'images')
    ct_saved_image_dir = os.path.join(data_path, 'json/slices/images')
    label_saved_dir = os.path.join(data_path, 'json/slices')

    labels = ['Sk', 'SMR', 'Skn']

    studies = [each[:-4] for each in os.listdir(os.path.join(image_dir, labels[0])) if each.endswith('BIM')]

    params = []
    for study in studies:
        im0_file = os.path.join(im0_dir, f'{study}.IM0')
        label_files = {}
        for label in labels:
            label_files[label] = os.path.join(image_dir, label, f'{study}.BIM')
        params.append((study, im0_file, label_files, labels[1]))

    fork(process_study, 8, params)

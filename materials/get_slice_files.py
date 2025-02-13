import os
import sys

import numpy as np
from jbag.parallel_processing import execute

sys.path.append('../')

from cavass.ops import read_cavass_file, get_image_resolution
from jbag.io import save_json


def process_study(study, im0_file, label_files):
    ct_data = None
    label_data = {}
    slice_number = get_image_resolution(im0_file)[2]
    for z in range(slice_number):
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

    labels = ['SAT', 'SR']

    studies = [each[:-4] for each in os.listdir(os.path.join(image_dir, labels[0])) if each.endswith('BIM')]

    params = []
    for study in studies:
        im0_file = os.path.join(im0_dir, f'{study}.IM0')
        label_files = {}
        for label in labels:
            label_files[label] = os.path.join(image_dir, label, f'{study}.BIM')
        params.append((study, im0_file, label_files))

    execute(process_study, 64, params)

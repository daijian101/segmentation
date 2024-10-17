import os

import numpy as np
from jbag.io import read_json, save_json, read_txt2list
from jbag.mp import fork


def collect_foreground_intensities(image_file, label_file, n_foreground):
    image_data = read_json(image_file)['data']
    label_data = read_json(label_file)['data']
    label_data = label_data.astype(bool)
    foreground_pixels = image_data[label_data]
    rs = np.random.RandomState(seed=1234)
    chosen_foreground_pixels = rs.choice(foreground_pixels, n_foreground, replace=True) if len(foreground_pixels) > 0 else []
    return chosen_foreground_pixels


if __name__ == '__main__':
    num_foreground_voxels_for_intensity_stats = 10e7
    data_path ='/data/dj/data/bca'
    image_data_path = '/data/dj/data/bca/json/slices'
    label = 'Skn'
    image_path = os.path.join(image_data_path, 'images')
    label_path = os.path.join(image_data_path, label)

    samples = read_txt2list('/data/dj/data/bca/dataset/Skn_training_set.txt')
    num_foreground_voxels_per_image = int(num_foreground_voxels_for_intensity_stats / len(samples))
    print(f'num_foreground_voxels_per_image: {num_foreground_voxels_per_image}')

    params = []
    for sample in samples:
        image_file = os.path.join(image_path, sample)
        label_file = os.path.join(label_path, sample)
        params.append((image_file, label_file, num_foreground_voxels_per_image))

    results = fork(collect_foreground_intensities, 16, params)
    all_chosen_intensities = np.concatenate(results)
    mean = np.mean(all_chosen_intensities)
    std = np.std(all_chosen_intensities)
    percentile_0_5, percentile_99_5 = np.percentile(all_chosen_intensities, [0.5, 99.5])

    dataset_properties = os.path.join(image_data_path, f'dataset/{label}_dataset_properties.json')
    if os.path.exists(dataset_properties):
        properties = read_json(dataset_properties)
    else:
        properties = {}
    properties['intensity_mean'] = mean
    properties['intensity_std'] = std
    properties['intensity_0_5_percentile'] = percentile_0_5
    properties['intensity_99_5_percentile'] = percentile_99_5
    save_json(dataset_properties, properties)

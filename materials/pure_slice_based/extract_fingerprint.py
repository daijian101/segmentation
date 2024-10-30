import os
from random import shuffle

import numpy as np
from jbag.io import save_json, read_json
from jbag.log import logger
from jbag.mp import fork


def split_dataset(all_samples):
    shuffle(all_samples)

    n_training = int(0.8 * len(all_samples))
    n_val = len(all_samples) - n_training

    training_set = all_samples[:n_training]
    val_set = all_samples[n_training:]

    n_test = len(all_samples) - n_training - n_val
    test_set = None

    logger.info(f'# training set is {n_training}')
    logger.info(f'# val set is {n_val}')
    logger.info(f'# test set is {n_test}')

    return training_set, val_set, test_set


def collect_foreground_intensities(image_file, label_file, n_foreground):
    image_data = read_json(image_file)['data']
    label_data = read_json(label_file)['data']
    label_data = label_data.astype(bool)
    foreground_pixels = image_data[label_data]
    rs = np.random.RandomState(seed=1234)
    chosen_foreground_pixels = rs.choice(foreground_pixels, n_foreground, replace=True) if len(
        foreground_pixels) > 0 else []
    return chosen_foreground_pixels


if __name__ == '__main__':
    dataset_dir = '/data1/dj/data/bca/'
    image_dir = os.path.join(dataset_dir, 'json/slices')
    label = 'Sk'
    sample_voxel_region_label = 'Skn'
    # Step 1: split dataset
    all_samples = [each[:-5] for each in os.listdir(os.path.join(image_dir, label)) if each.endswith('json')]
    training_set, val_set, test_set = split_dataset(all_samples)

    # Step 2: extract statistical values

    num_foreground_voxels_for_intensity_stats = 10e7
    n_training_samples = len(training_set)
    num_foreground_voxels_per_image = int(num_foreground_voxels_for_intensity_stats / n_training_samples)
    print(f'Number of foreground pixels for each volume: {num_foreground_voxels_per_image}')

    params = []
    for sample in training_set:
        image_file = os.path.join(image_dir, 'images', f'{sample}.json')
        label_file = os.path.join(image_dir, sample_voxel_region_label, f'{sample}.json')
        params.append((image_file, label_file, num_foreground_voxels_per_image))

    r = fork(collect_foreground_intensities, 8, params)
    all_chosen_intensities = np.concatenate(r)
    mean = np.mean(all_chosen_intensities)
    std = np.std(all_chosen_intensities)
    percentile_0_5, percentile_99_5 = np.percentile(all_chosen_intensities, [0.5, 99.5])

    dataset_properties = {'training_set': training_set, 'val_set': val_set, 'test_set': test_set,
                          'intensity_mean': mean, 'intensity_std': std,
                          'intensity_0_5_percentile': percentile_0_5, 'intensity_99_5_percentile': percentile_99_5}

    dataset_properties_file = os.path.join(dataset_dir, 'dataset', f'{label}_fingerprint.json')
    save_json(dataset_properties_file, dataset_properties)

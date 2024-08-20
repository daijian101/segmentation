import multiprocessing
import os

import numpy as np
from cavass.ops import read_cavass_file
from tqdm import tqdm
from time import sleep
from jbag.io import read_json, save_json


def collect_foreground_intensities(image_file, label_file, n_foreground):
    image_data = read_cavass_file(image_file)
    label_data = read_cavass_file(label_file)
    label_data = label_data.astype(bool)
    foreground_pixels = image_data[label_data]
    rs = np.random.RandomState(seed=1234)
    chosen_foreground_pixels = rs.choice(foreground_pixels, num_foreground_voxels_per_image, replace=True) if len(foreground_pixels) > 0 else []
    return chosen_foreground_pixels


if __name__ == '__main__':
    num_foreground_voxels_for_intensity_stats = 10e7
    labels = ['SAT', 'VAT', 'SMT', 'Sk', 'Dphm']

    for label in labels:
        label_path = f'/data1/dj/data/bca/cavass_data/{label}'
        all_label_files = os.listdir(label_path)
        num_foreground_voxels_per_image = int(num_foreground_voxels_for_intensity_stats / len(all_label_files))
        print(f'num_foreground_voxels_per_image_for_intensity_stats for {label}: {num_foreground_voxels_per_image}')
        r = []
        with multiprocessing.get_context('fork').Pool(16) as p:
            for each in all_label_files:
                ct_name = each[:-4]
                image_file = f'/data1/dj/data/bca/cavass_data/images/{ct_name}.IM0'
                label_files = os.path.join(label_path, each)
                r.append(p.starmap_async(collect_foreground_intensities, ((image_file, label_files, num_foreground_voxels_per_image),)))

            remaining = list(range(len(all_label_files)))
            workers = [j for j in p._pool]
            with tqdm(desc=None, total=len(all_label_files)) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('One of your background processes is missing. In that case reducing the number of workers might help.')
                    done = [i for i in remaining if r[i].ready()]
                    for _ in done:
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)
        results = [i.get()[0] for i in r]
        all_chosen_intensities = np.concatenate(results)
        mean = np.mean(all_chosen_intensities)
        std = np.std(all_chosen_intensities)
        percentile_0_5, percentile_99_5 = np.percentile(all_chosen_intensities, [0.5, 99.5])

        dataset_properties = f'/data1/dj/data/bca/dataset/{label}_dataset_properties.json'
        properties = read_json(dataset_properties)
        properties['intensity_mean'] = mean
        properties['intensity_std'] = std
        properties['intensity_0_5_percentile'] = percentile_0_5
        properties['intensity_99_5_percentile'] = percentile_99_5
        save_json(dataset_properties, properties)

import os.path

from cavass.ops import read_cavass_file

from jbag.io import read_txt_2_list, save_json
import numpy as np
from tqdm import tqdm

labels = ['SMT', 'Sk', 'Dphm']
for label in labels:

    training_images = read_txt_2_list(f'/data1/dj/data/bca/dataset/{label}_training_cts.txt')
    val_images = read_txt_2_list(f'/data1/dj/data/bca/dataset/{label}_val_cts.txt')

    label_file_path = f'/data1/dj/data/bca/cavass_data/{label}'

    dataset_properties = {'training_dataset':training_images, 'val_dataset':val_images, 'image_properties':{}}
    for each in tqdm(training_images):
        label_file = os.path.join(label_file_path, each + '.BIM')
        label_data = read_cavass_file(label_file)
        image_start = 0
        image_end = label_data.shape[2] - 1

        mask_coords = np.where(label_data)
        mask_start = np.min(mask_coords[2])
        mask_end = np.max(mask_coords[2])
        dataset_properties['image_properties'][each] = {'image_start': image_start, 'image_end': image_end,
                                                        'mask_start': mask_start, 'mask_end': mask_end}


    save_json(f'/data1/dj/data/bca/dataset/{label}_dataset_properties.json', dataset_properties)

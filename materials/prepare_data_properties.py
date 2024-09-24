import os.path

from cavass.ops import read_cavass_file

from jbag.io import read_txt_2_list, save_json
import numpy as np
from jbag.mp import fork

data_path = '/data1/dj/data/bca'
def get_image_property(subject_name):
    label_file = os.path.join(label_file_path, subject_name + '.BIM')
    label_data = read_cavass_file(label_file)
    image_start = 0
    image_end = label_data.shape[2] - 1

    mask_coords = np.where(label_data)
    mask_start = np.min(mask_coords[2])
    mask_end = np.max(mask_coords[2])
    return subject_name, image_start, image_end, mask_start, mask_end


labels = ['SMT']
for label in labels:

    training_images = read_txt_2_list(os.path.join(data_path, f'dataset/{label}_training_cts.txt'))
    val_images = read_txt_2_list(os.path.join(data_path, f'dataset/{label}_val_cts.txt'))

    label_file_path = os.path.join(data_path, f'cavass_data/{label}')

    dataset_properties = {'training_dataset':training_images, 'val_dataset':val_images, 'image_properties':{}}
    params = []
    for each in training_images:
        params.append((each,))
    results = fork(get_image_property, 16, params)
    for subject_name, image_start, image_end, mask_start, mask_end in results:
        dataset_properties['image_properties'][subject_name] = {'image_start': image_start, 'image_end': image_end,
                                                        'mask_start': mask_start, 'mask_end': mask_end}
    save_json(os.path.join(data_path, f'dataset/{label}_dataset_properties.json'), dataset_properties)

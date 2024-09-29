import os
import shutil

data_path = '/data/dj/wds11/redownloaded-cases'
new_dataset_path = '/data/dj/wds11/IM0_PET_CT_part6'
os.makedirs(new_dataset_path)

for each in os.listdir(data_path):
    for inner_each in os.listdir(os.path.join(data_path, each)):
        if inner_each.endswith('.IM0'):
            shutil.copy(os.path.join(data_path, each, inner_each), os.path.join(new_dataset_path, inner_each))
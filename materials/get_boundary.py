import os

import numpy as np
import pandas as pd
from cavass.ops import read_cavass_file
from jbag.io import read_txt_2_list
from pandas import read_csv
from tqdm import tqdm

data_path = '/data/dj/data/bca/cavass_data'
samples = read_txt_2_list(f'/data/dj/data/bca/dataset/all_Sk.txt')

label = 'Skn'
missing_label_samples = []

boundary_file = f'/data/dj/data/bca/boundaries/{label}_boundary.csv'
if os.path.exists(boundary_file):
    existing_df = read_csv(boundary_file)
    border_data = np.array(existing_df)
    existing_subjects = border_data[:, 0]
else:
    existing_subjects = []
    existing_df = None

subject_lst, inferior_lst, superior_lst = [], [], []
for ct in tqdm(samples):
    if ct in existing_subjects:
        continue

    label_file = os.path.join(data_path, label, f'{ct}.BIM')
    if not os.path.exists(label_file):
        print(f'label: {label}. label file is missing: {label_file}')
        missing_label_samples.append(ct)
    label_data = read_cavass_file(label_file).astype(bool)
    inferior, superior = None, None
    for i in range(0, label_data.shape[2]):
        if inferior is None and label_data[..., i].sum() > 0:
            inferior = i
            break

    for i in range(label_data.shape[2] - 1, 0, -1):
        if superior is None and label_data[..., i].sum() > 0:
            superior = i
            break
    subject_lst.append(ct)
    inferior_lst.append(inferior)
    superior_lst.append(superior)

data = {"subject": subject_lst, "inferior": inferior_lst, "superior": superior_lst}

df = pd.DataFrame(data)
if existing_df is not None:
    df = pd.concat([existing_df, df])
df.to_csv(boundary_file, index=False)

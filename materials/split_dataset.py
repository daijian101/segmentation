import os
import random

from jbag.io import read_txt_2_list, write_list_2_txt

data_path = '/data1/dj/data/bca'

label = 'Sk'
# all_data = read_txt_2_list(f'/data1/dj/data/bca/dataset/all_{label}.txt')
all_data = [each[:-4] for each in os.listdir(os.path.join(data_path, f'cavass_data/{label}'))]
random.shuffle(all_data)

n_val_samples = 15

val_samples = all_data[:n_val_samples]

training_samples = all_data[n_val_samples:]

write_list_2_txt(os.path.join(data_path, f'dataset/{label}_training_cts.txt'), training_samples)
write_list_2_txt(os.path.join(data_path, f'dataset/{label}_val_cts.txt'), val_samples)



# label = 'SMT'
# all_data = read_txt_2_list(f'/data1/dj/data/bca/dataset/all_{label}.txt')




# val_samples = [each[:-5] for each in os.listdir(f'/data1/dj/data/bca/json/volume/{label}')]
# training_samples = []
#
# for each in os.listdir(f'/data1/dj/data/bca/json/slices/{label}'):
#     ct_name = each[:-9]
#     training_samples.append(ct_name)
# training_samples = set(training_samples)
# training_samples = list(training_samples)
#
#
# write_list_2_txt(f'/data1/dj/data/bca/dataset/{label}_training_cts.txt', training_samples)
# write_list_2_txt(f'/data1/dj/data/bca/dataset/{label}_val_cts.txt', val_samples)
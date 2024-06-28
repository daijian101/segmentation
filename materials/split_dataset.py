import random

from jbag.io import read_txt_2_list, write_list_2_txt

label = 'Sk'
all_data = read_txt_2_list(f'/data1/dj/data/bca/dataset/all_{label}.txt')
random.shuffle(all_data)

n_val_samples = 15

val_samples = all_data[:n_val_samples]

training_samples = all_data[n_val_samples:]

write_list_2_txt(f'/data1/dj/data/bca/dataset/{label}_training_cts.txt', training_samples)
write_list_2_txt(f"/data1/dj/data/bca/dataset/{label}_val_cts.txt", val_samples)

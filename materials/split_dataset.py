import os

from jbag.io import read_txt2list, write_list2txt
from random import shuffle


all_samples = [each for each in os.listdir('/data/dj/data/bca/json/slices/Skn') if each.endswith('.json')]
shuffle(all_samples)

n_training = int(0.8 * len(all_samples))
n_val = len(all_samples) - n_training

training_set = all_samples[:n_training]
val_set = all_samples[n_training:]

write_list2txt('/data/dj/data/bca/dataset/Skn_training_set.txt', training_set)
write_list2txt('/data/dj/data/bca/dataset/Skn_val_set.txt', val_set)

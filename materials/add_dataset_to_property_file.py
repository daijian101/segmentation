from jbag.io import read_json, read_txt2list, save_json

p = read_json('/data/dj/data/bca/dataset/Skn_dataset_properties.json')

training_set  = read_txt2list('/data/dj/data/bca/dataset/Skn_training_set.txt')
val_set = read_txt2list('/data/dj/data/bca/dataset/Skn_val_set.txt')
p['training_dataset'] = training_set
p['val_dataset'] = val_set
save_json('/data/dj/data/bca/dataset/Skn_dataset_properties.json', p)
from jbag.io import read_json, save_json

file = '/data1/dj/data/bca/dataset/Sk_fingerprint.json'
data = read_json(file)

new_p = {}
new_p['intensity_mean'] = data['intensity_mean']
new_p['intensity_std'] = data['intensity_std']
new_p['intensity_0_5_percentile'] = data['intensity_0_5_percentile']
new_p['intensity_99_5_percentile'] = data['intensity_99_5_percentile']
save_json('/data1/dj/tmp/Sk_data_property.json', new_p)
pass

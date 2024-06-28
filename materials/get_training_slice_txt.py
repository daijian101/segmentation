import os

from jbag.io import write_list_2_txt

label = "Dphm"

write_list_2_txt(f"/data/dj/data/bca/dataset/{label}_training_slices.txt",
                 os.listdir(f"/data/dj/data/bca/json/slices/{label}"))

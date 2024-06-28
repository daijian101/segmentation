import numpy as np
from pandas import read_csv


def get_boundary(boundary_file):
    if boundary_file is None:
        return None
    df = read_csv(boundary_file)
    border_data = np.array(df)
    subject_list = border_data[:, 0]
    inferior_list = border_data[:, 1]
    superior_list = border_data[:, 2]
    boundary_dict = dict()
    for subject, inferior, superior in zip(subject_list, inferior_list, superior_list):
        boundary_dict[subject] = (int(inferior), int(superior))
    return boundary_dict

import os
import numpy as np
from jbag.io import read_json
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self,
                 data_indices,
                 raw_data_dir,
                 label_dir_dict: dict = None,
                 add_postfix=False,
                 transforms=None
                 ):
        """
        JSON format image dataset. Recently, I like JSON.

        Args:
            data_indices (sequence):
            raw_data_dir (str or pathlib.Path):
            label_dir_dict (dict or None, optional, default=None):
            add_postfix (bool, optional, default=False): If `True`, append ".json" to the end of file path.
            transforms (torchvision.transforms.Compose or None, optional, default=None):
        """

        self.data_indices = data_indices
        self.raw_data_dir = raw_data_dir
        self.label_dir_dict = label_dir_dict
        self.transforms = transforms
        self.postfix = ".json" if add_postfix else ""

    def __getitem__(self, index):
        json_file = self.data_indices[index]
        image_obj = read_json(os.path.join(self.raw_data_dir, json_file + self.postfix))
        image = image_obj["data"]
        data = {"data": image}
        subject = image_obj["subject"] if "subject" in image_obj else None
        if subject is not None:
            data["subject"] = subject

        if self.label_dir_dict:
            for label, label_dir in self.label_dir_dict.items():
                label_file = os.path.join(label_dir, json_file + self.postfix)
                label_obj = read_json(label_file)
                label_data = label_obj["data"]
                data[label] = label_data

        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.data_indices)


class BalancedForegroundRegionDataset(Dataset):
    def __init__(self,
                 data_indices,
                 image_properties,
                 raw_data_dir,
                 label_dir_dict: dict = None,
                 transforms=None,
                 foreground_sample_probability=0.33
                 ):
        """
        JSON format image dataset. Recently, I like JSON.

        Args:
            data_indices (sequence):
            image_properties (dict):
            raw_data_dir (str or pathlib.Path):
            label_dir_dict (dict or None, optional, default=None):
            transforms (torchvision.transforms.Compose or None, optional, default=None): image transform chain.
            foreground_sample_probability (float, optional, default=0.33): Probability of forced sampling foreground slice.
        """

        self.data_indices = data_indices
        self.raw_data_dir = raw_data_dir
        self.label_dict = label_dir_dict
        self.image_properties = image_properties
        self.transforms = transforms
        self.foreground_sample_probability = foreground_sample_probability

    def __getitem__(self, index):
        subject = self.data_indices[index]

        sample_fg = np.random.random_sample() <= self.foreground_sample_probability

        if sample_fg:
            slice_idx = np.random.randint(low=self.image_properties[subject]['mask_start'],
                                          high=self.image_properties[subject]['mask_end'] + 1)
        else:
            slice_idx = np.random.randint(low=self.image_properties[subject]['image_start'],
                                          high=self.image_properties[subject]['image_end'] + 1)

        slice_image_file = os.path.join(self.raw_data_dir, f'{subject}_{slice_idx:0>3d}.json')
        image_obj = read_json(slice_image_file)
        image = image_obj["data"]
        data = {"data": image}
        subject = image_obj["subject"] if "subject" in image_obj else None
        if subject is not None:
            data["subject"] = subject

        if self.label_dict:
            for label, label_dir in self.label_dict.items():

                label_file = os.path.join(label_dir, f'{subject}_{slice_idx:0>3d}.json')
                label_obj = read_json(label_file)
                label_data = label_obj["data"]
                data[label] = label_data

        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.data_indices)

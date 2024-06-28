import os

from jbag.io import read_json
from torch.utils.data import Dataset


class JSONImageDataset(Dataset):
    def __init__(self,
                 sample_list,
                 sample_dir,
                 label_dict: dict = None,
                 label_key='label',
                 add_postfix=False,
                 transforms=None
                 ):
        """
        JSON format image dataset. Recently, I like JSON.

        Args:
            sample_list (sequence):
            sample_dir (str or pathlib.Path):
            label_dict (dict or None, optional, default=None):
            label_key (str, optional, default='label'): key in label obj that indicates gt segmentation
            add_postfix (bool, optional, default=False): If `True`, append ".json" to the end of file path.
            transforms (torchvision.transforms.Compose or None, optional, default=None):
        """

        self.sample_list = sample_list
        self.json_dir = sample_dir
        self.label_dict = label_dict
        self.transforms = transforms
        self.key = label_key
        self.postfix = ".json" if add_postfix else ""

    def __getitem__(self, index):
        json_file = self.sample_list[index]
        image_obj = read_json(os.path.join(self.json_dir, json_file + self.postfix))
        image = image_obj["data"]
        data = {"data": image}
        subject = image_obj["subject"] if "subject" in image_obj else None
        if subject is not None:
            data["subject"] = subject

        if self.label_dict:
            for label, label_dir in self.label_dict.items():
                label_file = os.path.join(label_dir, json_file + self.postfix)
                label_obj = read_json(label_file)
                label_data = label_obj["data"]
                data[label] = label_data

        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.sample_list)

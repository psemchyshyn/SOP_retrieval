import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import os
import random

class EbayDataset(Dataset):
    def __init__(self, config_data, transforms=None):
        super(EbayDataset, self).__init__()

        self.transforms = transforms
        self.path = config_data["path"]
        self.image_size = config_data["image_size"]
        self.data_folder = os.path.join(*self.path.split(os.sep)[:-1])

        self.info_pd = pd.read_csv(self.path, sep=" ")
        self.info_pd["super_class_id"] = self.info_pd["super_class_id"] - 1
        self.info_pd = self.info_pd.groupby("super_class_id").head(config_data["samples_per_class"]).reset_index()
        self.classes = self.info_pd["super_class_id"].unique().tolist()


    def __getitem__(self, idx):
        item = self.info_pd.loc[idx, ["class_id", "super_class_id", "path"]]
        class_id, super_class_id, image_path = item.to_list()
        return self.get_image(image_path), class_id, super_class_id

    def get_random_image_from_class(self, super_class_id):
        df = self.info_pd.loc[self.info_pd["super_class_id"] == super_class_id, ["class_id", "super_class_id", "path"]]
        *_, image_path = df.iloc[random.randint(0, len(df) - 1)].to_list()
        return self.get_image(image_path)

    def get_image(self, image_path):
        image = Image.open(os.path.join(self.data_folder, image_path)).convert("RGB")
        image = np.array(image.resize((self.image_size, self.image_size)))
        if self.transforms:
            image = self.transforms(image=image)["image"]
        return torch.from_numpy(image).permute(2, 0, 1) / 255

    def __len__(self):
        return len(self.info_pd)


class EbayDatasetSiamse(EbayDataset):
    def __init__(self, config_data, transforms=None):
        super(EbayDatasetSiamse, self).__init__(config_data, transforms)


    def __getitem__(self, idx):
        item = self.info_pd.loc[idx, ["class_id", "super_class_id", "path"]]
        _, super_class_id, image_path = item.to_list()
        image1 = self.get_image(image_path)

        should_get_same_class = random.randint(0,1) 

        if should_get_same_class:
            clas = super_class_id
        else:
            clas = random.choice([i for i in self.classes if i != super_class_id])

        image2 = self.get_random_image_from_class(clas)
        return image1, image2, should_get_same_class

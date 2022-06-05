import yaml
import albumentations as A
from torch.utils.data import random_split


def train_valid_split(ds, train_percentage=0.8):
    train_size = int(len(ds)*train_percentage)
    return random_split(ds, [train_size, len(ds) - train_size])

def parse_config(config_path):
    with open(config_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def get_basic_augmentations():
    return A.Compose([
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
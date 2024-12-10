import numpy as np
import pandas as pd
import torch
import torch.utils.data as data  # Data handling utilities
import torchvision.transforms as tt  # Image transformations
import cv2  # Computer vision
import albumentations as A  # Image augmentations
from torch.utils.data import DataLoader  # Data loading
from torchvision.utils import make_grid  # Create image grids
from sklearn.model_selection import train_test_split
from skimage.morphology import binary_dilation
from data_processing.data_reader import data_reader
from utils import dataset_info


# Custom PyTorch Dataset class for loading images and masks from a DataFrame.
class BrainDataset(data.Dataset):
    def __init__(self, df, transform=None, explanation=False):
        super(BrainDataset, self).__init__()
        self.df = df
        self.transform = transform
        self.means = np.array((0.485, 0.456, 0.406))
        self.stds = np.array((0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 0])
        image = np.array(image)/255.
        mask = cv2.imread(self.df.iloc[idx, 1], 0)
        mask = np.array(mask)/255.
        label = np.array(self.df.iloc[idx, 2]).astype(int)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        image = image.transpose((2,0,1))
        image = torch.from_numpy(image).type(torch.float32)
        image = tt.Normalize(self.means, self.stds)(image)
        mask = np.expand_dims(mask, axis=-1).transpose((2,0,1))
        mask = torch.from_numpy(mask).type(torch.float32)

        return image, mask, label

def create_dataloader(ROOT_PATH, batch_size=64):
    files_df = data_reader(ROOT_PATH)
    # Splitting the dataset into training data (train_df), validation data (val_df),
    # and test data (test_df) with specified proportions.
    train_df, val_df = train_test_split(files_df, stratify=files_df['diagnosis'], test_size=0.1, random_state=0)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df, test_df = train_test_split(train_df, stratify=train_df['diagnosis'], test_size=0.15, random_state=0)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("Train: {}\nVal: {}\nTest: {}".format(train_df.shape, val_df.shape, test_df.shape))

    # Define transformations for training, validation, and testing datasets using Albumentations library.
    train_transform = A.Compose([
        A.Resize(width=128, height=128, p=1.0),  # Resize images to 128x128 pixels
        A.HorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
        A.VerticalFlip(p=0.5),  # Apply vertical flip with 50% probability
        A.RandomRotate90(p=0.5),  # Rotate randomly by 90 degrees with 50% probability
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        # Randomly shift, scale, and rotate
    ])

    val_transform = A.Compose([
        A.Resize(width=128, height=128, p=1.0),  # Resize images to 128x128 pixels
        A.HorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability (for data augmentation)
    ])

    test_transform = A.Compose([
        A.Resize(width=128, height=128, p=1.0),  # Resize images to 128x128 pixels
    ])

    train_ds = BrainDataset(train_df, train_transform)
    val_ds = BrainDataset(val_df, val_transform)
    test_ds = BrainDataset(test_df, test_transform)

    print('Train dataset:')
    dataset_info(train_ds)
    print('Validation dataset:')
    dataset_info(val_ds)
    print('Test dataset:')
    dataset_info(test_ds)

    train_dl = DataLoader(train_ds, batch_size)#, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size)
    test_dl = DataLoader(test_ds, batch_size)

    return train_dl, val_dl, test_dl, train_ds.means, train_ds.stds
import os
import numpy as np
import pandas as pd

import glob  # Filename pattern matching
import random
import cv2  # Computer vision

from mpl_toolkits.axes_grid1 import ImageGrid  # Image grid layout
from utils import set_seed
import matplotlib.pyplot as plt


def diagnosis(mask_path):
    return 1 if np.max(cv2.imread(mask_path)) > 0 else 0


def data_reader(ROOT_PATH):
    # Using glob.glob to collect paths of all mask files in subdirectories
    mask_files = glob.glob(ROOT_PATH + 'kaggle_3m/*/*_mask*')
    image_files = [file.replace('_mask', '') for file in mask_files]

    # Defining a function diagnosis(mask_path) that returns 1
    #if the maximum pixel value in the mask image (read using cv2) is greater than 0


    if not os.path.exists(os.path.join(ROOT_PATH, 'labels.csv')):
      files_df = pd.DataFrame({"image_path": image_files,
                        "mask_path": mask_files,
                        "diagnosis": [diagnosis(x) for x in mask_files]})
      files_df.to_csv(os.path.join(ROOT_PATH, 'labels.csv'))
    else:
      files_df = pd.read_csv(os.path.join(ROOT_PATH, 'labels.csv'), index_col=0, header=0)

    print("Total of No Tumor:", files_df['diagnosis'].value_counts()[0])
    print("Total of Tumor:", files_df['diagnosis'].value_counts()[1])

    return files_df


def view_data(files_df):
    set_seed()

    # Prepare the images and masks
    images, masks = [], []
    df_positive = files_df[files_df['diagnosis'] == 1].sample(5).values

    for sample in df_positive:
        img = cv2.imread(sample[0])
        mask = cv2.imread(sample[1])
        images.append(img)
        masks.append(mask)

    # Reverse the order of images and masks
    images = np.array(images[4::-1])
    masks = np.array(masks[4::-1])

    # Concatenate the images and masks horizontally
    images_concat = np.hstack(images)
    masks_concat = np.hstack(masks)

    # Plot the images, masks, and overlays
    fig = plt.figure(figsize=(15, 10))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 1), axes_pad=0.6)

    grid[0].imshow(images_concat)
    grid[0].set_title('Images', fontsize=15)
    grid[0].axis('off')

    grid[1].imshow(masks_concat)
    grid[1].set_title('Masks', fontsize=15)
    grid[1].axis('off')

    grid[2].imshow(images_concat)
    grid[2].imshow(masks_concat, alpha=0.6)
    grid[2].set_title('Brain MRI with mask', fontsize=15)
    grid[2].axis('off')

    plt.show()


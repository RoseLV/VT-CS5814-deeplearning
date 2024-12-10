import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from pytorch_grad_cam.utils.image import show_cam_on_image

def set_seed(seed=0):  # Function to set random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)  #
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def dataset_info(dataset):
    print(f'Size of dataset: {len(dataset)}')
    index = random.randint(1, 40)
    img, mask, label = dataset[index]
    print(f'Sample-{index} Image size: {img.shape}, Mask: {mask.shape}, Label: {label}\n')

# Function to plot Dice coefficient history across epochs.
def plot_dice_history(model_name, train_dice_history, val_dice_history, num_epochs):

    x = np.arange(num_epochs)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, train_dice_history, label='Train DICE Score', lw=3, c="r")
    plt.plot(x, val_dice_history, label='Validation DICE Score', lw=3, c="c")

    plt.title(f"{model_name}", fontsize=20)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("DICE", fontsize=15)

    plt.show()

# Function to plot loss history across epochs.
def plot_loss_history(model_name, train_loss_history, val_loss_history, num_epochs):

    x = np.arange(num_epochs)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, train_loss_history, label='Train Loss', lw=3, c="r")
    plt.plot(x, val_loss_history, label='Validation Loss', lw=3, c="c")

    plt.title(f"{model_name}", fontsize=20)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)

    plt.show()

# Function to calculate the Dice coefficient metric between prediction and ground truth.
def dice_coef_metric(pred, label):
    intersection = 2.0 * (pred * label).sum()
    union = pred.sum() + label.sum()
    if pred.sum() == 0 and label.sum() == 0:
        return 1.
    return intersection / union

# Function to calculate the Dice coefficient loss between prediction and ground truth.
def dice_coef_loss(pred, label):
    smooth = 1.0
    intersection = 2.0 * (pred * label).sum() + smooth
    union = pred.sum() + label.sum() + smooth
    return 1 - (intersection / union)

# Function to calculate the combined BCE (Binary Cross Entropy) and Dice loss.
def bce_dice_loss(pred, label):
    dice_loss = dice_coef_loss(pred, label)
    bce_loss = nn.BCELoss()(pred, label)
    return dice_loss + bce_loss

def inverse_transform(data, means, stds):
    if len(data.shape) == 4:
        return data * stds[None, :, None, None] + means[None, :, None, None]
    return data * stds + means

def plot_lime_results(explanation, means, stds):
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[1],
                                                positive_only=False,
                                                num_features=20,
                                                hide_rest=False,
                                                min_weight=-1)
    img_boundry2 = mark_boundaries(temp, mask)
    plt.imshow(inverse_transform(img_boundry2, means, stds))
    plt.show()

def plot_integrated_gradients_results(image, integrated_gradients, means, stds):
    temp = inverse_transform(image, means, stds)
    ig_mean = integrated_gradients.max(axis=-1)
    pos, neg = np.percentile(ig_mean, 90), np.percentile(ig_mean, 10)
    temp[ig_mean < neg, 0] = np.max(temp)
    temp[ig_mean > pos, 1] = np.max(temp)
    plt.imshow(temp)
    plt.show()

def plot_heatmap(rgb_img, mask_label, masks):
    visualizations = []
    for _mask in masks:
        visualization = show_cam_on_image(rgb_img, _mask, use_rgb=True)
        visualizations.append(visualization)
        # plt.imshow(visualization)
        # plt.show()

    visualization = show_cam_on_image(rgb_img, mask_label, use_rgb=True)
    visualizations.append(visualization)
    # plt.imshow(visualization)
    # plt.show()
    return visualizations

def scale_to_01(data):
    return (data - data.min()) / (data.max() - data.min() + 1e-6)

def compute_mask_score(mask, mask_label):
    k = np.sum(mask_label).astype(int)
    kth_largest = np.partition(mask.flatten(), -k)[-k]
    mask_top = (mask > kth_largest).astype(int)
    return (mask_top * mask_label).sum()

def plot_layout(arrs_1, arrs_2, images_1, images_2, ts, save_path):
    def plot_image(row_id, title, image, ax):
        ax.imshow(image)
        if row_id == 0:
            ax.set_title(title, wrap=True)

    matplotlib.rcParams.update({'font.size': 22})
    print(f"{arrs_1.shape[1]} samples of {arrs_1.shape[0]} timestamps")
    for k in range(arrs_1.shape[1]): # iterate by images
        # fig = plt.figure(figsize=(8*3, 5*3))
        fig = plt.figure(figsize=(4 * 5, 3 * 5))

        ax = []
        # for i in range(len(images_1)):
        #     for j, title in enumerate(["lime", "integrated_gradients", "grad_cam", "ground_truth"]):
        #         ax.append(plt.subplot2grid((5, 8), (i, j)))
        #         plot_image(i, title, images_1[i][title], ax[-1])
        # for i in range(len(images_2)):
        #     for j, title in enumerate(["lime", "integrated_gradients", "grad_cam", "ground_truth"]):
        #         ax.append(plt.subplot2grid((5, 8), (i, j+4)))
        #         plot_image(i, title, images_2[i][title], ax[-1])

        # ax.append(plt.subplot2grid((2, 3), (0, 0)))
        # plot_image(0, "Classification\nPretrained", images_1[k]["grad_cam"], ax[-1])
        ax.append(plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2))
        plot_image(0, "Grad-Cam", images_2[k]["grad_cam"], ax[-1])
        ax.append(plt.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=2))
        plot_image(0, "Ground Truth", images_2[k]["ground_truth"], ax[-1])

        ax_loss_1 = plt.subplot2grid((3, 4), (2, 0), colspan=4)
        # ax_loss_1.plot(arrs_1.mean(axis=-1), label="Classification Pretrained", linewidth=5)
        ax_loss_1.plot(arrs_2.mean(axis=-1), label="Top-K Intersection over Union (Top-K IoU)", linewidth=5)
        ax_loss_1.set_xlim([0, 40])
        ax_loss_1.set_ylabel("Accuracy")
        ax_loss_1.set_xlabel("Epoch")
        ax_loss_1.legend()

        # ax_loss_1 = plt.subplot2grid((5, 8), (3, 0), colspan=4)
        # ax_loss_1.plot(arrs_1[0], label="train")
        # ax_loss_1.plot(arrs_1[1], label="validation")
        # ax_loss_1.set_xlim([0, 120])
        # ax_loss_1.set_ylabel("Loss")
        # ax_loss_1.legend()
        #
        # ax_acc_1 = plt.subplot2grid((5, 8), (4, 0), colspan=4)
        # ax_acc_1.plot(arrs_1[2], label="train")
        # ax_acc_1.plot(arrs_1[3], label="validation")
        # ax_acc_1.set_xlim([0, 120])
        # ax_acc_1.set_ylabel("Accuracy")
        # ax_acc_1.set_xlabel("Epoch")
        # ax_acc_1.legend()
        #
        # ax_loss_2 = plt.subplot2grid((5, 8), (3, 4), colspan=4)
        # ax_loss_2.plot(arrs_2[0], label="train")
        # ax_loss_2.plot(arrs_2[1], label="validation")
        # ax_loss_2.set_xlim([0, 120])
        # ax_loss_2.set_ylabel("Loss")
        # ax_loss_2.legend()
        #
        # ax_acc_2 = plt.subplot2grid((5, 8), (4, 4), colspan=4)
        # ax_acc_2.plot(arrs_2[2], label="train")
        # ax_acc_2.plot(arrs_2[3], label="validation")
        # ax_acc_2.set_xlim([0, 120])
        # ax_acc_2.set_ylabel("Accuracy")
        # ax_acc_2.set_xlabel("Epoch")
        # ax_acc_2.legend()

        plt.tight_layout()

        print("Saving image....", k)
        plt.savefig(save_path + f"/figures/{k}_{ts}.png")
        plt.show()
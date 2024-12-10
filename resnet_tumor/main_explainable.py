import torch
import torch.nn as nn
import numpy as np
from models.ResNet import ResNet50, ResNet50_seg_t2
from utils import bce_dice_loss, set_seed, plot_layout
from data_processing.data_loader import create_dataloader
from training.training import eval_loop_classification_explainable
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"  # Select GPU if available, otherwise CPU
print("Using {} device".format(device))  # Print the selected device

set_seed()
ROOT_PATH = "../../data/tumor/brain_mri_seg/lgg-mri-segmentation/"
MODEL_NAME_1 = "ResNet"
MODEL_NAME_2 = "ResNet_seg_t2"
MODEL_PATH_1 = "../../data/tumor/brain_mri_seg/models/" + MODEL_NAME_1
MODEL_PATH_2 = "../../data/tumor/brain_mri_seg/models/" + MODEL_NAME_2
training = False
# Hyperparams for training
batch_size = 1

train_dl, val_dl, test_dl, means, stds = create_dataloader(ROOT_PATH, batch_size=batch_size)

train_loss_1 = np.load(MODEL_PATH_1 + '/train_loss_classification.npy')
val_loss_1 = np.load(MODEL_PATH_1 + '/val_loss_classification.npy')
train_dice_1 = np.load(MODEL_PATH_1 + '/train_dice_classification.npy')
val_dice_1 = np.load(MODEL_PATH_1 + '/val_dice_classification.npy')
train_loss_2 = np.load(MODEL_PATH_2 + '/train_loss_classification.npy')
val_loss_2 = np.load(MODEL_PATH_2 + '/val_loss_classification.npy')
train_dice_2 = np.load(MODEL_PATH_2 + '/train_dice_classification.npy')
val_dice_2 = np.load(MODEL_PATH_2 + '/val_dice_classification.npy')

# instance plot
max_1, max_2 = 71, 63
scores_1, scores_2 = [], []
for i in range(35):
    CHECKPOINT_1 = f"brain-mri-classification_{min(i, max_1)}"
    CHECKPOINT_2 = f"brain-mri-classification_{min(i, max_2)}"
    model_1 = ResNet50(2, True).to(device)
    model_2 = ResNet50_seg_t2(2, True).to(device)
    model_1.load_state_dict(torch.load(MODEL_PATH_1 + f'/{CHECKPOINT_1}.pth', weights_only=True))
    model_2.load_state_dict(torch.load(MODEL_PATH_2 + f'/{CHECKPOINT_2}.pth', weights_only=True))

    # print('Net\'s state_dict:')
    # total_param = 0
    # for param_tensor in model.state_dict():
    #     print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    #     total_param += np.prod(model.state_dict()[param_tensor].size())
    # print('Net\'s total params:', total_param)


    # label 0 -- health; 1 -- tumor
    dataset_name = 'test'
    print(f"Running {MODEL_NAME_1} {CHECKPOINT_1}")
    visualizations_1, explanations_1 = eval_loop_classification_explainable(model_1, test_dl, means, stds, dataset_name, 5, device)
    explanations_1 = pd.DataFrame(explanations_1)
    print(f"Running {MODEL_NAME_2} {CHECKPOINT_2}")
    visualizations_2, explanations_2 = eval_loop_classification_explainable(model_2, test_dl, means, stds, dataset_name, 5, device)
    explanations_2 = pd.DataFrame(explanations_2)

    # arrs_1 = [train_loss_1[:i + 1], val_loss_1[:i + 1], train_dice_1[:i + 1], val_dice_1[:i + 1]]
    # arrs_2 = [train_loss_2[:i + 1], val_loss_2[:i + 1], train_dice_2[:i + 1], val_dice_2[:i + 1]]
    #
    # plot_layout(arrs_1, arrs_2, visualizations_1, visualizations_2, i, "../../data/tumor/brain_mri_seg/models/")
    if isinstance(scores_1, list):
        scores_1 = np.expand_dims((explanations_1["grad_cam"] / explanations_1["total"]).values, axis=0)
        scores_2 = np.expand_dims((explanations_2["grad_cam"] / explanations_2["total"]).values, axis=0)
    else:
        scores_1 = np.concatenate((scores_1, (explanations_1["grad_cam"]/explanations_1["total"]).values[None, :]), axis=0)
        scores_2 = np.concatenate((scores_2, (explanations_2["grad_cam"]/explanations_2["total"]).values[None, :]), axis=0)
    plot_layout(scores_1, scores_2, visualizations_1, visualizations_2, i, "../../data/tumor/brain_mri_seg/models/")
    # explanations_1.to_csv(MODEL_PATH_1 + f'/{CHECKPOINT_1}_{dataset_name}_score.csv')
    # explanations_2.to_csv(MODEL_PATH_2 + f'/{CHECKPOINT_2}_{dataset_name}_score.csv')

# np.save(f"../../data/tumor/brain_mri_seg/models/{MODEL_NAME_1}/grad_cam_score.npy", scores_1, allow_pickle=True)
# np.save(f"../../data/tumor/brain_mri_seg/models/{MODEL_NAME_2}/grad_cam_score.npy", scores_2, allow_pickle=True)
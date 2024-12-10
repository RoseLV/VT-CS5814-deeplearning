import torch
import torch.nn as nn
from models.ResNet import ResNet50, ResNet50_seg, ResNet50_seg_t2
from utils import bce_dice_loss, set_seed
from data_processing.data_loader import create_dataloader
from training.training import train_model, eval_loop_classification
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"  # Select GPU if available, otherwise CPU
print("Using {} device".format(device))  # Print the selected device

set_seed()
MODEL_NAME = "ResNet"
ROOT_PATH = "../../data/tumor/brain_mri_seg/lgg-mri-segmentation/"
MODEL_PATH = "../../data/tumor/brain_mri_seg/models/" + MODEL_NAME
training = True
# Hyperparams for training
num_epochs = 1000
batch_size = 64

# Instantiate training
model = ResNet50(2, fine_tuning=True).to(device)
# model = ResNet50_seg(2, True).to(device)
# model = ResNet50_seg_t2(2, True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
loss = nn.CrossEntropyLoss() # bce_dice_loss

print('Net\'s state_dict:')
total_param = 0
for param_tensor in model.state_dict():
    print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    total_param += np.prod(model.state_dict()[param_tensor].size())
print('Net\'s total params:', total_param)

train_dl, val_dl, test_dl, _, _ = create_dataloader(ROOT_PATH, batch_size=batch_size)

if training:
    train_loss_history, train_dice_history, val_loss_history, val_dice_history = train_model(model, train_dl, val_dl, loss, optimizer, scheduler, num_epochs, device, MODEL_PATH)
    np.save(MODEL_PATH + '/train_loss_classification.npy', train_loss_history, allow_pickle=True)
    np.save(MODEL_PATH + '/train_dice_classification.npy', train_dice_history, allow_pickle=True)
    np.save(MODEL_PATH + '/val_loss_classification.npy', val_loss_history, allow_pickle=True)
    np.save(MODEL_PATH + '/val_dice_classification.npy', val_dice_history, allow_pickle=True)
else:
    model.load_state_dict(torch.load(MODEL_PATH + '/brain-mri-classification_72.pth'))
test_dice, test_loss = eval_loop_classification(model, test_dl, loss, device, scheduler, training=False)
print("Mean DICE: {:.3f}%, Loss: {:.3f}".format((100*test_dice), test_loss))
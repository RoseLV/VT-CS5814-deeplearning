import torch
import torch.nn.functional as F
import numpy as np
from utils import dice_coef_loss, bce_dice_loss, dice_coef_metric, plot_lime_results, plot_integrated_gradients_results, plot_heatmap, scale_to_01, compute_mask_score
from get_explainable import get_lime, get_integrated_gradients, get_shap, get_grad_cam
from lime import lime_image
import shap
from utils import inverse_transform

# Function to perform the training loop for the model.
def train_loop_classification(model, loader, loss_func, device, optimizer):
    model.train()
    train_losses = []
    train_dices = []

    for i, (image, mask, label) in enumerate(loader):
        image = image.to(device)
        label = label.long().to(device)
        outputs = model(image)

        loss = loss_func(outputs, label)
        res = outputs.argmax(dim=1)
        dice = (res == label).float().mean().item()
        train_losses.append(loss.item())
        train_dices.append(dice)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return train_dices, train_losses


# Function to perform evaluation loop for the model.
def eval_loop_classification(model, loader, loss_func, device, scheduler, training=True):
    model.eval()
    val_loss = 0
    val_dice = 0
    with torch.no_grad():
        for step, (image, mask, label) in enumerate(loader):
            image = image.to(device)
            label = label.long().to(device)
            outputs = model(image)

            loss = loss_func(outputs, label)
            res = outputs.argmax(dim=1)
            dice = (res == label).float().mean().item()
            val_loss += loss.item()
            val_dice += dice

        val_mean_dice = val_dice / (step + 1)
        val_mean_loss = val_loss / (step + 1)

        if training:
            scheduler.step(val_mean_dice)

    return val_mean_dice, val_mean_loss

def eval_loop_classification_explainable(model, loader, means, stds, dataset_name, stop_step, device, ids = None):
    # ids = [207, 93, 108, 453, 331, 106, 317, 333] #[453, 331, 317] # 71 vs 120
    # ids = [80, 425, 212, 102, 216, 446, 108, 168, 262, 525, 251, 227, 340, 20, 148]
    ids = [227]#[70, 57, 116, 186, 155, 151, 227]
    # ids = [409, 217, 381, 70, 57, 116, 186, 293, 492, 155, 178, 151, 227, 321, 268, 501]
    model.eval()
    explanations, visualizations = [], []
    valid_step = 0
    for step, (image, mask, label) in enumerate(loader):
        if label[0] < 1e-6 or step not in ids:
            continue
        print(f"step: {step}, picked_step: {valid_step} ------------------------------")
        batch_size = image.shape[0]
        assert batch_size == 1, "Only batch size 1 is supported."

        # lime_mask = get_lime(model, image, device) # H*W
        # lime_mask = np.expand_dims(scale_to_01(lime_mask), axis=-1)
        # ig_mask = get_integrated_gradients(model, image, device) # H*W*C
        # ig_mask = scale_to_01(ig_mask).mean(axis=-1, keepdims=True)
        # # shap = get_shap(model, image, means, stds, device) # TODO: finish the implementation
        gc_mask = get_grad_cam(model, image, device) # C*H*W
        gc_mask = gc_mask.transpose(1, 2, 0)
        # In this example grayscale_cam has only one image in the batch
        mask_label = mask[0].permute(1, 2, 0).numpy()
        rgb_img = inverse_transform(image[0].permute(1, 2, 0), means, stds).numpy()
        visualization = plot_heatmap(rgb_img, mask_label, [gc_mask]) # [lime_mask, ig_mask, gc_mask]
        # visualizations.append({"dataset": dataset_name, "id": step, "lime": visualization[0], "integrated_gradients": visualization[1],
        #                      "grad_cam": visualization[2], "ground_truth": visualization[3]})
        visualizations.append({"dataset": dataset_name, "id": step, "lime": None, "integrated_gradients": None,
                             "grad_cam": visualization[0], "ground_truth": visualization[1]})
        # explanations.append({"image": rgb_img, "mask": mask[0].permute(1, 2, 0), "label": label,
        #                      "lime": lime_mask, "integrated_gradients": ig_mask,
        #                      "grad_cam": gc_mask})
        # lime_score = compute_mask_score(lime_mask, mask_label)
        # ig_score = compute_mask_score(ig_mask, mask_label)
        gc_score = compute_mask_score(gc_mask, mask_label)
        # explanations.append({"dataset": dataset_name, "id": step, "lime": lime_score, "integrated_gradients": ig_score,
        #                      "grad_cam": gc_score, "total": mask_label.sum()})
        explanations.append({"dataset": dataset_name, "id": step, "lime": -1, "integrated_gradients": -1,
                             "grad_cam": gc_score, "total": mask_label.sum()})

        valid_step += 1
        # if valid_step == stop_step:
        #     break
    return visualizations, explanations

def train_loop_segmentation(model, loader, loss_func, device, optimizer):
    model.train()
    train_losses = []
    train_dices = []

    for i, (image, mask, label) in enumerate(loader):
        image = image.to(device)
        mask = mask.to(device)
        label = label.long().to(device)
        outputs = model(image)

        # Convert outputs to numpy array for post-processing
        out_cut = np.copy(outputs.data.cpu().numpy())
        out_cut[np.nonzero(out_cut < 0.5)] = 0.0
        out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

        dice = dice_coef_metric(out_cut, mask.data.cpu().numpy())
        loss = loss_func(outputs, mask)
        train_losses.append(loss.item())
        train_dices.append(dice)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return train_dices, train_losses


# Function to perform evaluation loop for the model.
def eval_loop_segmentation(model, loader, loss_func, device, scheduler, training=True):
    model.eval()
    val_loss = 0
    val_dice = 0
    with torch.no_grad():
        for step, (image, mask, label) in enumerate(loader):
            image = image.to(device)
            mask = mask.to(device)
            label = label.long().to(device)
            outputs = model(image)
            loss = loss_func(outputs, mask)

    # Convert outputs to numpy array for post-processing
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0
            dice = dice_coef_metric(out_cut, mask.data.cpu().numpy())

            val_loss += loss
            val_dice += dice

        val_mean_dice = val_dice / step
        val_mean_loss = val_loss / step

        if training:
            scheduler.step(val_mean_dice)

    return val_mean_dice, val_mean_loss

# Function to train the model and evaluate on validation data across epochs.
def train_model(model, train_loader, val_loader, loss_func, optimizer, scheduler, num_epochs, device, MODEL_PATH):
    train_loss_history = []
    train_dice_history = []
    val_loss_history = []
    val_dice_history = []
    best_epoch, best_val_loss = -1, np.inf

    for epoch in range(num_epochs):
        train_dices, train_losses = train_loop_classification(model, train_loader, loss_func, device, optimizer)
        train_mean_dice = np.array(train_dices).mean()
        train_mean_loss = np.array(train_losses).mean()
        val_mean_dice, val_mean_loss = eval_loop_classification(model, val_loader, loss_func, device, scheduler)

        torch.save(model.state_dict(), MODEL_PATH + f'/brain-mri-classification_{epoch}.pth')

        train_loss_history.append(np.array(train_losses).mean())
        train_dice_history.append(np.array(train_dices).mean())
        val_loss_history.append(val_mean_loss)
        val_dice_history.append(val_mean_dice)

        print('Epoch: {}/{} |  Train Loss: {:.3f}, Val Loss: {:.3f}, Train DICE: {:.3f}, Val DICE: {:.3f}'
              .format(epoch+1, num_epochs, train_mean_loss, val_mean_loss, train_mean_dice,val_mean_dice))

        if best_val_loss > val_mean_loss:
            best_epoch, best_val_loss = epoch, val_mean_loss
        if epoch - best_epoch > 50:
            break

    return train_loss_history, train_dice_history, val_loss_history, val_dice_history


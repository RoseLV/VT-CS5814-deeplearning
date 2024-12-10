import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from utils import plot_lime_results, plot_integrated_gradients_results, inverse_transform
from integrated_gradients.resnet_gradient import compute_integrated_gradient
from lime import lime_image
import shap

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import os
os.environ["OMP_NUM_THREADS"] = "1"

def get_lime(model, image, device):
    explainer = lime_image.LimeImageExplainer()
    func = lambda x: [F.softmax(model(torch.Tensor(x).permute(0, 3, 1, 2).to(device)), dim=1).detach().cpu().numpy()[0]]
    model.eval()
    with torch.no_grad():
        batch_size = image.shape[0]
        assert batch_size == 1, "Only batch size 1 is supported."
        image = image.squeeze()
        explanation = explainer.explain_instance(image.permute(1, 2, 0).numpy(),
                                                 func,  # classification function
                                                 top_labels=2,
                                                 hide_color=0,
                                                 num_samples=1000,
                                                 batch_size=batch_size,
                                                 random_seed=0)  # number of images that will be sent to classification function
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[1],
                                                    positive_only=False,
                                                    # negative_only=True,
                                                    num_features=20,
                                                    hide_rest=False,
                                                    min_weight=-5)
        # plot_lime_results(explanation, means, stds)
    return mask

def get_integrated_gradients(model, image, device):
    model.eval()
    idx = 1  # tumor class

    # Read images
    batch_x = image.to(device)
    batch_blank = torch.zeros_like(image).to(device)

    # Integrated gradient computation
    integrated_gradients = compute_integrated_gradient(batch_x, batch_blank, model, idx)

    # Change to channel last
    integrated_gradients = integrated_gradients.permute(0, 2, 3, 1)
    batch_x = batch_x.permute(0, 2, 3, 1)

    # Squeeze + move to cpu
    np_integrated_gradients = integrated_gradients[0, :, :, :].cpu().data.numpy()
    batch_x = batch_x[0, :, :, :].cpu().data.numpy()
    np_integrated_gradients = np.fabs(np_integrated_gradients)

    # normalize amplitudes
    np_integrated_gradients = np_integrated_gradients / np.max(np_integrated_gradients)

    # Overlay integrated gradient with image
    # images = [batch_x, np_integrated_gradients]
    # plot_integrated_gradients_results(batch_x, np_integrated_gradients, means, stds)
    return np_integrated_gradients

def get_shap(model, image, device): # TODO: debug this
    topk = 1 # top k class
    batch_size = 5
    n_evals = 100
    func = lambda x: F.softmax(model(torch.Tensor(x).to(device)), dim=1).detach().cpu().numpy()
    model.eval()

    # define a masker that is used to mask out partitions of the input image.
    masker_blur = shap.maskers.Image("blur(32,32)", image[0].shape)
    # create an explainer with model and image masker
    explainer = shap.Explainer(func, masker_blur, output_names=[0, 1])

    # feed only one image
    # here we explain two images using 100 evaluations of the underlying model to estimate the SHAP values
    shap_values = explainer(
        image,
        outputs=shap.Explanation.argsort.flip[:topk],
    )
    # mi, ma = shap_values.values[0].min(), shap_values.values[0].max()
    # plt.imshow(np.transpose(shap_values.values[0,:,:,:,0], (1, 2, 0))/(ma-mi)*255)
    # plt.show()
    # shap_values.data = np.transpose(inverse_transform(shap_values.data, means, stds).cpu().numpy()[0], (1, 2, 0))
    # shap_values.values = [np.transpose(val, (1, 2, 0)) for val in np.moveaxis(shap_values.values[0], -1, 0)]
    # print(shap_values.data.shape, shap_values.values[0].shape)
    # shap.image_plot(
    #     shap_values=shap_values.values,
    #     pixel_values=shap_values.data,
    #     labels=shap_values.output_names,
    #     true_labels=[1],
    # )

    return shap_values.values

def get_grad_cam(model, image, device):
    target_layers = [model.resnet.layer4[-1]]
    input_tensor = image # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor)

    return grayscale_cam

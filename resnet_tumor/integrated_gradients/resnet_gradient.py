import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import json
from pathlib import Path
import os

def compute_integrated_gradient(batch_x, batch_blank, model, idx):
    mean_grad = 0
    n = 100

    for i in tqdm(range(1, n + 1)):
        x = batch_blank + i / n * (batch_x - batch_blank)
        x.requires_grad = True
        y = model(x)[0, idx]
        (grad,) = torch.autograd.grad(y, x)
        mean_grad += grad / n

    integrated_gradients = (batch_x - batch_blank) * mean_grad

    return integrated_gradients


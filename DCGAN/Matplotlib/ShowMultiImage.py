import numpy as np
import matplotlib as mpt
from matplotlib import pyplot as plt
from matplotlib import image as matImg
import matplotlib.gridspec as gridspec
from torchvision import transforms as trf
import torch
plt.rcParams['figure.figsize'] = (10.0, 8.0) # 设置画图的尺寸
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
def show_images(images): # 定义画图工具 num * 3 * size * size

    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = images.shape[2]

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        img = torch.Tensor(img).permute(-2, -1, -3)
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img)
    return 
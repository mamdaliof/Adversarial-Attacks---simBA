import torch
import numpy as np
import torchvision.transforms as trans
import math
from scipy.fftpack import dct, idct

# mean and std for different datasets
IMAGENET_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_TRANSFORM = trans.Compose([
    trans.Resize(256),
    trans.CenterCrop(224),
    trans.ToTensor()])

INCEPTION_SIZE = 299
INCEPTION_TRANSFORM = trans.Compose([
    trans.Resize(342),
    trans.CenterCrop(299),
    trans.ToTensor()])

CIFAR_SIZE = 32
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
CIFAR_TRANSFORM = trans.Compose([
    trans.ToTensor()])

MNIST_SIZE = 28
MNIST_MEAN = [0.5]
MNIST_STD = [1.0]
MNIST_TRANSFORM = trans.Compose([
    trans.ToTensor()])


# reverses the normalization transformation
def invert_normalization(imgs, dataset):
    if dataset == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif dataset == 'cifar':
        mean = CIFAR_MEAN
        std = CIFAR_STD
    elif dataset == 'mnist':
        mean = MNIST_MEAN
        std = MNIST_STD
    imgs_trans = imgs.clone()
    if len(imgs.size()) == 3:
        for i in range(imgs.size(0)):
            imgs_trans[i, :, :] = imgs_trans[i, :, :] * std[i] + mean[i]
    else:
        for i in range(imgs.size(1)):
            imgs_trans[:, i, :, :] = imgs_trans[:, i, :, :] * std[i] + mean[i]
    return imgs_trans


# applies the normalization transformations
def apply_normalization(imgs, dataset):
    if dataset == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif dataset == 'cifar':
        mean = CIFAR_MEAN
        std = CIFAR_STD
    elif dataset == 'mnist':
        mean = MNIST_MEAN
        std = MNIST_STD
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]
    imgs_tensor = imgs.clone()
    if dataset == 'mnist':
        imgs_tensor = (imgs_tensor - mean[0]) / std[0]
    else:
        if imgs.dim() == 3:
            for i in range(imgs_tensor.size(0)):
                imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - mean[i]) / std[i]
        else:
            for i in range(imgs_tensor.size(1)):
                imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - mean[i]) / std[i]
    return imgs_tensor

# applies DCT to each block of size block_size
def block_dct(x, block_size=8, masked=False, ratio=0.5):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)].numpy()
            submat_dct = dct(dct(submat, axis=2, norm='ortho'), axis=3, norm='ortho')
            if masked:
                submat_dct = submat_dct * mask
            submat_dct = torch.from_numpy(submat_dct)
            z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = submat_dct
    return z

# applies IDCT to each block of size block_size
def block_idct(x, block_size=224, masked=False, ratio=0.5, linf_bound=0.0):
    """GPU-accelerated IDCT that processes everything in parallel."""
    # Ensure x is on GPU
    device = x.device
    batch, channels, h, w = x.shape
    
    # Vectorized masking: No loops!
    if masked:
        mask = torch.zeros((batch, channels, block_size, block_size), device=device)
        # Handle both float and per-image ratio
        if isinstance(ratio, float):
            r = int(block_size * ratio)
            mask[:, :, :r, :r] = 1
        else:
            for i in range(batch):
                r = int(block_size * ratio[i])
                mask[i, :, :r, :r] = 1
        x = x * mask

    # native PyTorch IDCT (using FFT-based implementation)
    # This replaces the slow scipy.idct loop
    z = torch.fft.irfft2(x, s=(h, w), norm='ortho')

    if linf_bound > 0:
        return z.clamp(-linf_bound, linf_bound)
    return z


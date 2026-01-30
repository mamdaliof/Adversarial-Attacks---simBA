# defense_hidde.py
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import random
from types import SimpleNamespace


class PreprocessingDefense(nn.Module):
    """
    Resize → Random Gaussian Blur → Logit Averaging
    Works on batches: (B, 3, 224, 224)
    """

    def __init__(
        self,
        model,
        resize_inner=200,
        blur_kernel=5,
        sigma_min=0.8,
        sigma_max=1.2,
        num_samples=5,
        device="cuda",
    ):
        super().__init__()
        self.model = model
        self.resize_inner = resize_inner
        self.blur_kernel = blur_kernel
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_samples = num_samples
        self.device = device

    @torch.no_grad()
    def preprocess_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        """
        processed = []

        for img in x:
            # resize down → up
            img = TF.resize(img, self.resize_inner)
            img = TF.resize(img, 224)

            # randomized Gaussian blur
            sigma = random.uniform(self.sigma_min, self.sigma_max)
            img = TF.gaussian_blur(
                img,
                kernel_size=self.blur_kernel,
                sigma=sigma,
            )
            processed.append(img)

        return torch.stack(processed).to(self.device)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Modified to handle Hugging Face objects and extract raw logits
        """
        x = pixel_values 
        logits_list = []

        for _ in range(self.num_samples):
            x_p = self.preprocess_batch(x)
            out = self.model(pixel_values=x_p) 
            
            logits_list.append(out.logits)

        avg_logits = torch.mean(torch.stack(logits_list), dim=0)
        return SimpleNamespace(logits=avg_logits)

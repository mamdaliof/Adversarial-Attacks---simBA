import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import random
from types import SimpleNamespace

class PreprocessingDefense(nn.Module):
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
        Preprocess entire batch (B, 3, 224, 224) at once.
        """
        img = TF.resize(x, [self.resize_inner, self.resize_inner])
        img = TF.resize(img, [224, 224])

        sigma = random.uniform(self.sigma_min, self.sigma_max)
        img = TF.gaussian_blur(img, kernel_size=self.blur_kernel, sigma=sigma)
        
        return img.to(self.device)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor = None, **kwargs) -> torch.Tensor:
        x = pixel_values.to(self.device)
        batch_size = x.shape[0]
        
        # Duplicate the batch num_samples times
        x_expanded = x.repeat_interleave(self.num_samples, dim=0)
        x_p = self.preprocess_batch(x_expanded)
        out = self.model(pixel_values=x_p) 
        
        logits = out.logits.view(batch_size, self.num_samples, -1)
        avg_logits = torch.mean(logits, dim=1)

        return SimpleNamespace(logits=avg_logits)
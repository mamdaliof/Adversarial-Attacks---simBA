# Adversarial Attacks on ConvNext Models using SimBA

A complete pipeline for running **SimBA** (Simple Black-box Adversarial) attacks on ConvNext models, with optional fine-tuning and multiple defense mechanisms.

---

## Table of Contents

- [Adversarial Attacks on ConvNext Models using SimBA](#adversarial-attacks-on-convnext-models-using-simba)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [CLI Arguments](#cli-arguments)
  - [Modules](#modules)
    - [models.py](#modelspy)
    - [fine\_tune.py](#fine_tunepy)
    - [simba\_attack.py](#simba_attackpy)
    - [defense.py](#defensepy)
  - [Notes](#notes)
  - [References](#references)
  - [License](#license)

---

## Overview

- Load pretrained ConvNext models (Tiny / Small / Base / Large)
- Fine-tune with warm-up + cosine-annealing scheduler
- Run query-efficient SimBA attacks (untargeted or targeted)
- Apply defenses: input transformations, ensemble voting, adversarial detection
- Evaluate and save results

---

## Project Structure

```
models.py          # Model loading utilities and ModelWrapper
fine_tune.py       # Trainer with warm-up scheduler, early stopping, mixed precision
simba_attack.py    # SimBA attack implementation
defense.py         # Defense mechanisms
main.py            # CLI entry point (orchestrates the full pipeline)
requirements.txt   # Dependencies
README.md          # This file
```

---

## Installation

```bash
git clone https://github.com/mamdaliof/Adversarial-Attacks---simBA.git
cd Adversarial-Attacks---simBA
pip install -r requirements.txt
```

Requires **Python 3.8+** and a CUDA GPU (recommended).

---

## Quick Start

```bash
# Basic attack on CIFAR-10 with input-transform defense
python main.py --dataset cifar10 --attack --defense input_transform

# Fine-tune then attack
python main.py --dataset cifar10 --fine-tune --epochs 10 --attack

# Targeted attack with custom epsilon
python main.py --dataset cifar10 --attack --targeted --epsilon 0.1
```

---

## CLI Arguments

Run `python main.py --help` for the latest flags.

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `convnext_tiny` | Model variant (`convnext_tiny`, `small`, `base`, `large`) |
| `--pretrained` | `True` | Use ImageNet-pretrained weights |
| `--num-classes` | `1000` | Number of output classes |
| `--data-path` | `./data` | Dataset root directory |
| `--dataset` | `imagenet` | Dataset (`imagenet`, `cifar10`, `cifar100`, `custom`) |
| `--batch-size` | `32` | Batch size |
| `--num-workers` | `4` | DataLoader workers |
| `--fine-tune` | `False` | Fine-tune before attack |
| `--epochs` | `10` | Fine-tuning epochs |
| `--learning-rate` | `1e-4` | Learning rate |
| `--warmup-epochs` | `2` | Warm-up epochs |
| `--attack` | `True` | Run SimBA attack |
| `--epsilon` | `0.2` | Max perturbation magnitude |
| `--max-iterations` | `10000` | Max attack iterations |
| `--targeted` | `False` | Targeted attack |
| `--max-samples` | `100` | Samples to attack |
| `--defense` | `input_transform` | Defense: `input_transform`, `ensemble`, `detector`, `none` |
| `--device` | auto | `cuda` if available, else `cpu` |
| `--seed` | `42` | Random seed |
| `--output-dir` | `./results` | Output directory |
| `--checkpoint-path` | `None` | Path to model checkpoint |

---

## Modules

### models.py

```python
from models import create_convnext_model

model = create_convnext_model('convnext_tiny', pretrained=True, num_classes=1000)
```

`ModelWrapper` expects **externally normalized** inputs (use torchvision weights' transforms).

### fine_tune.py

```python
from fine_tune import ModelTrainer

trainer = ModelTrainer(model, device='cuda')
history = trainer.fit(train_loader, val_loader, epochs=50, warmup_epochs=5)
```

Features: warm-up scheduler, cosine annealing, early stopping, mixed precision, checkpointing.

### simba_attack.py

```python
from simba_attack import SimBAAttack

attack = SimBAAttack(model, epsilon=0.2, max_iterations=10000, targeted=False)
x_adv, success, queries, perturbation = attack.attack(x, y_true)
```

### defense.py

```python
from defense import InputTransformationDefense

defense = InputTransformationDefense(model, transforms_list=['jpeg_compression', 'gaussian_blur'])
preds, probs = defense.predict(x)
```

Available defenses:
- **InputTransformationDefense** — JPEG compression, bit-depth reduction, Gaussian blur
- **EnsembleDefense** — soft/hard voting across models
- **AdversarialDetector** — confidence or reconstruction-based detection

---

## Notes

1. **Input normalization**: `ModelWrapper` does *not* normalize inputs internally. Preprocess images (including normalization) before passing them to the model, e.g., via the torchvision weights' transforms.
2. **ImageNet**: `get_data_loader` returns `None` for ImageNet; provide your own loader or use CIFAR for quick tests.
3. **Ensemble diversity**: The default ensemble duplicates the same model — for real defense, use distinct checkpoints or model variants.

---

## References

- Guo, C., et al. *Simple Black-box Adversarial Attacks*, ICML 2019.
- Liu, Z., et al. *A ConvNet for the 2020s*, CVPR 2022.

---

## License

MIT
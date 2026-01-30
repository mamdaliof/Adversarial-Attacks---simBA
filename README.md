# Adversarial Attacks on ConvNext Models using SimBA

A complete pipeline for running **SimBA** (Simple Black-box Adversarial) attacks on ConvNext models, with Gaussian Smoothing as defence mechanism.

---

## Table of Contents

- [Adversarial Attacks on ConvNext Models using SimBA](#adversarial-attacks-on-convnext-models-using-simba)
  - [Overview](#overview)
  - [Installation](#installation)
  - [License](#license)

---

## Overview

This github repository serves as the main code repo for the Deep Learning course 25/26 at the University of Twente.

What's in here:
- Load pretrained ConvNext models (Tiny / Small / Base / Large)
- Run query-efficient SimBA attacks (untargeted or targeted)
- Apply Gaussian Smoothing Defence
- Evaluate and save results
---

## Project Structure

```
simba.py           # SimBA attack implementation, from original paper
defense.py         # Defense mechanisms
main.py            # Main file to run
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
## References

- Guo, C., et al. *Simple Black-box Adversarial Attacks*, ICML 2019.
- Liu, Z., et al. *A ConvNet for the 2020s*, CVPR 2022.

---

## License

MIT
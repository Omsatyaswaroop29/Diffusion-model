# Diffusion Model on MNIST

A complete, from-scratch PyTorch implementation of a class-conditional Diffusion Model for generating MNIST handwritten digits. This repository showcases the latest generative AI techniques, featuring a custom U-Net architecture, time and class conditioning, and an interactive sampling interface.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Diffusion Model Theory](#diffusion-model-theory)
- [Implementation Highlights](#implementation-highlights)
  - [Noise Scheduling](#noise-scheduling)
  - [U-Net Architecture](#u-net-architecture)
  - [Class Conditioning](#class-conditioning)
- [Training](#training)
  - [Loss Function](#loss-function)
  - [Results](#results)
- [Sampling & Generation](#sampling--generation)
  - [Random & Conditional Generation](#random--conditional-generation)
  - [Text Prompt Interface](#text-prompt-interface)
- [Visualization](#visualization)
- [Technical Insights](#technical-insights)
- [Limitations & Future Directions](#limitations--future-directions)
- [Practical Applications](#practical-applications)
- [How to Run](#how-to-run)
- [References](#references)

---

## Project Overview

This project implements a Denoising Diffusion Probabilistic Model (DDPM) for MNIST digit generation. Diffusion models are generative models that learn to create new data by reversing a multi-step noising process. Unlike GANs, diffusion models offer stable training and produce high-quality, diverse outputs.

**Key Features:**
- Full forward and reverse diffusion process
- Custom U-Net with time and class conditioning
- Interactive generation for digits 0-9 and text prompts
- Training and sampling implemented from scratch

---

## Diffusion Model Theory

### How Diffusion Models Work

- **Forward Process:** Gradually adds Gaussian noise to real images over T timesteps, transforming real data into pure noise.
  - Equation:  
    `q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)`
- **Reverse Process:** The model learns to remove noise step by step, generating new samples from noise.
  - Equation:  
    `p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)`

---

## Implementation Highlights

### Noise Scheduling

Implemented with a linear schedule for β values:

```python
class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
```
- **Timesteps:** 1000
- **Beta Range:** 0.0001 to 0.02

### U-Net Architecture

- **Input:** 28×28 grayscale images + timestep + class label
- **Encoder:** Downsampling (28×28 → 7×7)
- **Bottleneck:** Feature processing
- **Decoder:** Upsampling with skip connections
- **Time Embeddings:** Sinusoidal positional encodings (128 → 256 dims)
- **Class Conditioning:** Embedding for digits 0-9

### Class Conditioning

- Embedding layer for class labels
- Enables guided, class-conditional generation

---

## Training

### Loss Function

The model is trained to predict the noise added at each timestep:

- **Objective:**  
  `L_simple = E[||ε - ε_θ(x_t, t, c)||²]`
- **Optimizer:** Adam, lr=1e-3
- **Batch Size:** 128
- **Epochs:** 10

### Results

- **Rapid Convergence:** Loss drops from 0.067 to 0.024 over 10 epochs
- **Stable Training:** No mode collapse, consistent improvement
- **Performance:** ~7-8 iterations/sec, 12min total training time

---

## Sampling & Generation

### Sampling Algorithm

Implements standard DDPM denoising:

```python
@torch.no_grad()
def sample(model, num_samples=16, class_labels=None, timesteps=1000):
    x = torch.randn(num_samples, 1, 28, 28, device=device)
    for t in reversed(range(timesteps)):
        noise_pred = model(x, t_batch, class_labels)
        x = (1/√α_t) * (x - β_t/√(1-ᾱ_t) * noise_pred) + σ_t * noise
```

### Random & Conditional Generation

- **Unconditional:** Generates random digits from pure noise
- **Conditional:** Generates specified digits (0-9) using class embeddings

### Text Prompt Interface

- Accepts both digit (e.g., "3") and text (e.g., "seven") prompts
- Interactive, real-time generation

---

## Visualization

- **Random Generation:** 4×4 grid, diverse digits
- **Class-Conditioned:** 2×5 grid, each digit 0-9
- **Text Prompts:** Recognizable digits from both text and number inputs

---

## Technical Insights

- **Pure PyTorch Implementation:** No external diffusion libraries used
- **Mathematical Rigor:** Forward/reverse processes, noise scheduling, DDPM sampling
- **Model Size:** ~6.1M parameters, efficient GPU utilization
- **Interactive Interface:** User-friendly, error handling, real-time feedback

---

## Limitations & Future Directions

### Current Constraints

- MNIST only (28×28 grayscale)
- Simple U-Net/no attention
- No fast sampling (full 1000 steps)

### Planned Improvements

- Higher resolution (64×64, 128×128)
- DDIM, DPM-Solver, consistency models for faster sampling
- Text-to-image with CLIP embeddings and cross-attention
- Latent diffusion for compressed space and speed
- Self- and cross-attention, transformer-based architectures

---

## Practical Applications

This project is an ideal learning platform for:
- Understanding diffusion theory and generative modeling
- Hands-on experience with modern AI architectures
- Exploring conditional generation and interactive interfaces
- Building production-grade PyTorch code

---

## How to Run

### Requirements

- Python 3.8+
- PyTorch
- torchvision

### Setup

```bash
git clone https://github.com/Omsatyaswaroop29/Diffusion-model.git
cd Diffusion-model
pip install -r requirements.txt
```

### Training

Run the training script to train from scratch:

```bash
python diffusion_model_mnist.py --train
```

### Sampling

Generate digits or use interactive sampling:

```bash
python diffusion_model_mnist.py --sample  # Generates random samples

python diffusion_model_mnist.py --prompt "seven"  # Generates the digit "7"
```

### More Options

See all available options:

```bash
python diffusion_model_mnist.py --help
```

---

## References

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

## License

This project is licensed under the MIT License.

---

## Author

- [Omsatyaswaroop29](https://github.com/Omsatyaswaroop29)

---

**If you find this repository helpful, please star it!**

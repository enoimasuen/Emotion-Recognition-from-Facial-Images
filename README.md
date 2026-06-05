# Emotion Recognition from Facial Images

### Does ImageNet transfer learning beat a CNN trained from scratch on FER-2013?

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-FER--2013-success)
![Status](https://img.shields.io/badge/Course-CS1090B%20Spring%202026-blue)

A study of facial emotion recognition (FER) on the **FER-2013** benchmark, comparing a custom CNN trained from scratch against ImageNet-pretrained transfer learning (ResNet-50, VGG-16, EfficientNet-B0). We run a four-phase experimental program covering a CNN capacity study, interpolation/upscaling methods, a multi-model benchmark, and a systematic ablation of augmentation, loss functions, hyperparameters, and fine-tuning depth — with Grad-CAM interpretability throughout.

> **TL;DR** — Transfer learning wins decisively. ResNet-50 with the **last two blocks + head fine-tuned** reaches **0.696 weighted F1 (~70% accuracy)**, versus **0.194 weighted F1** for the best from-scratch CNN — a **3.6× improvement**. Architecture is the dominant driver; interpolation method barely matters; and **full** fine-tuning is *catastrophic* on this noisy, imbalanced dataset.

---

## Table of contents

- [Research question](#research-question)
- [Dataset](#dataset)
- [Key results](#key-results)
- [Findings at a glance](#findings-at-a-glance)
- [Repository structure](#repository-structure)
- [Setup & reproduction](#setup--reproduction)
- [Methods](#methods)
- [Interpretability (Grad-CAM)](#interpretability-grad-cam)
- [Limitations](#limitations)
- [Ethics & broader impact](#ethics--broader-impact)
- [References](#references)
- [Authors](#authors)

---

## Research question

> **Does ImageNet-pretrained transfer learning improve facial emotion classification over a baseline CNN trained from scratch on a low-resolution, class-imbalanced dataset?**

The target is a seven-class emotion label — `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise` — for a single 48×48 grayscale image. Because the dataset is severely imbalanced and contains label noise, a model that scores high accuracy by ignoring minority classes is not genuinely useful. We therefore report **weighted *and* macro F1** alongside accuracy and track **minority-class recall** throughout.

## Dataset

[**FER-2013**](https://www.kaggle.com/datasets/msambare/fer2013) — 35,887 grayscale 48×48 face images scraped via search-engine queries (Goodfellow et al., 2013). We use the canonical split (28,709 train / 7,178 test) without carving out a separate validation set, to maximize training data and stay comparable with prior work; all reported metrics are on the held-out test set.

The class distribution is heavily skewed (a **16:1** happy-to-disgust ratio):

| Emotion  | Train | Test  | Total | % of dataset |
|----------|------:|------:|------:|-------------:|
| Angry    | 3,995 |   958 | 4,953 |       13.8%  |
| Disgust  |   436 |   111 |   547 |        1.5%  |
| Fear     | 4,097 | 1,024 | 5,121 |       14.3%  |
| Happy    | 7,215 | 1,774 | 8,989 |       25.0%  |
| Neutral  | 4,965 | 1,233 | 6,198 |       17.3%  |
| Sad      | 4,830 | 1,247 | 6,077 |       16.9%  |
| Surprise | 3,171 |   831 | 4,002 |       11.2%  |

**The dataset is not committed to this repo** (see [Setup](#setup--reproduction) for the download step).

## Key results

**Transfer learning vs. from-scratch — the headline comparison:**

| Phase | Model | Upscaling | Val. accuracy | Weighted F1 |
|-------|-------|-----------|:-------------:|:-----------:|
| Phase 1 — Baseline CNN | BaselineCNN | Nearest N. | ~0.31 | 0.194 |
| **Phase 2 — ResNet-50 TL** | **ResNet-50** | **Lanczos** | **~0.70** | **0.696** |
| Phase 3 — Multi-model | VGG-16 | Bicubic | ~0.57 | 0.574 |
| Phase 3 — Multi-model | ResNet-50 | Bilinear | ~0.58 | 0.580 |
| Phase 3 — Multi-model | EfficientNet-B0 | Bilinear | ~0.55 | 0.553 |
| Phase 3 — Multi-model | BaselineCNN | Bicubic | ~0.25 | 0.046 |

Transfer-learning models cluster tightly between **0.55–0.70** weighted F1; the from-scratch CNN never gets close.

**Fine-tuning depth — the most practically important ablation:**

| Strategy | ResNet-50 F1 | VGG-16 F1 |
|----------|:------------:|:---------:|
| Head only | 0.361 | 0.339 |
| Last block + head | 0.535 | 0.554 |
| **Last 2 blocks + head** | **0.579** | **0.571** |
| Full fine-tune | 0.126 | 0.147 |

Updating *all* parameters on noisy FER-2013 labels collapses the pretrained representation. **Partial fine-tuning is essential, not optional.**

**From-scratch CNN capacity study (the baseline these numbers improve on):**

| Model | Accuracy | Macro F1 | Weighted F1 | Precision (M) | Recall (M) |
|-------|:--------:|:--------:|:-----------:|:-------------:|:----------:|
| Run1_Small_BN | 0.516 | 0.429 | 0.512 | 0.424 | 0.436 |
| Run2_Baseline_BN | 0.598 | 0.489 | 0.579 | 0.492 | 0.499 |
| Run3_Deep_BN | 0.561 | 0.463 | 0.551 | 0.522 | 0.457 |
| **Run4_HighCapacity_BN** | **0.611** | **0.511** | **0.598** | **0.644** | **0.520** |

Capacity helps up to a point, but the persistent accuracy-vs-macro-F1 gap exposes a stubborn majority-class bias — `disgust` recall stays near zero regardless of capacity.

## Findings at a glance

1. **Architecture dominates.** Pretrained backbones beat a tuned from-scratch CNN by a wide, unambiguous margin (3.6× weighted F1).
2. **Interpolation barely matters.** Across nearest-neighbor, bilinear, bicubic, and Lanczos, ResNet-50's weighted F1 spans only ~2.7%. Bicubic/Lanczos are marginally best — no learned super-resolution needed.
3. **Partial fine-tuning is essential.** Last-2-blocks + head is optimal; full fine-tuning is catastrophic (0.579 → 0.126 for ResNet-50).
4. **Moderate augmentation captures most of the benefit.** Flip + rotation + color jitter does the heavy lifting; translation, random crop, and random erasing show diminishing returns.
5. **Loss choice is secondary once the backbone is adapted.** For ResNet-50, weighted CE, standard CE, and focal loss land within ~1.6% of each other — the pretrained features are robust enough that loss function is not the lever. (On the from-scratch CNN, a `WeightedRandomSampler` gave the biggest minority-class lift.)
6. **Hyperparameter sensitivity.** Adam ≫ SGD; learning rate is the sensitive knob (1e-3 best); weight decay has negligible effect.
7. **Grad-CAM confirms genuine feature learning.** Transfer-learning models attend to semantically meaningful regions — mouth for `happy`, brow/eyes for `angry`/`surprise` — not background artifacts.

## Repository structure

```
Emotion-Recognition-from-Facial-Images/
├── README.md
├── requirements.txt
├── .gitignore
├── cs1090b_ms4_main_group4.ipynb     # main notebook (CNN ablation + transfer learning)
├── docs/
│   ├── cs109b_ms4_report_group4.pdf  # full written report
│   └── MS4_Group4.pptx               # presentation slides
└── archive/                          # FER-2013 — NOT committed; download separately
    ├── train/<emotion>/*.jpg
    └── test/<emotion>/*.jpg
```

The notebook is organized into two parts:

- **Part 1 — CNN Ablation Study (`Run4` baseline):** enhanced augmentation pipeline, weighted random sampling, loss-function / learning-rate / optimizer / weight-decay sweeps, results, and Grad-CAM for the best CNN.
- **Part 2 — Transfer Learning:** dataset exploration, upscaling-method comparison, shared training utilities, baseline-CNN and ResNet-50 interpolation studies, multi-model benchmark, augmentation ablation, class-imbalance strategies, hyperparameter sensitivity, partial fine-tuning depth, and Grad-CAM explainability.

## Setup & reproduction

**1. Clone and install dependencies** (Python 3.12 recommended):

```bash
git clone https://github.com/enoimasuen/Emotion-Recognition-from-Facial-Images.git
cd Emotion-Recognition-from-Facial-Images

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Download the dataset.** Grab FER-2013 from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and extract it so the folder layout matches what the notebook expects (`archive/train/<emotion>/` and `archive/test/<emotion>/`). Using the Kaggle CLI:

```bash
kaggle datasets download -d msambare/fer2013 -p archive --unzip
```

After extraction you should have `archive/train/` and `archive/test/`, each with seven emotion subfolders.

**3. Run the notebook** — launch Jupyter from the repository root so the relative `archive/` path resolves correctly:

```bash
jupyter notebook cs1090b_ms4_main_group4.ipynb
```

> **GPU strongly recommended.** The transfer-learning runs fine-tune ResNet-50 / VGG-16 / EfficientNet-B0 on 224×224 inputs across many configurations; the notebook uses CUDA automatically when available and falls back to CPU otherwise.

## Methods

**Preprocessing.** Images are loaded as 3-channel RGB and normalized with ImageNet statistics for transfer-learning models (or grayscale-normalized to mean 0.5 / std 0.5 for the baseline CNN). Because pretrained backbones expect 224×224 inputs, 48×48 images are upscaled with classical interpolation — we compare **nearest-neighbor, bilinear, bicubic, and Lanczos**.

**Baseline CNN.** Four architectures of increasing capacity (`Conv2d → BatchNorm2d → ReLU → MaxPool2d` blocks) trained from scratch: `Run1` (small, underfit check) → `Run2` (baseline) → `Run3` (deep) → `Run4` (high-capacity stress test).

**Transfer learning.** Three torchvision backbones pretrained on ImageNet-1K — ResNet-50 (25.6M params), VGG-16 (138M), EfficientNet-B0 (5.3M) — with the head replaced by a seven-class linear layer. Two-stage protocol: **Stage 1** (5 epochs) trains the head with the backbone frozen (Adam, lr 1e-3); **Stage 2** (15 epochs) fine-tunes with a lower backbone lr (1e-4) and cosine annealing.

**Class-imbalance strategies.** Standard CE, weighted CE, focal loss (γ=2), `WeightedRandomSampler` + CE, and WRS + weighted CE.

**Fine-tuning depth.** Head-only → last-block + head → last-2-blocks + head → full fine-tune, each trained 20 epochs with cosine LR annealing.

## Interpretability (Grad-CAM)

We apply [Grad-CAM](https://arxiv.org/abs/1610.02391) (Selvaraju et al., 2017) to the best CNN and the best ResNet-50. The baseline CNN's attention is diffuse and spreads across the central face; ResNet-50's attention is far more semantically coherent — concentrating on the smile region for `happy`, the brow-eye region for `angry`, and widened eyes for `surprise`. This is qualitative evidence that transfer learning produces genuinely emotion-relevant features rather than higher-accuracy pattern matching. Misclassifications (e.g. `disgust → surprise`, `fear → angry`) tend to coincide with attention on incorrect or peripheral regions.

## Limitations

Results are confined to a single benchmark under a fixed set of architectures. FER-2013 has well-documented label noise, so some findings (e.g. Adam ≫ SGD) may not transfer to datasets with different noise profiles. We do not measure inference latency or model size, and we did not evaluate Vision Transformers or newer efficient backbones. Most fundamentally, FER-2013's hard one-hot labels poorly reflect the continuous, ambiguous nature of human emotion — `fear` and `surprise` share overlapping facial configurations, and a single image can plausibly carry multiple labels. Soft re-labeling from annotator-agreement distributions, or joint valence/arousal prediction, would better match this structure.

## Ethics & broader impact

Emotion inference from faces is **not reliable across demographic groups** — published work has found higher error rates for darker-skinned faces and for women in commercial FER systems (Buolamwini & Gebru, 2018). FER-2013 was scraped without demographic controls and may not represent diverse populations, so a model trained on it could perpetuate disparities in high-stakes settings (hiring, policing, mental-health triage), and could be misused for covert emotional surveillance. Any production deployment should include demographic bias audits before release, explicit consent from monitored individuals, human oversight for consequential decisions, and opt-out mechanisms. **This is a research study on a public benchmark, not a production-ready system.**

## References

1. Goodfellow et al. (2013). *Challenges in Representation Learning: A report on three machine learning contests.* arXiv:1307.0414.
2. He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
3. Simonyan & Zisserman (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* arXiv:1409.1556.
4. Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
5. Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
6. Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.
7. Buolamwini & Gebru (2018). *Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification.* FAT* 2018, PMLR 81, 77–91.
8. [FER-2013 dataset (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013) · [torchvision.models](https://pytorch.org/vision/stable/models.html)

## Authors

**Group 4 — CS1090B, Spring 2026**
Daniel Wei · Enogayin Imasuen · Suffian Haroon

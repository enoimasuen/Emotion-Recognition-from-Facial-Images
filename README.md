# Emotion Recognition from Facial Images

### Does ImageNet transfer learning beat a CNN trained from scratch on FER-2013?

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-FER--2013-success)
![Status](https://img.shields.io/badge/Course-CS1090B%20Spring%202026-blue)

A study of facial emotion recognition (FER) on the **FER-2013** benchmark, comparing a custom CNN trained from scratch against ImageNet-pretrained transfer learning (ResNet-50, VGG-16, EfficientNet-B0). We run a four-phase experimental program covering a CNN capacity study, interpolation/upscaling methods, a multi-model benchmark, and a systematic ablation of augmentation, loss functions, hyperparameters, and fine-tuning depth — with Grad-CAM interpretability throughout.

> **TL;DR:** The strongest result came from ResNet-50 trained on the full 28,709-image training set with a two-stage protocol: five epochs of head-only training followed by fifteen epochs of full-network fine-tuning with differential learning rates. It achieved **0.6955 weighted F1** and **0.6950 test accuracy**. The strongest MS3 high-capacity CNN achieved **0.598 weighted F1** and **0.611 accuracy**, so ResNet-50 improved weighted F1 by **0.0975** and accuracy by **8.4 percentage points**. Phases 3 and 4 used a smaller 3,436-image class-capped subset, so those results are reported separately from the full-data Phase 1 and Phase 2 experiments.

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

[**FER-2013**](https://www.kaggle.com/datasets/msambare/fer2013) — 35,887 grayscale 48×48 face images collected through search-engine queries (Goodfellow et al., 2013). Phases 1 and 2 use the full canonical training split of 28,709 images. To make the 16-run Phase 3 benchmark and the Phase 4 systematic ablations computationally tractable, both phases use a class-capped subset of 3,436 images: up to 500 images per class, with all 436 available disgust images retained. Results are only directly comparable within the same training set and protocol.

The project did not create a separate validation set. Because the canonical test split was consulted repeatedly during model comparison, the reported values should be interpreted as exploratory test-set comparisons rather than an untouched final generalization estimate.

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

| Phase | Model | Best upscaling | Training data | Test accuracy | Weighted F1 |
|---|---|---|---:|---:|---:|
| Phase 1 — BaselineCNN interpolation study | BaselineCNN | Nearest Neighbor | 28,709 | 0.3172 | 0.1937 |
| Phase 2 — ResNet-50 transfer learning | ResNet-50 | Lanczos | 28,709 | **0.6950** | **0.6955** |
| Phase 3 — Multi-model benchmark | VGG-16 | Lanczos | 3,436 | 0.5775 | 0.5795 |
| Phase 3 — Multi-model benchmark | ResNet-50 | Bilinear | 3,436 | 0.5786 | 0.5804 |
| Phase 3 — Multi-model benchmark | EfficientNet-B0 | Bilinear | 3,436 | 0.5463 | 0.5527 |
| Phase 3 — Multi-model benchmark | BaselineCNN | Bicubic | 3,436 | 0.1457 | 0.0456 |

Phase 2 used the full 28,709-image training set, while Phase 3 used the smaller 3,436-image class-capped subset. Cross-phase performance gaps therefore reflect differences in training data and protocol, not interpolation or architecture alone.

Phase 4 contains the systematic augmentation, class-imbalance, hyperparameter, fine-tuning-depth, and Grad-CAM experiments. Its quantitative ablations use the same 3,436-image class-capped subset as Phase 3 and are summarized separately below.

### Interpreting the two from-scratch comparisons

**Controlled Phase 1 versus Phase 2 comparison.** Within the full-data 224×224 experimental pipeline, ResNet-50 achieved 0.6955 weighted F1 compared with 0.1937 for the Phase 1 BaselineCNN, approximately a 3.6× improvement. Both experiments used the full 28,709-image training set and the same general preprocessing and training framework. However, the Phase 1 BaselineCNN was substantially weaker than the separate MS3 high-capacity CNN.

**Comparison with the strongest from-scratch CNN.** The strongest MS3 CNN trained from scratch achieved 0.598 weighted F1, while the Phase 2 ResNet-50 achieved 0.6955. This represents an absolute improvement of 0.0975 and a relative improvement of approximately 16.3%. Because the MS3 CNN used the original 48×48 grayscale pipeline while Phase 2 used 224×224 upscaled RGB inputs and a different training protocol, this comparison is informative but not fully controlled.

### Phase 4 partial fine-tuning-depth study

| Strategy | ResNet-50 weighted F1 | VGG-16 weighted F1 |
|---|---:|---:|
| Head only | 0.3608 | 0.3386 |
| Last block + head | 0.5345 | 0.5540 |
| Last 2 blocks + head | **0.5787** | **0.5707** |

Among the three valid partial-depth conditions evaluated in Phase 4, unfreezing the last two blocks plus the classification head achieved the strongest weighted F1 for both ResNet-50 and VGG-16.

The originally attempted full-network fine-tuning run is excluded because a code error caused only the first parameter tensor to be unfrozen while the classification head remained frozen. Its cached results are invalid. Therefore, this ablation identifies the strongest tested partial-depth configuration but does not determine whether partial fine-tuning outperforms correctly implemented full-network fine-tuning.

**From-scratch CNN capacity study (the baseline these numbers improve on):**

| Model | Accuracy | Macro F1 | Weighted F1 | Precision (M) | Recall (M) |
|-------|:--------:|:--------:|:-----------:|:-------------:|:----------:|
| Run1_Small_BN | 0.516 | 0.429 | 0.512 | 0.424 | 0.436 |
| Run2_Baseline_BN | 0.598 | 0.489 | 0.579 | 0.492 | 0.499 |
| Run3_Deep_BN | 0.561 | 0.463 | 0.551 | 0.522 | 0.457 |
| **Run4_HighCapacity_BN** | **0.611** | **0.511** | **0.598** | **0.644** | **0.520** |

Capacity helps up to a point, but the persistent accuracy-vs-macro-F1 gap exposes a stubborn majority-class bias — `disgust` recall stays near zero regardless of capacity.

## Findings at a glance

1. **Transfer learning produced the strongest full-data result.** Phase 2 ResNet-50 achieved 0.6955 weighted F1 and 0.6950 exploratory test accuracy.

2. **The two from-scratch comparisons answer slightly different questions.** Within the full-data 224×224 pipeline, ResNet-50 achieved approximately 3.6× the weighted F1 of the weaker Phase 1 BaselineCNN. Compared with the strongest MS3 scratch CNN, ResNet-50 improved weighted F1 from 0.598 to 0.6955, a relative improvement of approximately 16.3%, although the models used different pipelines.

3. **Interpolation had limited impact within the Phase 3 ResNet-50 setup.** Weighted F1 ranged from 0.5678 to 0.5804, an absolute spread of 0.0126.

4. **Training-data volume matters.** Phases 1 and 2 used 28,709 training images, while Phases 3 and 4 used the smaller 3,436-image class-capped subset. Cross-phase differences should not be attributed to interpolation or architecture alone.

5. **Moderate augmentation captured most of the observed benefit.** Horizontal flipping, rotation, and color jitter provided most of the improvement, while additional transforms produced smaller gains.

6. **Weighted cross-entropy performed best in the Phase 4 class-imbalance study.** It achieved 0.5917 weighted F1, although the differences among the leading loss and sampling strategies were relatively small.

7. **The last two blocks plus head performed best among the valid partial-depth conditions.** The invalid full-network condition was excluded, so this ablation does not establish whether partial fine-tuning outperforms correctly implemented full-network fine-tuning.

8. **Learning rate had a greater effect than weight decay.** Adam outperformed SGD in the subset-based hyperparameter study, while changes in weight decay produced comparatively small differences.

9. **Grad-CAM provides qualitative evidence of plausible model attention.** ResNet-50 activations were often concentrated on facial regions such as the mouth and brow-eye area, but these visualizations do not prove causal feature use, robustness, or demographic fairness.

## Repository structure

```
Emotion-Recognition-from-Facial-Images/
├── README.md
├── requirements.txt
├── .gitignore
├── Facial Emotion Recognition Notebook.ipynb     # main notebook (CNN ablation + transfer learning)
├── docs/
│   ├── Facial Emotion Recognition Notebook.pdf  # full written report
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
jupyter notebook "Facial Emotion Recognition Notebook.ipynb"
```

> **GPU strongly recommended.** The transfer-learning runs fine-tune ResNet-50 / VGG-16 / EfficientNet-B0 on 224×224 inputs across many configurations; the notebook uses CUDA automatically when available and falls back to CPU otherwise.

## Methods

**Preprocessing.** Transfer-learning images are converted to three-channel RGB, resized from 48×48 to 224×224, and normalized using ImageNet statistics. The separate Phase 1 BaselineCNN operates on grayscale images upscaled to 224×224. The MS3 Run1–Run4 CNNs instead use the original 48×48 grayscale inputs. We compare nearest-neighbor, bilinear, bicubic, and Lanczos interpolation.

**Augmentation pipelines.** The Phase 4 transfer-learning ablation uses horizontal flipping, ±15° rotation, and brightness/contrast jitter as its baseline augmentation. The separate Part 1 MS3 CNN-ablation pipeline uses ±10° rotation together with scaling, translation, color jitter, random cropping, and random erasing.

**CNN models.** The MS3 capacity study evaluates four custom CNN architectures, `Run1` through `Run4`, trained from scratch on the original 48×48 grayscale images. A separate three-block `BaselineCNN` is used in Phase 1 and Phase 3 of the transfer-learning pipeline and operates on grayscale images upscaled to 224×224. These are different model families with different input pipelines and should not be treated as the same baseline.

**Transfer learning.** Three torchvision backbones pretrained on ImageNet-1K are evaluated: ResNet-50 (25.6M parameters), VGG-16 (138M), and EfficientNet-B0 (5.3M), each with its classification head replaced by a seven-class output layer. The Phase 2 protocol has two stages. Stage 1 freezes the ResNet-50 backbone and trains only the classification head for five epochs using Adam at a learning rate of `1e-3`. Stage 2 unfreezes the full network for fifteen epochs and uses differential learning rates: `1e-4` for the pretrained backbone and `5e-4` for the classification head, with weight decay of `1e-4` and cosine annealing.

**Class-imbalance strategies.** Phase 4 compares standard cross-entropy, weighted cross-entropy, focal loss with γ=2, `WeightedRandomSampler` plus standard cross-entropy, and `WeightedRandomSampler` plus weighted cross-entropy. The class weights for these experiments are computed from the 3,436-image class-capped subset.

**Fine-tuning depth.** The Phase 4 depth ablation compares three verified partial-depth conditions: head only, last block plus head, and last two blocks plus head. Each condition is trained for 20 epochs with cosine learning-rate annealing. The originally executed full-network branch was invalid and is excluded from this ablation. Separately, the Phase 2 protocol correctly unfreezes the full ResNet-50 after the five-epoch head-only warm-up.

## Interpretability (Grad-CAM)

We apply [Grad-CAM](https://arxiv.org/abs/1610.02391) (Selvaraju et al., 2017) to the best custom CNN and the best ResNet-50. The custom CNN generally shows more diffuse attention across the central face, while ResNet-50 activations are often more spatially concentrated around plausible diagnostic regions, including the mouth for `happy` and the brow-eye region for `angry` and `surprise`. Misclassifications such as `disgust → surprise` and `fear → angry` sometimes coincide with attention on peripheral or less relevant regions.

These visualizations provide qualitative evidence that ResNet-50 attends to semantically plausible facial regions, but they do not prove causal feature use, model robustness, or demographic fairness.

## Limitations

The canonical test split was consulted during model comparison, so reported scores should be interpreted as exploratory rather than as an untouched estimate of generalization. Experimental scale also differed across phases: Phases 1 and 2 used all 28,709 training images, while Phases 3 and 4 used a 3,436-image class-capped subset, meaning cross-phase differences reflect both data volume and experimental design. The Phase 4 full-network fine-tuning run was invalid and excluded, so this study does not establish whether partial fine-tuning outperforms correctly implemented full-network fine-tuning. FER-2013 also contains noisy, ambiguous labels and lacks controlled demographic metadata, limiting conclusions about performance across race/ethnicity, skin tone, gender, and other subgroups. We additionally do not evaluate inference latency, deployment efficiency, Vision Transformers, or newer efficient backbones. Grad-CAM is treated strictly as a qualitative interpretability tool: its heatmaps indicate where model activations concentrate, but they do not establish causal feature use, robustness, or demographic fairness.

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
Daniel Wei · Enoghayin Imasuen · Suffian Haroon

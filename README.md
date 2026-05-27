# Adaptive Multi-Teacher Cross-Modal Self-Supervised Learning for Echocardiography Video Representation Learning

<p align="center">
  <strong>MT-DISCOVR</strong><br/>
  Exploratory self-supervised learning research for cardiac ultrasound video representation learning.
</p>

<p align="center">
  <a href="https://github.com/prajithJ/DL_Grp_49"><img src="https://img.shields.io/badge/Project-MT--DISCOVR-0A66C2" alt="Project Badge"></a>
  <img src="https://img.shields.io/badge/Domain-Medical%20Video%20AI-2E8B57" alt="Domain Badge">
  <img src="https://img.shields.io/badge/Focus-Self--Supervised%20Learning-B22222" alt="SSL Badge">
  <img src="https://img.shields.io/badge/Data-EchoNet--Dynamic-6A5ACD" alt="Dataset Badge">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C" alt="PyTorch Badge">
</p>

## Table of Contents

- [Abstract](#abstract)
- [Motivation](#motivation)
- [Key Contributions](#key-contributions)
- [Method Overview](#method-overview)
- [Mathematical Formulation](#mathematical-formulation)
- [Dataset](#dataset)
- [Experimental Phases](#experimental-phases)
- [Results](#results)
- [Ablation Discussion](#ablation-discussion)
- [Installation](#installation)
- [Training Commands](#training-commands)
- [Finetuning Commands](#finetuning-commands)
- [Linear Probing Commands](#linear-probing-commands)
- [Future Work](#future-work)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Abstract

**MT-DISCOVR** extends the DISCOVR framework for echocardiography video self-supervised learning by introducing **adaptive multi-teacher cross-modal supervision**. The central premise is that a video model should not rely on a single image teacher when learning semantic structure from noisy cardiac ultrasound sequences. Instead, we investigate whether multiple image-based self-supervised teachers can provide richer and more stable supervision for a video SSL backbone.

Our framework combines a **VideoMAE-style student encoder** with a **DINO image teacher**, a secondary **SimSiam or RAD-DINO teacher**, a **barycentric teacher fusion mechanism**, and **reliability-aware temporal motion weighting**. The result is a research framework aimed at learning stronger spatio-temporal video representations from unlabeled echocardiography while remaining careful about uncertainty in rapidly changing frames.

This repository should be read as **exploratory representation learning research**, not as a final clinical system. The objective is to study how cross-modal distillation behaves under multi-teacher supervision in a challenging medical video domain where labels are scarce, dynamics are subtle, and semantic guidance is inherently imperfect.

## Motivation

Labelled medical video data is expensive to curate, and this is especially true for echocardiography, where expert annotation requires both cardiac expertise and significant time. In practice, most available ultrasound video data remains weakly labeled or completely unlabeled, making self-supervised learning an attractive direction for scalable representation learning.

Echocardiography is also a difficult SSL regime. Compared with natural image benchmarks, cardiac ultrasound videos are noisy, low-contrast, and temporally complex. Fine-grained anatomical cues may only be visible in a subset of frames, while rapid cardiac motion can change visual appearance enough to weaken direct frame-wise semantic alignment.

Existing SSL approaches often capture either temporal dynamics or spatial semantics well, but struggle to preserve both simultaneously. Cross-modal methods like DISCOVR address this by transferring semantic guidance from image learning into video learning, but they typically rely on a **single teacher signal**. MT-DISCOVR explores whether **multiple image teachers**, fused into a consensus distribution and reweighted by motion reliability, can offer a more informative supervision signal for video representation learning.

## Key Contributions

- **Adaptive multi-teacher cross-modal supervision** for echocardiography video SSL, extending DISCOVR beyond its original single-teacher semantic alignment scheme.
- **Barycentric teacher fusion** using the normalized geometric mean of teacher distributions, designed to preserve consensus and suppress disagreement.
- **Reliability-aware temporal motion weighting** that modulates cross-modal supervision strength based on frame-to-frame motion magnitude.
- **Exploratory integration of heterogeneous image teachers** including DINO and SimSiam or RAD-DINO within a unified probabilistic distillation framework.
- **A research-oriented evaluation pipeline** spanning pretraining, linear probing, and full finetuning on EchoNet-Dynamic.

## Method Overview

MT-DISCOVR builds upon **DISCOVR: Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation**, which itself combines:

- VideoMAE-style video self-supervision
- an online image encoder
- cross-modal supervision
- Semantic Cluster Distillation (SCD)

Our extension replaces the original single-teacher semantic alignment mechanism with **adaptive multi-teacher probabilistic distillation**.

### Architecture Components

The framework contains the following components:

- **VideoMAE student backbone** for spatio-temporal video representation learning
- **DINO image teacher** to provide strong image-level semantic targets
- **SimSiam / RAD-DINO second image teacher** to introduce complementary supervisory structure
- **Barycentric fusion module** to construct a consensus target distribution
- **Temporal motion reliability weighting** to reduce supervision strength on high-motion, less reliable frame transitions

### Design Intuition

The method is built around a simple observation: not every frame transition in echocardiography is equally reliable for image-driven supervision. During rapid motion or appearance change, teacher predictions can be less aligned with the evolving video context. Conversely, when motion is moderate and anatomy is visually stable, image teachers can provide stronger semantic anchors. MT-DISCOVR therefore uses both **teacher diversity** and **temporal reliability modulation** to shape the cross-modal signal.

## Mathematical Formulation

### Multi-Teacher Barycentric Fusion

Let two teacher distributions be denoted by `P1` and `P2`. We fuse them using:

```math
P^* = \operatorname{softmax}\left(\frac{\log P_1 + \log P_2}{2}\right)
```

which corresponds to the normalized geometric mean:

```math
P^*_k = \frac{\sqrt{P_{1,k} P_{2,k}}}{\sum_j \sqrt{P_{1,j} P_{2,j}}}
```

This formulation is useful because:

- the **geometric mean suppresses disagreement** more strongly than naive arithmetic averaging
- it produces **consensus-preserving supervision**, emphasizing classes or prototypes that both teachers support
- it yields **sharper targets than simple averaging**, which is desirable for distillation when teacher agreement is meaningful

In a medical SSL setting, this is attractive because different teachers may encode complementary semantics, but unreliable overlap should not be over-amplified.

### Temporal Motion Reliability Weighting

We define motion magnitude between adjacent frames using a feature-space distance:

```math
m_t = \lVert \phi(x_t) - \phi(x_{t+1}) \rVert_2
```

where `\phi(\cdot)` denotes a frame embedding function. A reliability weight is then assigned as:

```math
w_t = \exp(-\beta m_t)
```

The cross-modal distillation objective becomes:

```math
L_{cm} = w_t \cdot \mathrm{KL}(P^* \parallel Q)
```

where `Q` is the student video distribution.

This weighting has a clear interpretation:

- **high motion leads to weaker image supervision**, since image semantics may be less stable across abrupt temporal change
- **low motion leads to stronger teacher guidance**, since semantic alignment is more likely to be reliable
- the result is **adaptive reliability-aware supervision**, rather than treating all temporal contexts equally

## Dataset

We evaluate on **EchoNet-Dynamic**, a widely used echocardiography benchmark containing apical four-chamber cardiac ultrasound videos with associated ejection fraction measurements.

### Binary Classification Setup

For downstream evaluation, we use a binary setup:

- **Normal:** `EF > 40`
- **Abnormal:** `EF <= 40`

### Class Imbalance

The dataset is meaningfully imbalanced:

- **~77.6% Normal**
- **~22.4% Abnormal**

Because of this imbalance, **raw accuracy alone is not sufficient** to characterize model quality. A model can appear strong while underperforming on the clinically important minority class.

We therefore focus on metrics that better reflect class-sensitive behavior:

- **Balanced Accuracy** accounts for performance across both classes instead of being dominated by the majority class.
- **Macro F1** treats each class more symmetrically and highlights uneven decision quality.
- **Abnormal Recall** is especially important in this setup because missing abnormal cardiac function is more consequential than simply optimizing aggregate accuracy.

## Experimental Phases

### Phase 1 — SSL Exploration and Reproduction

This phase focused on understanding and implementing a range of foundational SSL methods:

- DINO
- SwAV
- VideoMAE
- SimSiam
- DUNE
- SIGMA

The goal here was not only reproduction, but also understanding baseline SSL behavior under medical video constraints. This phase established implementation familiarity, optimization behavior, and failure modes relevant to ultrasound data.

### Phase 2 — DISCOVR Reproduction and Cross-Modal Understanding

This phase centered on:

- exploring the DISCOVR architecture
- understanding cross-modal learning for echocardiography
- integrating image SSL supervision into video SSL pipelines
- reproducing DISCOVR-style training behavior

This stage served as the bridge between generic SSL experimentation and a specifically echocardiography-aware representation learning setup.

### Phase 3 — Adaptive Multi-Teacher MT-DISCOVR

The final phase introduced the proposed framework:

- adaptive multi-teacher supervision
- barycentric fusion of teacher distributions
- motion-aware reliability weighting
- multi-teacher probabilistic distillation for video representation learning

This repository primarily documents the outcomes of this exploratory phase.

## Results

### Table 1 — Full Finetuning Results

| Model | Top-1 Acc | Balanced Acc | Macro F1 |
| --- | ---: | ---: | ---: |
| DISCOVR Baseline (800 ep pretrained) | 77.37 | 76.31 | 71.88 |
| MT-DISCOVR (50 ep MT pretrained) | 73.84 | 64.79 | 64.03 |

These numbers should be interpreted carefully. The comparison is **not perfectly fair**, since the DISCOVR baseline was trained with **substantially longer pretraining** and **larger-scale data exposure**. The purpose of this table is therefore not to claim superiority, but to situate MT-DISCOVR within a realistic performance envelope during early-stage exploration.

### Table 2 — Apple-to-Apple Linear Probe (50 vs 50)

| Model | Accuracy | Balanced Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| DISCOVR Baseline (50 ep pretrained) | 70.24 | 57.22 | 57.20 |
| MT-DISCOVR (50 ep pretrained) | 65.78 | 57.34 | 56.08 |

This comparison is closer to an **apple-to-apple representation evaluation**, since both models are compared at 50 pretraining epochs. In this context:

- **linear probing isolates representation quality** by evaluating frozen features
- **full finetuning measures adaptation capability** under supervised downstream optimization

The results suggest that MT-DISCOVR is an interesting research direction, but still early in its optimization cycle relative to the stronger and longer-trained DISCOVR baseline.

## Ablation Discussion

Although this repository is not positioned as a fully exhaustive ablation study, several observations motivate the current design:

- **Single-teacher supervision is likely too restrictive** for a domain where image semantics can be incomplete or unstable across views.
- **Barycentric fusion is preferable to naive averaging** when agreement between teachers should be rewarded and disagreement should be damped.
- **Motion-aware reweighting is especially relevant in echocardiography**, where rapid heart motion can make image-derived supervision less trustworthy for a video student.
- **Representation quality and full downstream adaptation may diverge**, which is why both linear probing and finetuning are reported.
- **Optimization maturity matters**: the current MT-DISCOVR results should be understood in the context of shorter pretraining and exploratory hyperparameter settings.

## Installation

The codebase is built around PyTorch-based self-supervised video learning and was developed for GPU training.

### Environment

- Python `3.10`
- PyTorch with CUDA support
- FFmpeg / TorchCodec where required by the data pipeline

### Setup

```bash
git clone https://github.com/prajithJ/DL_Grp_49.git
cd DL_Grp_49

conda create -n mt-discovr python=3.10 -y
conda activate mt-discovr

pip install -r requirements.txt
```

If your setup follows the existing cluster scripts, ensure that CUDA, distributed launch configuration, and codec dependencies are consistent with your execution environment.

## Training Commands

The main pretraining entrypoint is:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 12000 scripts/run_mae_pretraining.py \
  --data_path /path/to/videos \
  --data_path_csv /path/to/train.csv \
  --data_path_val /path/to/val.csv \
  --data_path_test /path/to/test.csv \
  --run_name MT_DISCOVR_PRETRAIN \
  --model pretrain_videomae_base_patch16_224 \
  --input_size 112 \
  --num_frames 64 \
  --batch_size 16 \
  --epochs 50 \
  --opt adamw \
  --opt_betas 0.9 0.95 \
  --warmup_epochs 40 \
  --decoder_depth 4 \
  --loss_func SWAV \
  --target_type mlp \
  --tokenizer_type default \
  --mask_type multi_local \
  --mask_ratio 0.9 \
  --global_mask_ratio 0.9 \
  --local_mask_ratio 0.9 \
  --num_local_views 4 \
  --num_prototypes 3000 \
  --sinkhorn_eps 0.05 \
  --sinkhorn_iterations 10 \
  --dino_out_dim 16384 \
  --use_combined_dino_swav \
  --use_video_dino \
  --use_multi_teacher \
  --teacher2_type simsiam \
  --output_dir /path/to/output
```

This configuration reflects the core MT-DISCOVR idea: a VideoMAE student trained with cross-modal distillation, clustering-based SSL, and adaptive multi-teacher supervision.

## Finetuning Commands

For full downstream finetuning on EchoNet-Dynamic classification:

```bash
python scripts/run_class_finetuning.py \
  --data_path /path/to/videos \
  --data_path_csv /path/to/train.csv \
  --data_path_val /path/to/val.csv \
  --data_path_test /path/to/test.csv \
  --model vit_base_patch16_224 \
  --num_frames 64 \
  --input_size 112 \
  --batch_size 16 \
  --epochs 50 \
  --nb_classes 2 \
  --finetune /path/to/pretrained_checkpoint.pth \
  --output_dir /path/to/finetune_output
```

Full finetuning measures how well the pretrained representation adapts under task-specific supervision.

## Linear Probing Commands

For frozen-backbone linear evaluation:

```bash
python scripts/run_class_finetuning.py \
  --data_path /path/to/videos \
  --data_path_csv /path/to/train.csv \
  --data_path_val /path/to/val.csv \
  --data_path_test /path/to/test.csv \
  --model vit_base_patch16_224 \
  --num_frames 64 \
  --input_size 112 \
  --batch_size 16 \
  --epochs 50 \
  --nb_classes 2 \
  --finetune /path/to/pretrained_checkpoint.pth \
  --linear_probe \
  --output_dir /path/to/linear_probe_output
```

Linear probing is the cleaner representation test because the pretrained encoder is not fully adapted during downstream learning.

## Future Work

The current repository captures an early-stage but promising direction. Natural next steps include:

- **Wasserstein barycenter fusion** instead of geometric-mean fusion
- **optimal transport teacher alignment** across heterogeneous semantic spaces
- **confidence-aware teacher weighting** rather than fixed symmetric fusion
- **prototype-level semantic alignment** beyond frame-wise distillation targets
- **multi-dataset large-scale pretraining** for stronger and more stable initialization
- **temporal attention reliability modeling** as a learned alternative to hand-designed motion weighting

## Citation

If this repository is useful in your work, please cite the underlying ideas that inspired it, especially DISCOVR and the major SSL foundations it builds upon.

```bibtex
@misc{mtdiscovr2026,
  title={Adaptive Multi-Teacher Cross-Modal Self-Supervised Learning for Echocardiography Video Representation Learning},
  author={Prajith J and collaborators},
  year={2026},
  note={Exploratory research repository}
}

@article{discovr2025,
  title={Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation},
  author={DISCOVR Authors},
  journal={NeurIPS},
  year={2025}
}
```

## Acknowledgements

This repository builds directly on ideas developed in the self-supervised learning community and in medical video representation learning. We specifically acknowledge:

- **DISCOVR** for cross-modal echocardiography SSL inspiration
- **VideoMAE** for masked video representation learning foundations
- **DINO** for self-distillation and teacher-student representation learning
- **SwAV** for online clustering and assignment-based learning
- **SimSiam** for non-contrastive image representation learning

MT-DISCOVR should be viewed as a research exploration at the intersection of these directions, adapted to the constraints and opportunities of cardiac ultrasound video learning. 🚑

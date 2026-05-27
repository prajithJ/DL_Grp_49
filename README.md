# Adaptive Multi Teacher Cross Modal SSL For ECG Videos

This repository is the working `TT/DISCOVR` implementation used for our echocardiography self-supervised learning experiments. The original upstream README described the authors' paper release at a high level. This version documents the code that is actually in this tree, the architectural additions we are using, and the current training and inference entrypoints.

## What This Repository Implements

At its core, this codebase keeps the DISCOVR idea of learning video representations from unlabeled echo clips, but the implementation in this folder has been extended into a more configurable pretraining framework around a VideoMAE-style backbone.

The current training stack supports:
- Video tokenization with either the default patch embedder or a sparse-tube tokenizer (`tokenizer_type=default|sparse_tube`)
- Global and local masking strategies, including `multi_local` masking for local-view distillation
- Joint SWAV-style clustering and DINO-style self-distillation (`--use_combined_dino_swav`)
- An image-teacher branch and an optional video-teacher branch (`--use_video_dino`)
- Optional multi-teacher distillation using SimSiam or RAD-DINO as a second teacher
- Optional turbo reconstruction that reconstructs only a selected subset of masked tokens
- A dedicated inference helper that loads a pretrained checkpoint and runs the `video_teacher` encoder when present

## Architecture Overview

The implementation is centered around `models/modeling_pretrain.py`, where the main pretraining model combines several pieces:

1. Video encoder backbone
- The main encoder is a VideoMAE-style vision transformer.
- The standard path uses patch-based video tokenization.
- An alternate sparse-tube tokenizer is also implemented through `models/TubeViT`, allowing tube-shaped token extraction instead of uniform patches.

2. Image teacher / student path
- When `--use_combined_dino_swav` is enabled, the code creates an image student and a momentum-updated image teacher.
- The image path runs DINO-style distillation on frame-level views.
- Local crops are supported through the DINO crop path and local positional embeddings.

3. Video teacher / student path
- When `--use_video_dino` is enabled, the main video encoder is paired with a momentum-updated `video_teacher`.
- Local video views are used to compute a video-level DINO objective.
- The inference helper in `run_discovr_encoder.py` explicitly prefers this `video_teacher` branch for representation extraction.

4. SWAV clustering path
- The pretraining model can compute prototype assignments with Sinkhorn normalization.
- Prototype count, Sinkhorn epsilon, and Sinkhorn iterations are configurable.
- In the combined regime, teacher image features are reused to guide the clustering branch.

5. Multi-teacher fusion
- The code supports an optional second teacher (`--use_multi_teacher`).
- The second teacher can be SimSiam or RAD-DINO.
- Teacher predictions can be fused with weighted-logit or geometric-mean style fusion.

6. Turbo reconstruction
- When `--use_turbo_training` is enabled, the model reconstructs only a selected subset of the masked tokens.
- This reduces reconstruction load while keeping a reconstruction signal in the objective.

## Key Architecture Changes From The Original README

The previous README mostly described the paper conceptually. The code in this repo now exposes several concrete implementation changes that matter for experiments:

- The training code is no longer just a single dual-branch description; it is a configurable pretraining framework with multiple loss and teacher combinations.
- The model supports both standard patch tokenization and sparse-tube tokenization.
- Masking is extended beyond simple tube masking to include `multi_local`, with separate global and local masks.
- DINO is used both for image-view distillation and, optionally, for video-view distillation.
- SWAV and DINO can be trained jointly in the same forward pass.
- A second teacher can be attached for adaptive multi-teacher distillation.
- Turbo training adds selective masked-token reconstruction on top of the self-distillation pipeline.
- The checkpoint inference path is wired to use the learned video teacher when available.

## Main Training Configuration

The default pretraining entrypoint is:

```bash
scripts/run_mae_pretraining.py
```

The arguments exposed there include:
- `--tokenizer_type default|sparse_tube`
- `--mask_type random|tube|parts|tube_fgbg|multi_local`
- `--loss_func L2|SWAV`
- `--use_combined_dino_swav`
- `--use_video_dino`
- `--use_multi_teacher`
- `--teacher2_type simsiam|rad_dino`
- `--use_turbo_training`
- `--num_local_views`
- `--num_frames`
- `--num_prototypes`
- `--sinkhorn_eps`
- `--sinkhorn_iterations`

The training loop in `engine/engine_for_pretraining.py` combines the enabled objectives and logs the individual components, including:
- SWAV loss
- image DINO loss
- video DINO loss
- turbo reconstruction loss

## Reference Pretraining Recipe

`config/run_pretraining.sh` contains the working template used for pretraining. The current recipe is configured around:

```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port 12000 scripts/run_mae_pretraining.py \
    --mask_type multi_local \
    --loss_func SWAV \
    --model pretrain_videomae_base_patch16_224 \
    --input_size 112 \
    --decoder_depth 4 \
    --batch_size 16 \
    --num_frames 64 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 40 \
    --epochs 400 \
    --num_prototypes 3000 \
    --sinkhorn_eps 0.05 \
    --sinkhorn_iterations 10 \
    --tokenizer_type default \
    --use_torchcodec \
    --dino_out_dim 16384 \
    --use_combined_dino_swav \
    --use_video_dino \
    --local_mask_ratio 0.9 \
    --global_mask_ratio 0.9 \
    --num_local_views 4
```

You still need to fill in the dataset CSV paths, video root path, output directory, and `PYTHONPATH` in that shell script before launching a run.

## Inference / Feature Extraction

The repository includes:

```bash
run_discovr_encoder.py
```

This helper script:
- builds the DISCOVR pretraining model with the expected configuration
- loads a checkpoint
- strips `module.` / `backbone.` prefixes when needed
- selects `video_teacher` if that branch exists
- runs a dummy tensor through the encoder and prints the CLS-token shape

Example usage:

```bash
python run_discovr_encoder.py \
    --checkpoint /path/to/checkpoint-799.pth \
    --num_frames 64 \
    --batch_size 2 \
    --device cuda
```

To use this for real data, replace the dummy tensor path with your own preprocessed echo clips and save the extracted embeddings downstream.

## Repository Layout

```text
TT/DISCOVR/
├── config/                  # Shell templates for training configuration
├── data/                    # Video dataset loaders and preprocessing
├── docs/                    # Figures and supporting documentation assets
├── engine/                  # Pretraining and finetuning loops
├── models/                  # VideoMAE backbone, pretraining model, TubeViT tokenizer support
├── output/                  # Experiment outputs and checkpoints
├── scripts/                 # Main training and finetuning entrypoints
├── utils/                   # Utilities for optimization, masking, logging, and visualization
├── run_discovr_encoder.py   # Checkpoint loading and encoder inference helper
├── requirements.txt
└── README.md
```

## Environment

The repo expects a Python environment compatible with the training scripts and the current cluster setup. The existing shell template assumes:
- Python 3.10
- PyTorch with CUDA 11.8
- distributed training through `torch.distributed.launch`
- FFmpeg / TorchCodec when `--use_torchcodec` is enabled

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

If you are recreating the original environment, also make sure the cluster modules and CUDA version match the settings in `config/run_pretraining.sh`.

## Notes For Our Team

When updating or comparing experiments in this repo, treat `models/modeling_pretrain.py`, `scripts/run_mae_pretraining.py`, and `engine/engine_for_pretraining.py` as the ground truth for behavior. The original upstream README does not describe the current feature set in this working tree.

## Upstream Credit

This repository is derived from the DISCOVR codebase and still builds on ideas and components from the original work, along with dependencies and design patterns taken from projects such as VideoMAE, DINO, SIGMA, and TubeViT.

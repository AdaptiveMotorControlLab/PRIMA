# PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation


This is the official implementation of the approach described in the preprint:

PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation
Xiaohang Yu, Ti Wang, Mackenzie Weygandt Mathis

![PRIMA teaser](images/teaser.png)


<!-- This repository provides:
- animal detection + 3D pose/shape inference demo
- test-time adaptation (TTA) demo with SuperAnimal 2D keypoints
- training pipeline (Stage 1 / Stage 2)
- evaluation on configured datasets -->

---


## 🚀 TL;DR
PRIMA creates a 3D quadruped mesh from a single 2D image. It leverages BioCLIP-based biological priors for robust cross-species shape understanding, then applies test-time adaptation with 2D reprojection and auxiliary keypoint guidance to refine SMAL pose and shape predictions. It further uses this adaptation pipeline to build Quadruped3D, a large-scale pseudo-3D dataset with diverse species and poses, achieving state-of-the-art results on Animal3D, CtrlAni3D, Quadruped80K, and Animal Kingdom.

## Installation

### Environment Setup

<!-- > Recommended: Python 3.10 + CUDA-enabled PyTorch. -->

```bash
git clone <your_repo_url>
cd PRIMA

conda create -n prima python=3.10 -y
conda activate prima

# PyTorch (example for CUDA 12.1; change if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install numpy opencv-python tqdm yacs hydra-core omegaconf pyrootutils pytorch-lightning einops trimesh pyrender

# Vision/model dependencies used by PRIMA
pip install timm transformers open-clip-torch

# Detectron2 (pick the wheel matching your torch/cuda)
# See: https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md

# Optional (for demo_tta.py)
pip install deeplabcut
```

---

## Demo

### Checkpoints and data

Place model checkpoints and config under `data/`, download stage-1 checkpoint from [here](https://drive.google.com/drive/folders/1pwIpYwP3aJ6W2M3-WhEvcFjW38-4j405?usp=drive_link), and stage-3 checkpoint from [here](https://drive.google.com/drive/folders/1DO6idTCORL5G6PLjikRaIjCXmo_-Ut31?usp=drive_link)

Download SMAL model from [here](https://drive.google.com/drive/folders/1O1tWYimVMA7hEbnwuPyiDWh90tUGoTPB?usp=drive_link), and place under `data/`

Download the pretrained backbone from [here](https://drive.google.com/file/d/1jOJXJVPXnWX7W7vqYVt0joJZr4C8x-Yo/view?usp=drive_link)

---

### Demo (without tta)

Run animal detection + PRIMA 3D pose/shape inference:

```bash
python demo.py \
  --checkpoint data/PRIMAS1/checkpoints/s1ckpt.ckpt \
  --img_folder demo_data/ \
  --out_folder demo_out/
```

Optional:
- `--side_view` render side view
- `--save_mesh` export `.obj`

---

### Demo (with tta)

`demo_tta.py` pipeline, specify learning rate and numbers of iteration:

Example:

```bash
python demo_tta.py \
  --checkpoint data/PRIMAS1/checkpoints/s1ckpt.ckpt \
  --tta_lr 1e-6 \
  --tta_num_iters 30
```

Notes:
- `.obj` is exported only when `--save_mesh` is provided.
- 26-keypoint visualization is saved as `*_prima26_kpts.png`.

---


## Training and Evaluation 

### Dataset Setup

Download dataset from [Animal3D](https://xujiacong.github.io/Animal3D/), [CtrlAni3D](https://github.com/luoxue-star/AniMer?tab=readme-ov-file#training), and [Quadruped80K](https://zenodo.org/records/14016777). 

### Training 

Two-stage training script:

```bash
bash train.sh
```

<!-- This launches:
- Stage 1: `experiment=primaStage1`
- Stage 2: `experiment=primaStage2`

Main configs:
- `prima/configs_hydra/train.yaml`
- `prima/configs_hydra/experiment/primaStage1.yaml`
- `prima/configs_hydra/experiment/primaStage2.yaml` -->

Training outputs are written to `logs/train/runs/<exp_name>/`.


### Evaluation

```bash
python eval.py \
  --config data/PRIMAS1/.hydra/config.yaml \
  --checkpoint data/PRIMAS1/checkpoints/s1ckpt.ckpt \
```

Common values for `--dataset` are controlled by:
- `prima/configs_hydra/experiment/default_val.yaml`

---


## Acknowledgements

This release builds on several open-source projects, including:
- Detectron2
- PyTorch Lightning
- DeepLabCut SuperAnimal
- AniMer
- Transformer/backbone ecosystems (timm, Hugging Face Transformers, OpenCLIP)

---

## Citation

If you use this code in your research, please cite our PRIMA paper (update BibTeX here in your final camera-ready release).

```bibtex
@misc{yu_prima,
  title={PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation},
  author={Xiaohang Yu and Ti Wang and Mackenzie Weygandt Mathis},
  note={EPFL project page. Update publication year, venue, and links when available.}
}
```

---

## Contact

For issues, please open a GitHub issue in this repository.
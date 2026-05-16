---
title: PRIMA Demo
emoji: 🦮
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.10"
app_file: app.py
startup_duration_timeout: 60m
---

# PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation


This is the official implementation of the approach described in the preprint:

PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation \
Xiaohang Yu, Ti Wang, Mackenzie Weygandt Mathis

![PRIMA teaser](images/teaser.png)


---


## 🚀 TL;DR
PRIMA creates a 3D quadruped mesh from a single 2D image. It leverages BioCLIP-based biological priors for robust cross-species shape understanding, then applies test-time adaptation with 2D reprojection and auxiliary keypoint guidance to refine SMAL pose and shape predictions. 

It further can be used to build Quadruped3D, a large-scale pseudo-3D dataset with diverse species and poses. 

PRIMA achieves state-of-the-art results on Animal3D, CtrlAni3D, Quadruped2D, and Animal Kingdom datasets.

## Installation

### Install from PyPI

> Recommended: Python 3.10 and a CUDA-enabled PyTorch installation.

```bash
conda create -n prima python=3.10 -y
conda activate prima

# Install PyTorch matching your CUDA (example: CUDA 11.8)
pip install --index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.2.1" "torchvision==0.17.1" "torchaudio==2.2.1"

# Install chumpy and PyTorch3D
python -m pip install --no-build-isolation \
      "git+https://github.com/mattloper/chumpy.git"
python -m pip install --no-build-isolation \
      "git+https://github.com/facebookresearch/pytorch3d.git"

# Install PRIMA from PyPI
pip install prima-animal
```

`prima-animal` includes demo runtime dependencies used by `demo.py`, `demo_tta.py`, and `app.py` (including Detectron2 and DeepLabCut).

### Clean install from this repository

Use these when developing from a **git clone** (not the PyPI wheel). The shell scripts are **non-interactive** (pip uses `--no-input`; `GIT_TERMINAL_PROMPT=0` for git). Put Hugging Face credentials in your environment or git credential helper before pushing the Space.

**Local (fresh venv, LFS assets, Hub demo weights, smoke test)** — requires **Python 3.10+**
(Gradio 5.1+ / Space-provided Gradio 6.x and `app.py` type hints). On macOS without `python3.10` on your `PATH`, install
`brew install python@3.10` and set `PRIMA_PYTHON=/opt/homebrew/bin/python3.10`.

```bash
chmod +x scripts/clean_install_local.sh scripts/clean_redeploy_hf_space.sh scripts/deploy_hf_space.sh
PRIMA_PYTHON=/opt/homebrew/bin/python3.10 ./scripts/clean_install_local.sh
```

Options:

- `PRIMA_VENV=.venv ./scripts/clean_install_local.sh --skip-data` — skip the large `setup_demo_data` download if `data/` is already populated.
- `./scripts/clean_install_local.sh --wipe-data --force-data` — delete downloaded `data/` assets and redownload.
- `./scripts/clean_install_local.sh --no-editable` — only `requirements.txt` (no `pip install -e .`); use if editable install fails and you will install the training stack via conda as in the PyPI section above. You still need **Python 3.10+** for Gradio 5.1+. The smoke test sets `PYTHONPATH` to the repo root so `import prima` works without an editable install.
- **macOS:** the script omits the `deeplabcut` line from `pip install` because DeepLabCut’s pinned PyTables version often does not build on Apple Silicon. Use conda/mamba for DeepLabCut if you need SuperAnimal + TTA (`tta_num_iters` &gt; 0). **Linux** (including Hugging Face Space builds) uses the full `requirements.txt` including `deeplabcut`.

After `requirements.txt`, the script runs **`pip install --no-deps -e .`** so the `prima` package is registered without re-resolving `pyproject.toml` (which would pull **Detectron2** and **DeepLabCut** again and often fail on macOS). Full `pip install -e .` is still recommended from a **conda** environment per the PyPI section if you need every training extra matched exactly.

**Hugging Face Space (full redeploy from your working tree):**

Requires [Git LFS / Xet](https://huggingface.co/docs/hub/xet/using-xet-storage#git) tooling (`brew install git-lfs git-xet`, `git xet install`, `git lfs install`). Then:

```bash
./scripts/clean_redeploy_hf_space.sh
```

This is equivalent to `./scripts/deploy_hf_space.sh` and force-pushes a fresh snapshot to the Space.

---

## Demo

### Checkpoints and data

The demo scripts auto-download their default Stage 1 PRIMA assets from Hugging
Face when the checkpoint or matching Hydra config is missing. If you want to
pre-download all necessary checkpoints and data ahead of time, run:

```bash
python scripts/setup_demo_data.py --hf-repo-id MLAdaptiveIntelligence/PRIMA
```

Approximate default prefetch volume from Hugging Face is ~5.5 GB total
(`s1ckpt_inference.ckpt` ~3 GB + `amr_vitbb.pth` ~2.5 GB + SMAL files).
Expected time is roughly:
- 100 Mbps: ~7-10 minutes
- 300 Mbps: ~2-4 minutes
- 1 Gbps: ~1 minute

Existing files are reused by default; pass `--force` only if you need to redownload them. If you also need the Stage 3 pretrained model, add `--include-stage3`.

Expected files in that Hugging Face repo root:
- `my_smpl_00781_4_all.pkl`
- `my_smpl_data_00781_4_all.pkl`
- `walking_toy_symmetric_pose_prior_with_cov_35parts.pkl`
- `amr_vitbb.pth`
- `config_s1_HYDRA.yaml`
- `s1ckpt_inference.ckpt`

Optional Stage 3 prefetch expects:
- `config_s3_HYDRA.yaml`
- `s3ckpt.ckpt`

### Demo (without TTA)

Run animal detection + PRIMA 3D pose/shape inference:

```bash
bash demo.sh
```

Outputs are written to `demo_out/`. Edit `demo.sh` if you want to use a custom
checkpoint path.

---

### Demo (with TTA)

Run PRIMA inference with test-time adaptation:

```bash
bash demo_tta.sh
```

Outputs are written to `demo_out_tta/` (before/after TTA renders, keypoints, and
optional meshes). Edit `demo_tta.sh` if you want to change the checkpoint, TTA
learning rate, or number of iterations.

---

### Gradio demo

We also provide a simple Gradio-based web demo for interactive testing in the
browser:

```bash
python app.py \
  --checkpoint data/PRIMAS1/checkpoints/s1ckpt_inference.ckpt \
  --out_folder demo_out_tta_gradio/
```

This starts a local Gradio app (by default on http://127.0.0.1:7860), where
you can upload images and visualize PRIMA predictions and adaptation results.
The `s1ckpt_inference.ckpt` checkpoint is downloaded automatically if missing.

#### Hugging Face Space (maintainers)

Demo images under `demo_data/` and `images/teaser.png` are tracked with **Git LFS**
(see `.gitattributes`) so they can be pushed to a Hugging Face Space under the Hub’s
LFS / **Xet** bridge. Install tooling once:

```bash
brew install git-lfs git-xet
git xet install
git lfs install
```

Then from a clean checkout with LFS files present, redeploy the Space (same as `clean_redeploy_hf_space.sh`):

```bash
./scripts/deploy_hf_space.sh
# or
./scripts/clean_redeploy_hf_space.sh
```

The script rsyncs the working tree (not `git archive`) so image files are materialized
before `git add` turns them into LFS blobs.

---


## Training and Evaluation 

### Dataset Setup

Download datasets from [Animal3D](https://xujiacong.github.io/Animal3D/), [CtrlAni3D](https://github.com/luoxue-star/AniMer?tab=readme-ov-file#training), Quadruped2D, and [Animal Kingdom](https://drive.google.com/file/d/1dk2a0qB0fbVZ4X6eAgP6VJVXj0rxVfsJ/view?usp=drive_link). For Quadruped2D, download the images from [SuperAnimal-Quadruped80K](https://zenodo.org/records/14016777) and our processed annotations from [here](https://drive.google.com/drive/folders/1eBNboxVwl_eGPoC93zxf-U3hmE6e2f-f?usp=sharing). Put all the datasets under `datasets/`.

### Training 

Two-stage training script:

```bash
bash train.sh
```

Training outputs are written to `logs/train/runs/<exp_name>/`.


### Evaluation

```bash
python eval.py \
  --config data/PRIMAS1/.hydra/config.yaml \
  --checkpoint data/PRIMAS1/checkpoints/s1ckpt_inference.ckpt
```

Common values for `--dataset` are controlled by:
- `configs_hydra/experiment/default_val.yaml`

---


## Acknowledgements

This release builds on several open-source projects, including:
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [BioCLIP](https://github.com/Imageomics/BioCLIP)
- [AniMer](https://github.com/luoxue-star/AniMer)
- [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)

---

## Citation

If you use this code in your research, please cite our PRIMA paper.

```bibtex
@misc{yu_prima,
  title={PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation},
  author={Xiaohang Yu and Ti Wang and Mackenzie Weygandt Mathis},
}
```

---

## Contact

For issues, please open a GitHub issue in this repository.

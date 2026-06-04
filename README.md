# PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation


This is the official implementation of the approach described in the preprint:

PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation \
Xiaohang Yu, Ti Wang, Mackenzie Weygandt Mathis

![PRIMA teaser](images/teaser.png)


---


## TL;DR
PRIMA creates a 3D quadruped mesh from a single 2D image. It leverages BioCLIP-based biological priors for robust cross-species shape understanding, then applies test-time adaptation with 2D reprojection and auxiliary keypoint guidance to refine SMAL pose and shape predictions. 

It further can be used to build Quadruped3D, a large-scale pseudo-3D dataset with diverse species and poses. 

PRIMA achieves state-of-the-art results on Animal3D, CtrlAni3D, Quadruped2D, and Animal Kingdom datasets.

## Installation

PRIMA requires Python 3.10 or newer. A CUDA-enabled PyTorch installation is
recommended for local inference and training.

### Install from PyPI

Create a clean environment, install PyTorch for your CUDA version, then install
the package:

```bash
conda create -n prima python=3.10 -y
conda activate prima

# Example for CUDA 11.8. Adjust this command for your CUDA version.
pip install --index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.2.1" "torchvision==0.17.1" "torchaudio==2.2.1"

# Install chumpy and PyTorch3D
python -m pip install --no-build-isolation \
      "git+https://github.com/mattloper/chumpy.git"
python -m pip install --no-build-isolation \
      "git+https://github.com/facebookresearch/pytorch3d.git"

# Install PRIMA from PyPI-test (for now)
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple prima-animal==0.1.7

# Install PRIMA from PyPI
pip install prima-animal
```

`prima-animal` includes demo runtime dependencies used by `demo.py`, `demo_tta.py`, and `app.py` (including Detectron2 and DeepLabCut).

### Install from this repository

Use this path if you want to run the code from a fresh clone.

```bash
git clone https://github.com/AdaptiveMotorControlLab/PRIMA.git
cd PRIMA
```

The helper script below creates a fresh virtual environment, installs runtime
dependencies, pulls Git LFS assets if available, downloads the default demo
checkpoints/data, and verifies that the demo dependencies can be imported:


```bash
PRIMA_PYTHON=/path/to/python3.10 \
PRIMA_VENV=prima_env \
./scripts/clean_install_local.sh
source prima_env/bin/activate
```

Useful options:

- `--skip-data` skips the large demo data download if `data/` is already populated.
- `--wipe-data --force-data` removes downloaded demo assets and downloads them again.
- `--no-editable` installs dependencies without registering the repo as an editable package.

On macOS, install Python 3.10 if needed:

```bash
brew install python@3.10
PRIMA_PYTHON=/opt/homebrew/bin/python3.10 ./scripts/clean_install_local.sh
```

The installer uses `pip install --no-deps -e .` after installing
`requirements.txt`, so the local `prima` package is registered without
re-resolving the full dependency list.

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
- `s3ckpt_inference.ckpt`

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

`app.py` picks a **demo profile** automatically:

| | **Local** (`python app.py`) | **Hugging Face Space** |
|--|--|--|
| PRIMA device | GPU if available, else CPU | CPU only |
| Detector | Detectron2 X-101-FPN | DeepLabCut SuperAnimal detector |
| Default TTA iterations | 30 | 30 |
| Save `.obj` meshes | on | off |
| Preload checkpoint at startup | off | on |

Override for testing: `PRIMA_DEMO_MODE=local` or `PRIMA_DEMO_MODE=space`.

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
- [SAM3DB](https://github.com/facebookresearch/sam-3d-body)

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

# PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation


This is the official implementation of the approach described in the preprint:

PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation \
Xiaohang Yu, Ti Wang, Mackenzie Weygandt Mathis

![PRIMA teaser](images/teaser.png)


---


## 🚀 TL;DR
PRIMA creates a 3D quadruped mesh from a single 2D image. It leverages BioCLIP-based biological priors for robust cross-species shape understanding, then applies test-time adaptation with 2D reprojection and auxiliary keypoint guidance to refine SMAL pose and shape predictions. It further uses this adaptation pipeline to build Quadruped3D, a large-scale pseudo-3D dataset with diverse species and poses, achieving state-of-the-art results on Animal3D, CtrlAni3D, Quadruped2D, and Animal Kingdom datasets.

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
pip install prima-animal==0.1.7
```

`prima-animal` includes demo runtime dependencies used by `demo.py`, `demo_tta.py`, and `app.py` (including Detectron2 and DeepLabCut).

---

## Demo

### Checkpoints and data

Create a `data/` folder under the project root and download the required files into it:

```bash
mkdir -p data/smal
```

1. **SMAL model** -- download from [here](https://drive.google.com/drive/folders/1O1tWYimVMA7hEbnwuPyiDWh90tUGoTPB?usp=drive_link) and place the `.pkl` files under `data/smal/`
2. **Pretrained backbone** -- download from [here](https://drive.google.com/file/d/1jOJXJVPXnWX7W7vqYVt0joJZr4C8x-Yo/view?usp=drive_link) and place at `data/amr_vitbb.pth`
3. **Stage-1 checkpoint** -- download from [here](https://drive.google.com/drive/folders/1pwIpYwP3aJ6W2M3-WhEvcFjW38-4j405?usp=drive_link) and place under `data/PRIMAS1/`
4. **Stage-3 checkpoint** -- download from [here](https://drive.google.com/drive/folders/1DO6idTCORL5G6PLjikRaIjCXmo_-Ut31?usp=drive_link) and place under `data/PRIMAS3/`

After downloading, the expected layout is:

```
PRIMA/
└── data/
    ├── smal/
    │   ├── my_smpl_00781_4_all.pkl
    │   ├── my_smpl_data_00781_4_all.pkl
    │   └── walking_toy_symmetric_pose_prior_with_cov_35parts.pkl
    ├── amr_vitbb.pth
    ├── PRIMAS1/
    │   ├── .hydra/
    │   │   └── config.yaml
    │   └── checkpoints/
    │       └── s1ckpt.ckpt
    └── PRIMAS3/
        ├── .hydra/
        │   └── config.yaml
        └── checkpoints/
            └── s3ckpt.ckpt
```

---

### Demo (without TTA)

Run animal detection + PRIMA 3D pose/shape inference:

```bash
python demo.py \
  --checkpoint data/PRIMAS1/checkpoints/s1ckpt.ckpt \
  --img_folder demo_data/ \
  --out_folder demo_out/
```

Outputs are written to `demo_out/`.

---

### Demo (with TTA)

`demo_tta.py` pipeline: specify learning rate and number of iterations:

Example:

```bash
python demo_tta.py \
  --checkpoint data/PRIMAS1/checkpoints/s1ckpt.ckpt \
  --img_folder demo_data/ \
  --out_folder demo_out_tta/ \
  --tta_lr 1e-6 \
  --tta_num_iters 30
```

Outputs are written to `demo_out_tta/` (before/after TTA renders, keypoints, and optional meshes).

---

### Gradio demo

We also provide a simple Gradio-based web demo for interactive testing in the
browser:

```bash
python app.py \
  --checkpoint data/PRIMAS1/checkpoints/s1ckpt.ckpt \
  --out_folder demo_out_tta_gradio/
```

This starts a local Gradio app (by default on http://127.0.0.1:7860), where
you can upload images and visualize PRIMA predictions and adaptation results.

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
  --checkpoint data/PRIMAS1/checkpoints/s1ckpt.ckpt
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
  note={EPFL project page. Update publication year, venue, and links when available.}
}
```

---

## Contact

For issues, please open a GitHub issue in this repository.

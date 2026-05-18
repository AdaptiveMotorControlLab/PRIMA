
# Empty checkpoint uses the default PRIMA Stage 1 inference checkpoint:
#   data/PRIMAS1/checkpoints/s1ckpt_inference.ckpt
#
# This standard path is auto-downloaded from the PRIMA Hugging Face repo if missing.
# To use another local checkpoint instead, update this path.
# For example: checkpoint='data/PRIMAS3/checkpoints/s3ckpt.ckpt'
checkpoint='data/PRIMAS1/checkpoints/s1ckpt_inference.ckpt'

python3 demo_tta.py \
  --checkpoint "${checkpoint}" \
  --img_folder demo_data/ \
  --out_folder demo_out_tta/ \
  --tta_lr 1e-6 \
  --tta_num_iters 30

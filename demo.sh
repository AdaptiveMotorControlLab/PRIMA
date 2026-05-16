# Default PRIMA Stage 1 inference checkpoint:
#   data/PRIMAS1/checkpoints/s1ckpt_inference.ckpt
#
# If this local file is missing, it will be downloaded from the PRIMA Hugging Face repo.
# To use another local checkpoint instead, update this path.
# For example: checkpoint='data/PRIMAS3/checkpoints/s3ckpt.ckpt'
checkpoint='data/PRIMAS1/checkpoints/s1ckpt_inference.ckpt'

python demo.py \
  --checkpoint "${checkpoint}" \
  --img_folder demo_data/ \
  --out_folder demo_out/

# Default PRIMA inference checkpoint:
#   data/PRIMA/checkpoints/best_inference.ckpt
#
# If this local file is missing, it will be downloaded from the PRIMA Hugging Face repo.
# To use another local checkpoint instead, update this path.
checkpoint='data/PRIMA/checkpoints/best_inference.ckpt'

python demo.py \
  --checkpoint "${checkpoint}" \
  --img_folder demo_data/ \
  --out_folder demo_out/

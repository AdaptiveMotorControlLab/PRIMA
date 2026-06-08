# Default PRIMA Stage 1 inference checkpoint:
#   data/PRIMAS1/checkpoints/s1ckpt_inference.ckpt
#
# If this local file is missing, it will be downloaded from the PRIMA Hugging Face repo.
checkpoint='data/PRIMAS1/checkpoints/s1ckpt_inference.ckpt'

# Update this to your video path before running.
video_path='demo_data/hati.mp4'

python3 demo_video.py \
  --checkpoint "${checkpoint}" \
  --video_path "${video_path}" \
  --out_video demo_video_out.mp4 \
  --out_folder demo_video_out/ \
  --rotate auto

# Default PRIMA inference checkpoint:
#   data/PRIMA/checkpoints/best_inference.ckpt
#
# If this local file is missing, it will be downloaded from the PRIMA Hugging Face repo.
checkpoint='data/PRIMA/checkpoints/best_inference.ckpt'

# Update this to your video path before running.
video_path='demo_data/hati.mp4'

python3 demo_video.py \
  --checkpoint "${checkpoint}" \
  --video_path "${video_path}" \
  --out_video demo_video_out.mp4 \
  --rotate auto \
  --full_frame

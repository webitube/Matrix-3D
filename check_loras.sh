#!/usr/bin/env bash
cd /mnt/g/Dev/Matrix-3D
echo "=== LoRA files at corrected paths ==="
for f in \
  checkpoints/flux_lora/checkpoints/text2panoimage_lora.safetensors \
  checkpoints/Wan-AI/wan_lora/checkpoints/pano_video_gen_720p.bin \
  checkpoints/Wan-AI/wan_lora/checkpoints/pano_video_gen_480p.ckpt \
  checkpoints/Wan-AI/wan_lora/checkpoints/pano_video_gen_720p_5b.safetensors
do
  if [ -s "$f" ]; then
    sz=$(du -h "$f" | cut -f1)
    echo "OK   $f ($sz)"
  else
    echo "MISS $f"
  fi
done
echo
echo "=== 5B base model present? (Wan2.2-TI2V-5B) ==="
if [ -d checkpoints/Wan-AI/Wan2.2-TI2V-5B ]; then
  du -sh checkpoints/Wan-AI/Wan2.2-TI2V-5B
else
  echo "NOT downloaded yet (will be fetched at runtime, ~10-20GB)"
fi

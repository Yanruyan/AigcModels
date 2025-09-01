#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python text2image.py \
  --steps 30 \
  --guidance_scale 7.5 \
  --width 1024 \
  --height 1024 \
  --seed 1024 \
  --denoising_end 0.8 \
  --gpu 0 \
  --scheduler dpmpp_2m \
  --output ../gen_text_images/1.png
#!/usr/bin/env python3
"""
Minimal SDXL text-to-image script (single image, single GPU).

Default model: stabilityai/stable-diffusion-xl-base-1.0
Optionally use the SDXL refiner: stabilityai/stable-diffusion-xl-refiner-1.0

Usage examples:
  CUDA_VISIBLE_DEVICES=0 python generate_image.py \
    --prompt "a cozy cabin in the snowy mountains, golden hour" \
    --output out.png

  # Use explicit device and seed
  python generate_image.py --prompt "studio portrait of a cat, 85mm" \
    --seed 123 --device cuda:0 --output cat.png

Test CLI without downloading models:
  python generate_image.py --help
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLRefinerPipeline,
    DPMSolverMultistepScheduler,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a single image from a text prompt using SDXL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality, worst quality, bad anatomy, bad hands, watermark, text",
        help="Negative text prompt",
    )
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--width", type=int, default=1024, help="Image width (multiple of 8)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (multiple of 8)")
    parser.add_argument("--steps", type=int, default=30, help="Denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base model repo or local path",
    )
    parser.add_argument(
        "--use-refiner",
        action="store_true",
        help="Use the SDXL refiner (adds a few seconds but can improve details)",
    )
    parser.add_argument(
        "--refiner-model",
        type=str,
        default="stabilityai/stable-diffusion-xl-refiner-1.0",
        help="Refiner model repo or local path (used only with --use-refiner)",
    )

    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device to use")
    parser.add_argument(
        "--no-xformers",
        action="store_true",
        help="Disable xFormers memory-efficient attention if installed",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable CPU offload (slower but reduces GPU memory)",
    )
    return parser.parse_args()


def maybe_enable_memory_efficient_attention(pipe, disable: bool) -> None:
    if disable:
        return
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        # xformers not available; fall back to attention slicing
        pipe.enable_attention_slicing()


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available() and not args.device.startswith("cpu"):
        raise RuntimeError("CUDA is not available. Specify --device cpu or install CUDA PyTorch.")

    device = torch.device(args.device)

    generator: Optional[torch.Generator] = None
    if args.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)

    # Load base pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        variant="fp16" if device.type == "cuda" else None,
        use_safetensors=True,
    )

    # Replace scheduler with a performant default
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device)
    else:
        pipe.to(device)

    maybe_enable_memory_efficient_attention(pipe, disable=args.no_xformers)

    if args.use_refiner:
        refiner = StableDiffusionXLRefinerPipeline.from_pretrained(
            args.refiner_model,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            variant="fp16" if device.type == "cuda" else None,
            use_safetensors=True,
        )
        if args.cpu_offload:
            refiner.enable_model_cpu_offload(device)
        else:
            refiner.to(device)
        maybe_enable_memory_efficient_attention(refiner, disable=args.no_xformers)
    else:
        refiner = None

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Two-stage SDXL workflow when refiner is used
    if refiner is not None:
        denoising_end = 0.8
        base_result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            denoising_end=denoising_end,
            generator=generator,
            output_type="latent",
        )

        image = refiner(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=base_result.images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=max(10, args.steps // 3),
            denoising_start=denoising_end,
            generator=generator,
        ).images[0]
    else:
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            generator=generator,
        ).images[0]

    image.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()


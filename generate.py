#!/usr/bin/env python3

import os
import argparse
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from diffusers import StableDiffusionXLPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal SDXL text-to-image generator (single-GPU, single image)"
    )
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt (undesired attributes)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width (multiple of 8)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height (multiple of 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (0-based)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Directory to save the generated image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Model repo or local path",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="",
        help="Optional scheduler alias: ddim, dpmpp_2m, dpmpp_2m_sde, euler_a, euler",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for slight speedup (PyTorch >= 2.0)",
    )
    return parser.parse_args()


def ensure_cuda_device(gpu_index: int) -> torch.device:
    # Limit visibility to the selected GPU for safety
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure NVIDIA drivers and CUDA-enabled PyTorch are installed.")
    return torch.device("cuda:0")


def maybe_set_scheduler(pipe: StableDiffusionXLPipeline, alias: str) -> None:
    if not alias:
        return
    alias = alias.lower()
    try:
        from diffusers import (
            DDIMScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            DPMSolverMultistepScheduler,
        )
        if alias == "ddim":
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif alias == "euler_a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif alias == "euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        elif alias in {"dpmpp_2m", "dpmpp_2m_sde"}:
            # Use Karras for better quality at fewer steps
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, algorithm_type="sde-dpmsolver++" if alias.endswith("sde") else "dpmsolver++", use_karras_sigmas=True
            )
        else:
            print(f"[warn] Unknown scheduler alias: {alias}. Using default.")
    except Exception as exc:
        print(f"[warn] Could not set scheduler '{alias}': {exc}")


def load_pipeline(model_id: str, device: torch.device, do_compile: bool) -> StableDiffusionXLPipeline:
    print(f"Loading model: {model_id}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    # Enable memory optimizations
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Enabled xFormers memory-efficient attention.")
    except Exception as exc:
        print(f"[warn] xFormers not available or failed to enable: {exc}")
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload() if torch.cuda.get_device_properties(device).total_memory < 16 * 1024**3 else pipe.to(device)

    # Optional: compile UNet for a small speedup on PyTorch >= 2.0
    if do_compile and hasattr(torch, "compile"):
        try:
            pipe.unet = torch.compile(pipe.unet, mode="max-autotune")
            print("Compiled UNet with torch.compile.")
        except Exception as exc:
            print(f"[warn] torch.compile failed: {exc}")
    return pipe


def generate_image(
        pipe: StableDiffusionXLPipeline,
        prompt: str,
        nprompt: str,
        steps: int,
        guidance: float,
        width: int,
        height: int,
        seed: Optional[int],
        output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = pipe.device

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    start_time = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        result = pipe(
            prompt=prompt,
            negative_prompt=nprompt or None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
    elapsed = time.time() - start_time
    print(f"Image generated in {elapsed:.2f}s")

    image: Image.Image = result.images[0]
    filename = "sdxl.png"
    path = output_dir / filename
    image.save(path)
    return path


def main():
    args = parse_args()
    device = ensure_cuda_device(args.gpu)

    pipe = load_pipeline(args.model, device, args.compile)
    maybe_set_scheduler(pipe, args.scheduler)

    out_path = generate_image(
        pipe=pipe,
        prompt=args.prompt,
        nprompt=args.negative_prompt,
        steps=args.num_inference_steps,
        guidance=args.guidance_scale,
        width=args.width,
        height=args.height,
        seed=args.seed,
        output_dir=Path(args.output),
    )

    print("Saved:")
    print(str(out_path))


if __name__ == "__main__":
    main()
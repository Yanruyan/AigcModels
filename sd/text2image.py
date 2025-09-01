#!/usr/bin/env python3

import os
import argparse
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SDXL text-to-image generator"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="CFG guidance scale",
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
        default=1024,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--denoising_end",
        type=float,
        default=0.8,
        help="denoising_end",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../gen_text_images",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Model repo or local path",
    )
    parser.add_argument(
        "--refiner_model",
        type=str,
        default="stabilityai/stable-diffusion-xl-refiner-1.0",
        help="Refiner model repo or local path",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="dpmpp_2m",
        help="Optional scheduler alias: ddim, dpmpp_2m, dpmpp_2m_sde, euler_a, euler",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for slight speedup (PyTorch >= 2.0)",
    )
    return parser.parse_args()


def set_cuda_device(gpu_index: int) -> torch.device:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure NVIDIA drivers and CUDA-enabled PyTorch are installed.")
    return torch.device(f"cuda:{gpu_index}")


def set_scheduler(pipe: StableDiffusionXLPipeline, alias: str) -> None:
    alias = alias.lower()
    try:
        if alias == "ddim":
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif alias == "euler_a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif alias == "euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        elif alias in {"dpmpp_2m", "dpmpp_2m_sde"}:
            algo_type = "sde-dpmsolver++" if alias.endswith("sde") else "dpmsolver++"
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, algorithm_type=algo_type, use_karras_sigmas=True
            )
        else:
            print(f"[warn] Unknown scheduler alias: {alias}. Using default.")
    except Exception as exc:
        print(f"[warn] Could not set scheduler '{alias}': {exc}")


def load_pipeline(model_id: str, device: torch.device, do_compile: bool) -> StableDiffusionXLPipeline:
    print(f"Loading sdxl model: {model_id}")
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


def load_refiner(model_id: str, device: torch.device) -> StableDiffusionXLImg2ImgPipeline:
    # load model
    print(f"Loading refiner model: {model_id}")
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        variant="fp16" if device.type == "cuda" else None,
        use_safetensors=True,
        safety_checker=None  # 禁用安全检查加速
    )
    # load to cpu or gpu
    refiner.enable_model_cpu_offload() if torch.cuda.get_device_properties(
        device).total_memory < 16 * 1024 ** 3 else refiner.to(device)
    # enable memory optimizations
    try:
        refiner.enable_xformers_memory_efficient_attention()
    except Exception:
        # xformers not available; fall back to attention slicing
        refiner.enable_attention_slicing()
    return refiner


def generate_images(
        pipe: StableDiffusionXLPipeline,
        prompt: str,
        nprompt: str,
        steps: int,
        guidance: float,
        width: int,
        height: int,
        output_path: str,
        refiner: StableDiffusionXLImg2ImgPipeline,
        denoising_end: float,
        generator: torch.Generator,
) -> None:
    print("Generating with base model...")
    base_image = pipe(
        prompt=prompt,
        negative_prompt=nprompt,
        width=width,
        height=height,
        guidance_scale=guidance,
        num_inference_steps=steps,
        denoising_end=denoising_end,
        generator=generator,
        output_type="latent",
    ).images[0]
    image = refiner(
        prompt=prompt,
        image=base_image,
        generator=generator,
        num_inference_steps=max(10, int(steps * (1.0 - denoising_end))),
        strength=0.3,
        guidance_scale=guidance,
    ).images[0]
    image.save(output_path)
    print(f"Saved: {output_path}")


def main():
    args = parse_args()
    device = set_cuda_device(args.gpu)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # load models
    pipe = load_pipeline(args.model, device, args.compile)
    set_scheduler(pipe, args.scheduler)
    refiner = load_refiner(args.refiner_model, device)

    # prompt
    SD_PROMPT = "Professional fashion photography of a beautiful young female model wearing elegant summer dress, standing in a sunlit garden with blooming flowers, natural daylight, soft focus, fresh and natural style, clean background, full body shot, perfect composition, 8k resolution, ultra detailed, sharp focus, masterpiece quality, vogue style fashion photography, natural makeup, wind blowing gently through hair"
    SD_N_PROMPT = "blurry, low quality, bad anatomy, distorted face, deformed hands, extra limbs, poorly drawn, watermark, text, signature, oversaturated, dark, gloomy, unnatural skin tone, plastic look, heavy makeup, studio backdrop, artificial lighting"

    # generate image
    generate_images(
        pipe=pipe,
        prompt=SD_PROMPT,
        nprompt=SD_N_PROMPT,
        steps=args.steps,
        guidance=args.guidance_scale,
        width=args.width,
        height=args.height,
        output_path=args.output,
        refiner=refiner,
        denoising_end=args.denoising_end,
        generator=generator,
    )


if __name__ == "__main__":
    main()
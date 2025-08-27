import os
import argparse
import time
from typing import Optional

import torch
from PIL import Image

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    PNDMScheduler,
)


SUPPORTED_SCHEDULERS = {
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
    "ddim": DDIMScheduler,
    "pndm": PNDMScheduler,
}


def str2dtype(name: str) -> torch.dtype:
    if name.lower() in {"fp16", "float16", "half"}:
        return torch.float16
    if name.lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def prepare_device(gpu_id: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please install a CUDA-enabled PyTorch build.")
    num_gpus = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= num_gpus:
        raise ValueError(f"Requested --gpu-id={gpu_id}, but available GPU count is {num_gpus}.")
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    return device


def build_scheduler_from_name(pipeline, scheduler_name: str):
    scheduler_name = scheduler_name.lower()
    if scheduler_name not in SUPPORTED_SCHEDULERS:
        raise ValueError(
            f"Unsupported scheduler '{scheduler_name}'. Choose from: {list(SUPPORTED_SCHEDULERS.keys())}"
        )
    scheduler_cls = SUPPORTED_SCHEDULERS[scheduler_name]
    return scheduler_cls.from_config(pipeline.scheduler.config)


def load_pipeline(
    mode: str,
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    scheduler_name: str,
):
    if mode == "txt2img":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )
    elif mode == "img2img":
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )
    else:
        raise ValueError("mode must be 'txt2img' or 'img2img'")

    pipe = pipe.to(device)

    try:
        pipe.scheduler = build_scheduler_from_name(pipe, scheduler_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to set scheduler '{scheduler_name}': {exc}")

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    if torch.backends.cuda.sdp_kernel(
        enable_math=False, enable_flash=True, enable_mem_efficient=True
    ):
        pass
    else:
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    if hasattr(pipe, "watermark"):
        pipe.watermark = None

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)

    return pipe


def ensure_output_path(output: Optional[str]) -> str:
    if output:
        out_dir = os.path.dirname(output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        return output
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"sdxl_{ts}.png")


def run_txt2img(
    model_id: str,
    prompt: str,
    negative_prompt: Optional[str],
    guidance_scale: float,
    num_inference_steps: int,
    height: int,
    width: int,
    seed: Optional[int],
    gpu_id: int,
    dtype_name: str,
    scheduler_name: str,
    output: Optional[str],
):
    device = prepare_device(gpu_id)
    dtype = str2dtype(dtype_name)
    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device=device).manual_seed(seed)

    pipe = load_pipeline("txt2img", model_id, device, dtype, scheduler_name)

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        generator=generator,
    ).images

    out_path = ensure_output_path(output)
    images[0].save(out_path)
    print(f"Saved: {out_path}")


def run_img2img(
    model_id: str,
    prompt: str,
    negative_prompt: Optional[str],
    guidance_scale: float,
    num_inference_steps: int,
    strength: float,
    init_image_path: str,
    seed: Optional[int],
    gpu_id: int,
    dtype_name: str,
    scheduler_name: str,
    output: Optional[str],
):
    if not os.path.exists(init_image_path):
        raise FileNotFoundError(f"init image not found: {init_image_path}")

    device = prepare_device(gpu_id)
    dtype = str2dtype(dtype_name)
    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device=device).manual_seed(seed)

    init_image = Image.open(init_image_path).convert("RGB")

    pipe = load_pipeline("img2img", model_id, device, dtype, scheduler_name)

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        image=init_image,
        strength=strength,
        generator=generator,
    ).images

    out_path = ensure_output_path(output)
    images[0].save(out_path)
    print(f"Saved: {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SDXL inference: txt2img & img2img on a single GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Model repo or local path",
    )
    common.add_argument("--gpu-id", type=int, default=0, help="GPU index to use")
    common.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for inference",
    )
    common.add_argument(
        "--scheduler",
        type=str,
        default="dpmpp_2m",
        choices=list(SUPPORTED_SCHEDULERS.keys()),
        help="Scheduler to use",
    )
    common.add_argument("--steps", type=int, default=30, help="Denoising steps")
    common.add_argument("--guidance", type=float, default=5.0, help="CFG guidance scale")
    common.add_argument("--seed", type=int, default=None, help="Random seed")
    common.add_argument("--negative-prompt", type=str, default=None, help="Negative prompt")
    common.add_argument("--out", type=str, default=None, help="Output image path")

    p_txt2img = subparsers.add_parser("txt2img", parents=[common], help="Text-to-Image")
    p_txt2img.add_argument("--prompt", type=str, required=True, help="Prompt text")
    p_txt2img.add_argument("--height", type=int, default=1024, help="Output height")
    p_txt2img.add_argument("--width", type=int, default=1024, help="Output width")

    p_img2img = subparsers.add_parser("img2img", parents=[common], help="Image-to-Image")
    p_img2img.add_argument("--prompt", type=str, default="", help="Prompt text")
    p_img2img.add_argument("--init-image", type=str, required=True, help="Init image path")
    p_img2img.add_argument("--strength", type=float, default=0.65, help="Noise strength [0,1]")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "txt2img":
        run_txt2img(
            model_id=args.model,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            seed=args.seed,
            gpu_id=args.gpu_id,
            dtype_name=args.dtype,
            scheduler_name=args.scheduler,
            output=args.out,
        )
    elif args.command == "img2img":
        run_img2img(
            model_id=args.model,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            strength=args.strength,
            init_image_path=args.init_image,
            seed=args.seed,
            gpu_id=args.gpu_id,
            dtype_name=args.dtype,
            scheduler_name=args.scheduler,
            output=args.out,
        )
    else:
        raise AssertionError("unreachable")


if __name__ == "__main__":
    main()


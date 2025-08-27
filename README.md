## Stable Diffusion XL Text-to-Image (Single GPU)

Minimal CLI to generate images with SDXL on a single NVIDIA GPU (e.g., one 4090). Supports batch generation, custom size, steps, guidance, and reproducible seeds.

### 1) Environment

Recommended: Python 3.10+ and a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch with CUDA (choose one matching your system).

- CUDA 12.1 wheels (recommended for Ada/4090):
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

- CUDA 11.8 wheels:
```bash
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

If `xformers` fails to install, you can skip it; the script will fall back gracefully.

### 2) Usage

Basic example (uses GPU 0):

```bash
python generate.py "a high quality photo of a corgi astronaut walking on the moon" \
  --seed 1234 --num-inference-steps 30 --width 1024 --height 1024 --output outputs --gpu 0
```

Important flags:
- `--negative-prompt`: Avoid unwanted attributes (e.g., "low quality, blurry")
- `--num-inference-steps`: More steps, generally slower but better quality
- `--guidance-scale`: 5–9 works well for most prompts; lower for more creativity
- `--width/--height`: Multiples of 8. 1024x1024 is SDXL native.
- `--batch-size`: How many images per forward pass (fit to your VRAM)
- `--num-images`: Total images to generate
- `--scheduler`: One of `ddim`, `euler`, `euler_a`, `dpmpp_2m`, `dpmpp_2m_sde`
- `--compile`: Try `torch.compile` for a small speedup (PyTorch 2.x)
- `--model`: Model ID or local path (default `stabilityai/stable-diffusion-xl-base-1.0`)

Output images are saved to the `outputs/` folder, named `sdxl_00000.png`, `sdxl_00001.png`, ...

### 3) Tips
- For a single 4090, batch size 1–2 at 1024² usually fits comfortably.
- If you see OOM, reduce `--batch-size`, `--width`, or `--height`.
- To use a different GPU, set `--gpu` or export `CUDA_VISIBLE_DEVICES`.
- The first run will download the model weights from Hugging Face.

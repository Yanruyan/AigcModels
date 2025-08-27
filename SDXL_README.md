## SDXL 文生图/图生图 推理脚本

本仓库包含一个简洁的 SDXL 推理脚本 `sd_infer.py`，支持：

- 文生图（txt2img）
- 图生图（img2img）
- 指定使用单张 GPU（例如 8 卡 4090 中的任意 1 卡）

默认模型：`stabilityai/stable-diffusion-xl-base-1.0`（SDXL Base 1.0）。

### 1) 环境安装

建议创建独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

安装 PyTorch（CUDA 版本，按你的 CUDA 版本选择）：

```bash
# 以 CUDA 12.4 为例（请根据你的环境更换 index-url）
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```

安装其它依赖：

```bash
pip install -r requirements.txt
```

可选：如果你想启用 xFormers 注意力（PyTorch 2 的 SDP 已足够），可尝试：

```bash
pip install xformers
```

### 2) 使用说明

你可以通过 `--gpu-id` 精确指定使用哪一张 GPU（从 0 开始）。也可以通过 `CUDA_VISIBLE_DEVICES` 控制可见 GPU，再配合 `--gpu-id` 选择。

#### 文生图（txt2img）

```bash
# 例：在第 3 张 4090 上（索引从 0 开始），生成 1024x1024 图片
CUDA_VISIBLE_DEVICES=0,1,2,3 python sd_infer.py txt2img \
  --gpu-id 3 \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --prompt "a cinematic photo of a futuristic city at dusk, ultra-detailed" \
  --negative-prompt "low quality, blurry, watermark" \
  --steps 30 \
  --guidance 5.0 \
  --height 1024 --width 1024 \
  --seed 42 \
  --out outputs/city.png
```

常用可调参数：

- `--steps`：去噪步数。SDXL 常用 20-40。
- `--guidance`：CFG 引导强度。SDXL 常用 4-7。
- `--scheduler`：采样器，支持 `dpmpp_2m`、`euler_a`、`ddim`、`pndm`。
- `--dtype`：精度（`float16`/`bfloat16`/`float32`）。4090 推荐 `float16`。

#### 图生图（img2img）

```bash
python sd_infer.py img2img \
  --gpu-id 0 \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --prompt "a watercolor painting style" \
  --negative-prompt "low quality, artifacts" \
  --steps 30 \
  --guidance 5.0 \
  --strength 0.65 \
  --init-image path/to/your_input.jpg \
  --seed 123 \
  --out outputs/painted.png
```

提示：`--strength` 越大，改动（去噪）越强，越接近“重绘”；越小保留原图越多。

### 3) 性能与显存建议

- 4090（24GB）使用 SDXL Base，1024x1024、30 步、`float16` 通常可稳定运行。
- 脚本默认启用 VAE slicing/tiling，并优先使用 PyTorch 2 的 SDP 注意力。
- 如果你遇到显存不足，可尝试降低分辨率、步数或使用 `--dtype bfloat16`（视显卡支持）。

### 4) 常见问题

- 如果首次运行下载模型较慢，可配置 HuggingFace 镜像或提前 `git lfs` 拉取。
- PyTorch GPU 版本需要与你的 CUDA 驱动匹配，安装失败请参考 PyTorch 官网选择正确指令。
- 如果想使用本地模型目录，`--model` 指向本地路径即可。


### Stable Diffusion XL 文生图（单卡/单张）

使用 Diffusers 以 SDXL 实现文生图，默认单张、单 GPU（可通过 `--device` 选择具体 GPU）。

#### 1) 安装

- 先安装与你 CUDA 匹配的 PyTorch（示例为 CUDA 12.1）：

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

- 然后安装依赖：

```bash
pip install -r requirements.txt
```

- 可选：安装 xFormers 以降低显存占用（与你的 PyTorch/CUDA 版本匹配）：

```bash
pip install xformers
```

#### 2) 用法

最简示例：

```bash
CUDA_VISIBLE_DEVICES=0 python generate_image.py \
  --prompt "a cozy cabin in the snowy mountains, golden hour" \
  --output out.png
```

指定设备、尺寸与随机种子：

```bash
python generate_image.py \
  --prompt "studio portrait of a cat, 85mm" \
  --width 1024 --height 1024 \
  --steps 30 --guidance-scale 5.0 \
  --seed 123 \
  --device cuda:0 \
  --output cat.png
```

启用 SDXL Refiner（提升细节，稍慢）：

```bash
python generate_image.py \
  --prompt "ultra detailed concept art of a cyberpunk city" \
  --use-refiner \
  --output city.png
```

常用参数：
- **--prompt**: 文本提示词
- **--negative-prompt**: 负面提示词（默认已含常见低质项）
- **--width/--height**: 分辨率（8 的倍数，SDXL 推荐 1024×1024）
- **--steps**: 采样步数（默认 30）
- **--guidance-scale**: CFG 指导强度（默认 5.0）
- **--seed**: 随机种子（固定可复现）
- **--device**: 设备，如 `cuda:0` 或 `cpu`
- **--use-refiner**: 启用 SDXL Refiner 二阶段细化
- **--no-xformers**: 禁止使用 xFormers 注意力
- **--cpu-offload**: 启用 CPU offload（更省显存但更慢）

#### 3) 模型

默认：
- Base: `stabilityai/stable-diffusion-xl-base-1.0`
- Refiner: `stabilityai/stable-diffusion-xl-refiner-1.0`

如需替换本地或私有模型，使用：

```bash
python generate_image.py --prompt "..." --model /path/to/base --use-refiner --refiner-model /path/to/refiner
```

#### 4) 备注

- 首次运行会从 `huggingface` 下载模型，需网络可用且已接受相应条款（如需登录：`huggingface-cli login`）。
- 4090 单卡 FP16 推理 1024×1024 一般 6~10GB 显存，开启 refiner 会略增。无 xFormers 时可开启 `--cpu-offload` 以降低显存（更慢）。

# AigcModels
基于开源aigc生图模型实现webui/compyui的生图工作流，实现多种风格迁移功能，包括：换背景、换模特、换商品等

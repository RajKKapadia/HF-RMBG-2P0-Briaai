# RMBG-2.0 Background Removal

A simple Python script for removing backgrounds from images using BriaAI's [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) model.

## Requirements

- Python 3.12+
- CUDA-capable GPU (optional, falls back to CPU)

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

## Usage

1. Place your input image in the project directory
2. Update the `input_image_path` variable in `main.py` to point to your image
3. Run the script:

```bash
python main.py
```

The output image with transparent background will be saved as `no_bg_image.png`.

## Dependencies

- torch
- torchvision
- transformers
- Pillow
- kornia
- timm

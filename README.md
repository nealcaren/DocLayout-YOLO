<div align="center">

English | [简体中文](./README-zh_CN.md)

<h1>DocLayout-YOLO: Python 3.12 Compatible Fork</h1>

Fork of [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) with full Python 3.9-3.12 support and dependency fixes for modern development environments.

[![arXiv](https://img.shields.io/badge/arXiv-2405.14458-b31b1b.svg)](https://arxiv.org/abs/2410.12628) [![Original Repo](https://img.shields.io/badge/Original-Repo-blue)](https://github.com/opendatalab/DocLayout-YOLO)

</div>

## What's Different in This Fork?

This fork addresses several dependency and compatibility issues to enable training and inference on modern Python environments (3.9-3.12) and Apple Silicon:

### Dependency Updates for Python 3.12 Compatibility

- **torch**: Updated to `>=2.2.0` (Python 3.12 wheel support)
- **torchvision**: Updated to `>=0.17.0` (compatible with torch 2.2+)
- **scipy**: Updated to `>=1.11.0` (Python 3.12 support)
- **pandas**: Updated to `>=2.1.0` (Python 3.12 wheels for macOS)
- **pywavelets**: Conditionally `>=1.5.0` for Python 3.12+ (fixes build failures)
- **albumentations**: Constrained to `>=1.0.3,<1.4.11` (numpy compatibility with tensorflow)
- **huggingface_hub**: Added as core dependency for model downloads

### Platform Support

- **Python versions**: 3.9, 3.10, 3.11, 3.12 (dropped 3.8)
- **Apple Silicon**: MPS device support added to training script
- **macOS**: All dependencies now have pre-built wheels (no Xcode compilation needed)

### Package Manager Compatibility

- Tested with `uv` and `pip` package managers
- No source builds required - all wheels available

## Abstract

> DocLayout-YOLO is a real-time and robust layout detection model for diverse documents, based on YOLO-v10. This model is enriched with diversified document pre-training and structural optimization tailored for layout detection. In the pre-training phase, we introduce Mesh-candidate BestFit, viewing document synthesis as a two-dimensional bin packing problem, and create a large-scale diverse synthetic document dataset, DocSynth-300K. In terms of model structural optimization, we propose a module with Global-to-Local Controllability for precise detection of document elements across varying scales.

<p align="center">
  <img src="assets/comp.png" width=52%>
  <img src="assets/radar.png" width=44%> <br>
</p>

## Quick Start

**💡 Need cloud GPU training?** We have guides for:
- **[Google Colab](COLAB_TRAINING.md)** - Free GPU (T4), great for learning and experiments
- **[Hugging Face Jobs](HF_JOBS_TRAINING.md)** - Paid GPU options (T4-A100), best for production training

### 1. Environment Setup

Using `uv` (recommended for faster installs):

```bash
git clone https://github.com/nealcaren/DocLayout-YOLO.git
cd DocLayout-YOLO
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

Or using `pip`:

```bash
git clone https://github.com/nealcaren/DocLayout-YOLO.git
cd DocLayout-YOLO
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

**Note:** For inference only:

```bash
pip install doclayout-yolo
```

### 2. Prediction

You can make predictions using either a script or the SDK:

- **Script**

  Run the following command to make a prediction using the script:

  ```bash
  python demo.py --model path/to/model --image-path path/to/image
  ```

- **SDK**

  Here is an example of how to use the SDK for prediction:

  ```python
  import cv2
  from doclayout_yolo import YOLOv10

  # Load the pre-trained model
  model = YOLOv10("path/to/provided/model")

  # Perform prediction
  det_res = model.predict(
      "path/to/image",   # Image to predict
      imgsz=1024,        # Prediction image size
      conf=0.2,          # Confidence threshold
      device="cuda:0"    # Device: 'cuda:0', 'cpu', or 'mps' (Apple Silicon)
  )

  # Annotate and save the result
  annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
  cv2.imwrite("result.jpg", annotated_frame)
  ```

We provide model fine-tuned on **DocStructBench** for prediction, **which is capable of handling various document types**. Model can be downloaded from [here](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/tree/main) and example images can be found under `assets/example`.

<p align="center">
  <img src="assets/showcase.png" width=100%> <br>
</p>

**Loading models from Hugging Face:**

```python
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10

# Method 1: Download and load
filepath = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
)
model = YOLOv10(filepath)

# Method 2: Direct load
model = YOLOv10.from_pretrained("juliozhao/DocLayout-YOLO-DocStructBench")
```

## Training

**🚀 Cloud Training Options:**
- **[Google Colab Guide](COLAB_TRAINING.md)** - Free T4 GPU, perfect for getting started
- **[Hugging Face Jobs Guide](HF_JOBS_TRAINING.md)** - Scalable paid options (T4 to A100), ideal for production

### Data Preparation

Your training data should follow this structure:

```
your_dataset/
├── dataset.yaml          # Dataset configuration
└── dataset_name/         # Data folder
    ├── train.txt         # List of training image paths
    ├── val.txt           # List of validation image paths
    ├── images/           # Image files
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── labels/           # YOLO format annotations
        ├── img1.txt
        ├── img2.txt
        └── ...
```

**dataset.yaml example:**

```yaml
# Dataset root path
path: /path/to/your_dataset/dataset_name

# Train and validation data (relative to 'path')
train: train.txt
val: val.txt

# Class names
names:
  0: headline
  1: text
  2: image
  3: caption
  4: table
  # ... add your classes
```

**train.txt and val.txt format:**

Each line should contain a path to an image (relative to `path` or absolute):

```
images/img1.jpg
images/img2.jpg
images/img3.jpg
```

**Label format (YOLO):**

Each `.txt` file in `labels/` corresponds to an image and contains one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized (0-1).

### Training Examples

#### Example 1: Training from Scratch (Custom Dataset)

Train a small model on your custom newspaper layout dataset:

```bash
python train.py \
  --data /path/to/colorado-historical/layout_data/newspaper \
  --model n \
  --epoch 100 \
  --image-size 640 \
  --batch-size 16 \
  --device mps \
  --project ./runs/train
```

**Parameters:**
- `--data`: Path to your dataset (without .yaml extension)
- `--model`: Model size (`n`=nano, `s`=small, `m`=medium, `l`=large, `x`=xlarge)
- `--epoch`: Number of training epochs
- `--image-size`: Input image size (640 for smaller docs, 1024+ for complex layouts)
- `--batch-size`: Batch size per device
- `--device`: Device to use (`mps` for Apple Silicon, `cuda:0` for NVIDIA GPU, `cpu`)
- `--project`: Output directory for results

For CPU training (slower):

```bash
python train.py \
  --data /path/to/colorado-historical/layout_data/newspaper \
  --model n \
  --epoch 50 \
  --image-size 640 \
  --batch-size 8 \
  --device cpu \
  --workers 2 \
  --project ./runs/train
```

#### Example 2: Fine-tuning DocStructBench Model

Fine-tune the pre-trained DocStructBench model on your custom dataset for better performance:

```bash
# First, download the pre-trained model
python -c "from huggingface_hub import hf_hub_download; \
           hf_hub_download(repo_id='juliozhao/DocLayout-YOLO-DocStructBench', \
                          filename='doclayout_yolo_docstructbench_imgsz1024.pt', \
                          local_dir='./pretrained')"

# Then fine-tune on your data
python train.py \
  --data /path/to/your_dataset/dataset_name \
  --model m \
  --epoch 50 \
  --image-size 1024 \
  --batch-size 8 \
  --lr0 0.001 \
  --pretrain ./pretrained/doclayout_yolo_docstructbench_imgsz1024.pt \
  --device mps \
  --project ./runs/finetune \
  --patience 20
```

**Key differences for fine-tuning:**
- `--pretrain`: Path to pre-trained model weights
- `--lr0`: Lower learning rate (0.001 vs 0.02) for fine-tuning
- `--patience`: Early stopping patience (stops if no improvement after N epochs)
- `--image-size`: Match the pre-trained model size (1024)

#### Example 3: Multi-GPU Training

For systems with multiple NVIDIA GPUs:

```bash
python train.py \
  --data /path/to/your_dataset/dataset_name \
  --model m \
  --epoch 100 \
  --image-size 1024 \
  --batch-size 16 \
  --device 0,1,2,3 \
  --workers 8 \
  --project ./runs/train_multi_gpu
```

### Advanced Training Options

```bash
python train.py \
  --data dataset_name \
  --model m \
  --epoch 100 \
  --image-size 1024 \
  --batch-size 16 \
  --optimizer AdamW \          # Optimizer: 'SGD', 'Adam', 'AdamW', 'auto'
  --lr0 0.02 \                 # Initial learning rate
  --momentum 0.9 \             # SGD momentum/Adam beta1
  --warmup-epochs 3.0 \        # Warmup epochs
  --mosaic 1.0 \               # Mosaic augmentation probability
  --val 1 \                    # Enable validation during training
  --val-period 1 \             # Validate every N epochs
  --plot 1 \                   # Generate training plots
  --save-period 10 \           # Save checkpoint every N epochs
  --patience 50 \              # Early stopping patience
  --device mps \               # 'mps', 'cuda:0', '0,1,2,3', or 'cpu'
  --workers 4 \                # Number of data loading workers
  --project ./runs/train \     # Project directory
  --pretrain path/to/weights.pt  # Optional: pretrained weights
```

### Resuming Training

If training is interrupted, resume from the last checkpoint:

```bash
python train.py \
  --data dataset_name \
  --model m \
  --epoch 100 \
  --image-size 1024 \
  --batch-size 16 \
  --device mps \
  --project ./runs/train \
  --resume
```

## DocSynth300K Dataset

<p align="center">
  <img src="assets/docsynth300k.png" width=100%>
</p>

### Data Download

Use following command to download dataset (about 113G):

```python
from huggingface_hub import snapshot_download

# Download DocSynth300K
snapshot_download(
    repo_id="juliozhao/DocSynth300K",
    local_dir="./docsynth300k-hf",
    repo_type="dataset"
)

# If download is interrupted, resume with:
snapshot_download(
    repo_id="juliozhao/DocSynth300K",
    local_dir="./docsynth300k-hf",
    repo_type="dataset",
    resume_download=True
)
```

### Data Formatting & Pre-training

Convert original `.parquet` format to YOLO format:

```bash
python format_docsynth300k.py
```

The converted data will be stored at `./layout_data/docsynth300k`.

For DocSynth300K pre-training commands, see [assets/script.sh](assets/script.sh#L2).

## Training on Public DLA Datasets

### Data Preparation

Download prepared YOLO-format datasets:

| Dataset | Download |
|:--:|:--:|
| D4LA | [link](https://huggingface.co/datasets/juliozhao/doclayout-yolo-D4LA) |
| DocLayNet | [link](https://huggingface.co/datasets/juliozhao/doclayout-yolo-DocLayNet) |

Expected structure:

```bash
./layout_data
├── D4LA
│   ├── images
│   ├── labels
│   ├── test.txt
│   └── train.txt
└── doclaynet
    ├── images
    ├── labels
    ├── val.txt
    └── train.txt
```

### Benchmark Results

Training on 8 GPUs with global batch size of 64:

| Dataset | Model | DocSynth300K Pretrained? | imgsz | AP50 | mAP | Checkpoint |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| D4LA | DocLayout-YOLO | ✗ | 1600 | 81.7 | 69.8 | [checkpoint](https://huggingface.co/juliozhao/DocLayout-YOLO-D4LA-from_scratch) |
| D4LA | DocLayout-YOLO | ✓ | 1600 | 82.4 | 70.3 | [checkpoint](https://huggingface.co/juliozhao/DocLayout-YOLO-D4LA-Docsynth300K_pretrained) |
| DocLayNet | DocLayout-YOLO | ✗ | 1120 | 93.0 | 77.7 | [checkpoint](https://huggingface.co/juliozhao/DocLayout-YOLO-DocLayNet-from_scratch) |
| DocLayNet | DocLayout-YOLO | ✓ | 1120 | 93.4 | 79.7 | [checkpoint](https://huggingface.co/juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained) |

For training/evaluation commands, see [assets/script.sh](assets/script.sh).

## Notes

- **PDF Extraction**: For PDF content extraction, see [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) and [MinerU](https://github.com/opendatalab/MinerU)
- **Batch Inference**: Pass a list of image paths to `model.predict()` for batch processing
- **Memory Issues**: If pre-training on large datasets is interrupted, use `--resume` to continue

## Acknowledgement

This fork is based on [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) by the OpenDataLab team.

The original code base is built with [ultralytics](https://github.com/ultralytics/ultralytics) and [YOLO-v10](https://github.com/lyuwenyu/RT-DETR).

Thanks to all contributors!

## Citation

```bibtex
@misc{zhao2024doclayoutyoloenhancingdocumentlayout,
      title={DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception},
      author={Zhiyuan Zhao and Hengrui Kang and Bin Wang and Conghui He},
      year={2024},
      eprint={2410.12628},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.12628},
}

@article{wang2024mineru,
  title={MinerU: An Open-Source Solution for Precise Document Content Extraction},
  author={Wang, Bin and Xu, Chao and Zhao, Xiaomeng and Ouyang, Linke and Wu, Fan and Zhao, Zhiyuan and Xu, Rui and Liu, Kaiwen and Qu, Yuan and Shang, Fukai and others},
  journal={arXiv preprint arXiv:2409.18839},
  year={2024}
}
```

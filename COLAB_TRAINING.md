# Training DocLayout-YOLO on Google Colab

This guide walks you through training DocLayout-YOLO on Google Colab, which provides free GPU access and eliminates local hardware constraints.

## Why Use Google Colab?

- **Free GPU access** (NVIDIA T4, up to 15GB VRAM)
- **No local memory constraints** - perfect if your computer can't handle training
- **Pre-installed ML libraries** - faster setup
- **No installation required** - runs in your browser

## Table of Contents

1. [Quick Start - Complete Notebook](#quick-start---complete-notebook)
2. [Step-by-Step Setup](#step-by-step-setup)
3. [Uploading Your Training Data](#uploading-your-training-data)
4. [Training from Scratch](#training-from-scratch)
5. [Fine-tuning Pre-trained Models](#fine-tuning-pre-trained-models)
6. [Downloading Results](#downloading-results)
7. [Tips & Troubleshooting](#tips--troubleshooting)

---

## Quick Start - Complete Notebook

Here's a complete Colab notebook you can copy and run:

### Option 1: Training from Scratch

```python
# ===== CELL 1: Setup Environment =====
# Enable GPU: Runtime > Change runtime type > GPU

# Check GPU availability
!nvidia-smi

# Clone the repository
!git clone https://github.com/nealcaren/DocLayout-YOLO.git
%cd DocLayout-YOLO

# Install dependencies
!pip install -e . -q

print("✓ Setup complete!")

# ===== CELL 2: Upload Training Data =====
# Option A: Upload from local computer (small datasets < 1GB)
from google.colab import files
import zipfile
import os

print("Please upload your training data as a ZIP file...")
print("Your ZIP should contain: dataset.yaml, images/, and labels/")
uploaded = files.upload()

# Extract the uploaded ZIP
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('/content/training_data')
        print("✓ Data extracted to /content/training_data")

# Option B: Upload from Google Drive (large datasets)
# Uncomment these lines if using Google Drive:
# from google.colab import drive
# drive.mount('/content/drive')
# !cp -r /content/drive/MyDrive/your_dataset /content/training_data

# ===== CELL 3: Verify Data Structure =====
# Check your data structure
!ls -la /content/training_data/

# Display dataset config
!cat /content/training_data/newspaper.yaml  # Replace with your YAML filename

# ===== CELL 4: Start Training =====
# Training from scratch (adjust parameters as needed)
!python train.py \
  --data /content/training_data/newspaper \
  --model n \
  --epoch 100 \
  --image-size 640 \
  --batch-size 16 \
  --device 0 \
  --project /content/runs/train \
  --workers 2 \
  --val 1 \
  --plot 1

print("✓ Training complete!")

# ===== CELL 5: Download Results =====
# Compress results
!zip -r training_results.zip /content/runs/train

# Download
from google.colab import files
files.download('training_results.zip')

print("✓ Results downloaded!")
```

### Option 2: Fine-tuning Pre-trained Model

```python
# ===== CELL 1: Setup Environment =====
!nvidia-smi

!git clone https://github.com/nealcaren/DocLayout-YOLO.git
%cd DocLayout-YOLO
!pip install -e . -q

print("✓ Setup complete!")

# ===== CELL 2: Download Pre-trained Model =====
from huggingface_hub import hf_hub_download

print("Downloading pre-trained DocStructBench model...")
model_path = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt",
    local_dir="/content/pretrained"
)
print(f"✓ Model downloaded to: {model_path}")

# ===== CELL 3: Upload Your Training Data =====
from google.colab import files
import zipfile

print("Please upload your training data as a ZIP file...")
uploaded = files.upload()

for filename in uploaded.keys():
    if filename.endswith('.zip'):
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('/content/training_data')
        print("✓ Data extracted")

# ===== CELL 4: Fine-tune Model =====
!python train.py \
  --data /content/training_data/newspaper \
  --model m \
  --epoch 50 \
  --image-size 1024 \
  --batch-size 8 \
  --lr0 0.001 \
  --pretrain /content/pretrained/doclayout_yolo_docstructbench_imgsz1024.pt \
  --device 0 \
  --project /content/runs/finetune \
  --workers 2 \
  --patience 20 \
  --val 1 \
  --plot 1

print("✓ Fine-tuning complete!")

# ===== CELL 5: Download Results =====
!zip -r finetuned_model.zip /content/runs/finetune

from google.colab import files
files.download('finetuned_model.zip')

print("✓ Results downloaded!")
```

---

## Step-by-Step Setup

### 1. Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account
3. Click **File > New notebook**

### 2. Enable GPU

**IMPORTANT:** You must enable GPU for training!

1. Click **Runtime** in the menu bar
2. Select **Change runtime type**
3. Under **Hardware accelerator**, select **GPU**
4. Click **Save**

### 3. Verify GPU Access

In your first cell, run:

```python
!nvidia-smi
```

You should see GPU information (typically Tesla T4 with ~15GB memory).

### 4. Clone Repository and Install

```python
# Clone the repository
!git clone https://github.com/nealcaren/DocLayout-YOLO.git
%cd DocLayout-YOLO

# Install dependencies (this may take 2-3 minutes)
!pip install -e . -q

# Verify installation
from doclayout_yolo import YOLOv10
print("✓ Installation successful!")
```

---

## Uploading Your Training Data

You have three options for getting your data into Colab:

### Option 1: Upload Directly (Best for < 1GB)

```python
from google.colab import files
import zipfile

# Create a ZIP file of your data locally first, containing:
# - dataset.yaml
# - dataset_name/ folder with images/ and labels/

print("Upload your training data ZIP file:")
uploaded = files.upload()

# Extract
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('/content/training_data')
        print(f"✓ Extracted to /content/training_data")
```

### Option 2: Google Drive (Best for Large Datasets)

1. Upload your data to Google Drive first
2. In Colab:

```python
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Copy data from Drive to Colab
!cp -r /content/drive/MyDrive/path/to/your/dataset /content/training_data

# Verify
!ls -la /content/training_data
```

### Option 3: Download from URL

If your data is hosted online:

```python
# Example: Download from a public URL
!wget https://your-url.com/dataset.zip -O dataset.zip
!unzip dataset.zip -d /content/training_data
```

### Verify Data Structure

```python
# Check folder structure
!ls -la /content/training_data/

# View your YAML config
!cat /content/training_data/newspaper.yaml
```

Expected structure:
```
/content/training_data/
├── newspaper.yaml
└── newspaper/
    ├── train.txt
    ├── val.txt
    ├── images/
    └── labels/
```

---

## Training from Scratch

### Basic Training

```python
!python train.py \
  --data /content/training_data/newspaper \
  --model n \
  --epoch 100 \
  --image-size 640 \
  --batch-size 16 \
  --device 0 \
  --project /content/runs/train \
  --workers 2 \
  --val 1 \
  --plot 1
```

### Recommended Settings for Colab Free Tier

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--model` | `n` or `s` | Smaller models fit in free GPU memory |
| `--batch-size` | `8-16` | Avoid OOM (Out of Memory) errors |
| `--workers` | `2` | Colab CPU cores |
| `--device` | `0` | Use the single GPU |
| `--image-size` | `640` | Start smaller, increase if memory allows |

### Monitor Training

Training progress will display in the cell output:

```
Epoch   GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
  1/100   8.45G     1.234      2.456      0.789        145       640
  2/100   8.45G     1.123      2.234      0.756        152       640
...
```

### If Training Stops (Session Timeout)

Colab free tier sessions last ~12 hours. To resume:

```python
# Resume from last checkpoint
!python train.py \
  --data /content/training_data/newspaper \
  --model n \
  --epoch 100 \
  --image-size 640 \
  --batch-size 16 \
  --device 0 \
  --project /content/runs/train \
  --resume
```

---

## Fine-tuning Pre-trained Models

Fine-tuning is **highly recommended** - it gives better results with less training time!

### Download Pre-trained Model

```python
from huggingface_hub import hf_hub_download

# Download DocStructBench model
model_path = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt",
    local_dir="/content/pretrained"
)

print(f"Model saved to: {model_path}")
```

### Fine-tune on Your Data

```python
!python train.py \
  --data /content/training_data/newspaper \
  --model m \
  --epoch 50 \
  --image-size 1024 \
  --batch-size 8 \
  --lr0 0.001 \
  --pretrain /content/pretrained/doclayout_yolo_docstructbench_imgsz1024.pt \
  --device 0 \
  --project /content/runs/finetune \
  --workers 2 \
  --patience 20 \
  --val 1 \
  --plot 1
```

**Key differences for fine-tuning:**
- `--lr0 0.001`: Lower learning rate (don't change pre-trained weights too much)
- `--patience 20`: Early stopping if no improvement after 20 epochs
- `--image-size 1024`: Match pre-trained model size
- Fewer epochs needed (50 vs 100)

---

## Downloading Results

After training completes, download your results:

### Download Everything

```python
# Compress all training outputs
!zip -r training_results.zip /content/runs/train

# Download to your computer
from google.colab import files
files.download('training_results.zip')
```

### Download Only Best Model

```python
# Find the best model weights
!ls -lh /content/runs/train/*/weights/

# Download just the best weights
from google.colab import files
files.download('/content/runs/train/yolov10n_newspaper_epoch100_imgsz640_bs16_pretrain_None/weights/best.pt')
```

### Save to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Drive
!cp -r /content/runs/train /content/drive/MyDrive/doclayout_training_results

print("✓ Results saved to Google Drive")
```

---

## Tips & Troubleshooting

### Out of Memory (OOM) Errors

If you get `CUDA out of memory` errors:

```python
# Try these adjustments:
!python train.py \
  --batch-size 4 \        # Reduce batch size
  --image-size 512 \      # Reduce image size
  --model n \             # Use smaller model (n instead of s/m/l)
  ...
```

### Speed Up Training

```python
# Use mixed precision (faster training)
!python train.py \
  --batch-size 16 \
  --image-size 640 \
  --device 0 \
  --amp \                 # Enable automatic mixed precision
  ...
```

### Check Training Progress

View training metrics in real-time:

```python
# In a separate cell, while training runs:
from IPython.display import Image, display
import time

while True:
    try:
        # Display latest results
        display(Image('/content/runs/train/yolov10n_*/results.png'))
        time.sleep(30)  # Update every 30 seconds
    except:
        break
```

### Free Tier Limitations

**Google Colab Free Tier:**
- ~12 hour session limit
- T4 GPU (~15GB VRAM)
- Sessions can disconnect if idle
- Limited to sequential training (can't run multiple experiments in parallel)

**Solutions:**
- Download checkpoints regularly
- Use `--save-period 5` to save every 5 epochs
- Consider Colab Pro ($10/month) for:
  - Longer sessions (24 hours)
  - Better GPUs (V100, A100)
  - Priority access

### Keeping Session Active

To prevent disconnection during long training:

```javascript
// Run this in browser console (F12)
// Simulates activity to keep session alive
function ClickConnect(){
    console.log("Clicking connect button");
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)  // Every 60 seconds
```

### Monitor GPU Usage

```python
# In a separate cell, monitor GPU while training:
!watch -n 1 nvidia-smi
```

---

## Complete Example Workflow

Here's a complete workflow from start to finish:

```python
# ===== Setup =====
!nvidia-smi
!git clone https://github.com/nealcaren/DocLayout-YOLO.git
%cd DocLayout-YOLO
!pip install -e . -q

# ===== Upload Data from Google Drive =====
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/newspaper_dataset /content/training_data

# ===== Download Pre-trained Model =====
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt",
    local_dir="/content/pretrained"
)

# ===== Train =====
!python train.py \
  --data /content/training_data/newspaper \
  --model m \
  --epoch 50 \
  --image-size 1024 \
  --batch-size 8 \
  --lr0 0.001 \
  --pretrain /content/pretrained/doclayout_yolo_docstructbench_imgsz1024.pt \
  --device 0 \
  --project /content/runs/finetune \
  --workers 2 \
  --patience 20 \
  --val 1 \
  --plot 1 \
  --save-period 5

# ===== Save Results to Drive =====
!cp -r /content/runs/finetune /content/drive/MyDrive/newspaper_model_results

# ===== Download Best Model =====
from google.colab import files
files.download('/content/runs/finetune/yolov10m_newspaper_*/weights/best.pt')

print("✓ All done!")
```

---

## Additional Resources

- **Colab Keyboard Shortcuts**: `Cmd/Ctrl + M H`
- **Check GPU Quota**: Runtime > View resources
- **Reconnect After Disconnect**: Runtime > Reconnect
- **Clear Outputs**: Edit > Clear all outputs (before saving notebook)

---

## Questions or Issues?

If you encounter problems:

1. Check the training output for error messages
2. Verify your data structure matches the expected format
3. Try reducing `--batch-size` if you get OOM errors
4. Make sure GPU is enabled (Runtime > Change runtime type)
5. Review the main [README.md](README.md) for data format details

Happy training! 🚀

# Training DocLayout-YOLO on Hugging Face Jobs

This guide shows you how to train DocLayout-YOLO using [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs), which provides dedicated GPU infrastructure for training jobs.

## What are Hugging Face Jobs?

Hugging Face Jobs let you run training jobs on Hugging Face's infrastructure with various GPU options. Unlike Colab, you get:

- **Dedicated GPUs** - No session timeouts or interruptions
- **Various hardware options** - From T4 to A100 GPUs
- **Longer running times** - No 12-hour limits
- **Integrated with HF Hub** - Easy model/dataset access
- **Pay-per-use** - Only pay for what you use

## Benefits vs. Colab

| Feature | HF Jobs | Google Colab Free |
|---------|---------|-------------------|
| Session time | Unlimited | ~12 hours |
| GPU options | T4, L4, A10G, A100 | T4 only |
| Interruptions | None | Idle disconnects |
| Pricing | Pay-per-use (~$0.60/hr for T4) | Free (limited) |
| Integration | HF datasets/models | Manual upload |

## Prerequisites

1. **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co)
2. **HF Token**: Create at [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. **Payment Method**: Add at [Settings > Billing](https://huggingface.co/settings/billing)
4. **CLI Installed**: `pip install huggingface_hub[cli]`

## Table of Contents

1. [Quick Start](#quick-start)
2. [Setup](#setup)
3. [Preparing Your Training Data](#preparing-your-training-data)
4. [Creating a Training Job](#creating-a-training-job)
5. [Job Configuration Options](#job-configuration-options)
6. [Monitoring Jobs](#monitoring-jobs)
7. [Retrieving Results](#retrieving-results)
8. [Pricing & Hardware Options](#pricing--hardware-options)
9. [Example Workflows](#example-workflows)

---

## Quick Start

Here's a complete example to train DocLayout-YOLO on HF Jobs:

### 1. Install and Login

```bash
# Install the CLI
pip install "huggingface_hub[cli]"

# Login with your token
huggingface-cli login
```

### 2. Prepare Training Script

Create `train_job.py`:

```python
#!/usr/bin/env python3
"""
Training script for Hugging Face Jobs
This script will be executed on HF infrastructure
"""

import os
from huggingface_hub import snapshot_download
from doclayout_yolo import YOLOv10

# Download training data from HF Hub (if hosted there)
# Or use pre-uploaded data in your HF Space storage
def setup_data():
    """Download or prepare training data"""
    data_path = os.getenv("HF_JOB_DATA_PATH", "/data")

    # Option 1: Download from HF dataset
    # dataset_path = snapshot_download(
    #     repo_id="your-username/your-dataset",
    #     repo_type="dataset",
    #     local_dir=f"{data_path}/training_data"
    # )

    # Option 2: Data is already in job storage
    dataset_path = f"{data_path}/training_data"

    return dataset_path

def train():
    """Main training function"""
    data_path = setup_data()

    # Download pre-trained model (for fine-tuning)
    from huggingface_hub import hf_hub_download
    pretrained_model = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        local_dir="/tmp/pretrained"
    )

    # Load model
    model = YOLOv10(pretrained_model)

    # Train
    results = model.train(
        data=f"{data_path}/newspaper.yaml",
        epochs=50,
        imgsz=1024,
        batch=16,
        lr0=0.001,
        device="0",  # Use first GPU
        workers=4,
        val=True,
        plots=True,
        project="/tmp/runs/train",
        name="newspaper_finetune",
        patience=20,
        save_period=5
    )

    print("✓ Training complete!")

    # Upload results to HF Hub
    upload_results()

def upload_results():
    """Upload trained model to Hugging Face Hub"""
    from huggingface_hub import HfApi

    api = HfApi()

    # Create model repository (or use existing)
    repo_id = f"{os.getenv('HF_USERNAME')}/doclayout-newspaper"

    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
    except:
        pass

    # Upload best model weights
    api.upload_folder(
        folder_path="/tmp/runs/train/newspaper_finetune",
        repo_id=repo_id,
        path_in_repo="training_results"
    )

    print(f"✓ Results uploaded to: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    # Install DocLayout-YOLO if not in environment
    os.system("pip install git+https://github.com/nealcaren/DocLayout-YOLO.git")

    train()
```

### 3. Create Job Configuration

Create `job_config.yaml`:

```yaml
name: doclayout-newspaper-training
compute:
  gpu: "t4"  # Options: t4, l4, a10g, a100
  gpu_count: 1
  cpu: 4
  memory: "16GB"

environment:
  python_version: "3.10"
  requirements:
    - torch>=2.2.0
    - torchvision>=0.17.0
    - huggingface_hub

script: train_job.py

timeout: 7200  # 2 hours max (in seconds)
```

### 4. Submit Job

```bash
# Submit the job
huggingface-cli jobs submit \
  --config job_config.yaml \
  --script train_job.py

# You'll get a job ID like: job-123abc
```

### 5. Monitor and Retrieve Results

```bash
# Check job status
huggingface-cli jobs status job-123abc

# View logs
huggingface-cli jobs logs job-123abc

# Download results (once complete)
huggingface-cli download your-username/doclayout-newspaper
```

---

## Setup

### Install Hugging Face CLI

```bash
pip install "huggingface_hub[cli]"

# Verify installation
huggingface-cli --version
```

### Login to Hugging Face

```bash
huggingface-cli login
```

You'll be prompted to enter your access token. Get one from:
https://huggingface.co/settings/tokens

**Token Permissions Required:**
- ✓ Read access to repos
- ✓ Write access to repos
- ✓ Manage billing (for Jobs)

### Add Payment Method

1. Go to [Settings > Billing](https://huggingface.co/settings/billing)
2. Click **Add payment method**
3. Enter your credit/debit card details
4. Set a spending limit (optional but recommended)

---

## Preparing Your Training Data

You have three options for making your training data available to jobs:

### Option 1: Upload to Hugging Face Dataset (Recommended)

```bash
# Install dataset tools
pip install datasets

# Create dataset structure
mkdir -p my_dataset/newspaper
# Add your images/, labels/, train.txt, val.txt, newspaper.yaml

# Create dataset
python create_hf_dataset.py
```

`create_hf_dataset.py`:
```python
from datasets import Dataset
from huggingface_hub import HfApi
import os

# Create dataset from your files
api = HfApi()

# Upload to HF Hub
api.upload_folder(
    folder_path="./my_dataset",
    repo_id="your-username/newspaper-layout-dataset",
    repo_type="dataset"
)

print("✓ Dataset uploaded!")
```

Then in your training script:
```python
from huggingface_hub import snapshot_download

data_path = snapshot_download(
    repo_id="your-username/newspaper-layout-dataset",
    repo_type="dataset",
    local_dir="/tmp/data"
)
```

### Option 2: Bundle with Training Script

For smaller datasets (<500MB), zip and include with job:

```bash
# Create zip
zip -r training_data.zip newspaper/

# Submit job with data
huggingface-cli jobs submit \
  --config job_config.yaml \
  --script train_job.py \
  --data training_data.zip
```

In script:
```python
import zipfile

# Extract bundled data
with zipfile.ZipFile("/job/data/training_data.zip", "r") as zip_ref:
    zip_ref.extractall("/tmp/data")
```

### Option 3: Use Job Storage

For large datasets, use HF Spaces storage (requires Space):

```bash
# Upload to Space storage
huggingface-cli upload your-space-name \
  training_data/ \
  --repo-type=space
```

---

## Creating a Training Job

### Basic Job Configuration

Create `job_config.yaml`:

```yaml
name: my-training-job

# Hardware configuration
compute:
  gpu: "t4"        # GPU type
  gpu_count: 1     # Number of GPUs
  cpu: 4           # CPU cores
  memory: "16GB"   # RAM

# Environment setup
environment:
  python_version: "3.10"
  requirements:
    - torch>=2.2.0
    - torchvision>=0.17.0
    - opencv-python>=4.6.0
    - huggingface_hub

# Script to run
script: train_job.py

# Optional settings
timeout: 14400  # 4 hours (in seconds)
retry_on_failure: true
max_retries: 2
```

### Submit the Job

```bash
# Basic submission
huggingface-cli jobs submit --config job_config.yaml

# With custom name
huggingface-cli jobs submit \
  --config job_config.yaml \
  --name "newspaper-layout-v1"

# With environment variables
huggingface-cli jobs submit \
  --config job_config.yaml \
  --env BATCH_SIZE=16 \
  --env EPOCHS=50
```

---

## Job Configuration Options

### Hardware Options

```yaml
compute:
  # GPU options (with approximate pricing per hour):
  gpu: "t4"      # ~$0.60/hr  - Good for small models, fine-tuning
  # gpu: "l4"    # ~$1.20/hr  - Better performance than T4
  # gpu: "a10g"  # ~$2.00/hr  - Mid-range, good balance
  # gpu: "a100"  # ~$4.00/hr  - High-end, large models

  gpu_count: 1   # 1, 2, 4, or 8
  cpu: 4         # 2, 4, 8, 16
  memory: "16GB" # "8GB", "16GB", "32GB", "64GB"
```

### Environment Options

```yaml
environment:
  python_version: "3.10"  # "3.8", "3.9", "3.10", "3.11", "3.12"

  # Install packages
  requirements:
    - torch>=2.2.0
    - git+https://github.com/nealcaren/DocLayout-YOLO.git

  # System packages (if needed)
  apt_packages:
    - ffmpeg
    - libsm6
```

### Execution Options

```yaml
# Maximum job runtime (seconds)
timeout: 3600  # 1 hour

# Retry on failure
retry_on_failure: true
max_retries: 3

# Notifications (optional)
notifications:
  email: your-email@example.com
  on_completion: true
  on_failure: true
```

---

## Monitoring Jobs

### List All Jobs

```bash
# List your jobs
huggingface-cli jobs list

# Filter by status
huggingface-cli jobs list --status running
huggingface-cli jobs list --status completed
```

### Check Job Status

```bash
# Get detailed status
huggingface-cli jobs status job-123abc

# Output:
# Job ID: job-123abc
# Status: running
# Progress: 45%
# GPU: t4 (1x)
# Runtime: 23m 15s
# Cost so far: $0.23
```

### View Logs

```bash
# Stream logs (follow mode)
huggingface-cli jobs logs job-123abc --follow

# View completed logs
huggingface-cli jobs logs job-123abc

# Save logs to file
huggingface-cli jobs logs job-123abc > training.log
```

### Cancel a Job

```bash
# Cancel running job
huggingface-cli jobs cancel job-123abc
```

---

## Retrieving Results

### Option 1: Upload to Hugging Face Hub (Recommended)

In your training script:

```python
from huggingface_hub import HfApi

def save_results():
    api = HfApi()

    # Upload trained model
    api.upload_folder(
        folder_path="/tmp/runs/train",
        repo_id="your-username/doclayout-newspaper",
        path_in_repo="training_results"
    )

    # Upload logs
    api.upload_file(
        path_or_fileobj="/tmp/training.log",
        repo_id="your-username/doclayout-newspaper",
        path_in_repo="logs/training.log"
    )
```

Then download:

```bash
# Download trained model
huggingface-cli download your-username/doclayout-newspaper

# Or use Python
from huggingface_hub import snapshot_download
local_dir = snapshot_download(
    repo_id="your-username/doclayout-newspaper"
)
```

### Option 2: Job Artifacts

```bash
# Download job artifacts (if configured)
huggingface-cli jobs artifacts job-123abc --output ./results
```

---

## Pricing & Hardware Options

### GPU Options (Approximate Pricing)

| GPU | VRAM | $/hour | Best For |
|-----|------|--------|----------|
| T4 | 16GB | $0.60 | Fine-tuning, small models |
| L4 | 24GB | $1.20 | Medium models, faster training |
| A10G | 24GB | $2.00 | Large models, production |
| A100 | 40GB/80GB | $4.00+ | Very large models, fastest |

### Cost Estimation

**Example: Fine-tuning for 2 hours on T4**
- Hardware: $0.60/hr × 2 hrs = $1.20
- Total: ~$1.20-1.50

**Example: Training from scratch for 8 hours on A10G**
- Hardware: $2.00/hr × 8 hrs = $16.00
- Total: ~$16.00-18.00

**Tips to Reduce Costs:**
- Use T4 for fine-tuning (usually sufficient)
- Set `timeout` to prevent runaway costs
- Use `patience` parameter for early stopping
- Test with small epochs first
- Enable `save_period` to save checkpoints frequently

### Set Spending Limits

```bash
# Set monthly spending limit
huggingface-cli billing set-limit 50.00  # $50/month
```

---

## Example Workflows

### Example 1: Quick Fine-tuning (T4, ~$2)

```yaml
# job_config.yaml
name: quick-finetune
compute:
  gpu: "t4"
  gpu_count: 1
  cpu: 4
  memory: "16GB"
environment:
  python_version: "3.10"
script: train_finetune.py
timeout: 7200  # 2 hours max
```

```python
# train_finetune.py
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10

# Download pre-trained model
model_path = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
)

# Quick fine-tune
model = YOLOv10(model_path)
model.train(
    data="newspaper.yaml",
    epochs=20,  # Just 20 epochs for quick results
    imgsz=1024,
    batch=8,
    lr0=0.001,
    device="0",
    patience=10,
    project="/tmp/runs"
)
```

Submit:
```bash
huggingface-cli jobs submit --config job_config.yaml
```

### Example 2: Full Training (A10G, ~$16)

```yaml
# job_config.yaml
name: full-training
compute:
  gpu: "a10g"
  gpu_count: 1
  cpu: 8
  memory: "32GB"
environment:
  python_version: "3.10"
script: train_full.py
timeout: 28800  # 8 hours max
```

```python
# train_full.py
from doclayout_yolo import YOLOv10

# Train from scratch
model = YOLOv10("yolov10n.yaml")
model.train(
    data="newspaper.yaml",
    epochs=100,
    imgsz=1024,
    batch=16,
    lr0=0.02,
    device="0",
    patience=30,
    project="/tmp/runs",
    save_period=10  # Save every 10 epochs
)
```

### Example 3: Multi-GPU Training (2x A100, ~$80)

```yaml
# job_config.yaml
name: multi-gpu-training
compute:
  gpu: "a100"
  gpu_count: 2
  cpu: 16
  memory: "64GB"
environment:
  python_version: "3.10"
script: train_multi_gpu.py
timeout: 36000  # 10 hours max
```

```python
# train_multi_gpu.py
from doclayout_yolo import YOLOv10

model = YOLOv10("yolov10m.yaml")
model.train(
    data="newspaper.yaml",
    epochs=100,
    imgsz=1600,
    batch=32,  # Larger batch with more GPUs
    device="0,1",  # Use both GPUs
    workers=8,
    project="/tmp/runs"
)
```

---

## Complete Example: End-to-End Workflow

Here's a complete workflow from data preparation to trained model:

### Step 1: Prepare Data

```bash
# Upload your dataset to HF Hub
huggingface-cli upload \
  your-username/newspaper-dataset \
  ./training_data/ \
  --repo-type=dataset
```

### Step 2: Create Training Script

`train_job.py`:
```python
#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download, HfApi, hf_hub_download

def main():
    print("=== DocLayout-YOLO Training Job ===")

    # 1. Download dataset
    print("Downloading dataset...")
    data_path = snapshot_download(
        repo_id="your-username/newspaper-dataset",
        repo_type="dataset",
        local_dir="/tmp/data"
    )

    # 2. Download pre-trained model
    print("Downloading pre-trained model...")
    model_path = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        local_dir="/tmp/pretrained"
    )

    # 3. Train
    print("Starting training...")
    from doclayout_yolo import YOLOv10

    model = YOLOv10(model_path)
    results = model.train(
        data=f"{data_path}/newspaper.yaml",
        epochs=50,
        imgsz=1024,
        batch=16,
        lr0=0.001,
        device="0",
        workers=4,
        val=True,
        plots=True,
        project="/tmp/runs",
        name="newspaper_model",
        patience=20,
        save_period=5
    )

    # 4. Upload results
    print("Uploading results...")
    api = HfApi()

    # Create model repo
    model_repo = f"{os.getenv('HF_USERNAME', 'user')}/doclayout-newspaper"
    api.create_repo(repo_id=model_repo, exist_ok=True)

    # Upload training results
    api.upload_folder(
        folder_path="/tmp/runs/newspaper_model",
        repo_id=model_repo,
        path_in_repo="training_results"
    )

    print(f"✓ Complete! Model at: https://huggingface.co/{model_repo}")

if __name__ == "__main__":
    # Install dependencies
    os.system("pip install git+https://github.com/nealcaren/DocLayout-YOLO.git -q")
    main()
```

### Step 3: Create Configuration

`job_config.yaml`:
```yaml
name: newspaper-layout-training

compute:
  gpu: "t4"
  gpu_count: 1
  cpu: 4
  memory: "16GB"

environment:
  python_version: "3.10"
  requirements:
    - torch>=2.2.0
    - torchvision>=0.17.0
    - huggingface_hub

script: train_job.py
timeout: 10800  # 3 hours
retry_on_failure: true
```

### Step 4: Submit and Monitor

```bash
# Submit job
JOB_ID=$(huggingface-cli jobs submit \
  --config job_config.yaml \
  --output json | jq -r '.job_id')

echo "Job ID: $JOB_ID"

# Monitor progress
huggingface-cli jobs logs $JOB_ID --follow

# Once complete, check cost
huggingface-cli jobs status $JOB_ID
```

### Step 5: Use Your Model

```python
from doclayout_yolo import YOLOv10

# Load your trained model
model = YOLOv10.from_pretrained("your-username/doclayout-newspaper")

# Run inference
results = model.predict("test_image.jpg", imgsz=1024, conf=0.2)
```

---

## Tips & Best Practices

### 1. Start Small

```bash
# Test with short job first
timeout: 600  # 10 minutes
epochs: 5     # Just a few epochs
```

### 2. Save Checkpoints Frequently

```python
model.train(
    ...,
    save_period=5,  # Save every 5 epochs
    patience=20     # Early stopping
)
```

### 3. Monitor Costs

```bash
# Check spending
huggingface-cli billing usage

# Set alerts
huggingface-cli billing set-limit 50.00
```

### 4. Use Appropriate Hardware

- **Fine-tuning**: T4 is usually sufficient
- **Training from scratch**: A10G or A100
- **Large models**: A100

### 5. Upload Results

Always upload results to HF Hub - don't rely on job storage:

```python
# At end of training
api.upload_folder(
    folder_path="/tmp/runs",
    repo_id="your-username/model-name"
)
```

---

## Troubleshooting

### Job Fails Immediately

Check logs for errors:
```bash
huggingface-cli jobs logs job-123abc
```

Common issues:
- Missing dependencies in `requirements`
- Invalid data paths
- Insufficient memory

### Out of Memory (OOM)

Reduce batch size or image size:
```python
model.train(
    batch=8,      # Reduce from 16
    imgsz=640     # Reduce from 1024
)
```

Or upgrade to larger GPU:
```yaml
compute:
  gpu: "a10g"  # More memory than T4
```

### Job Timeout

Increase timeout or reduce epochs:
```yaml
timeout: 14400  # 4 hours instead of 2
```

### Billing Issues

Check payment method:
```bash
huggingface-cli billing info
```

---

## Comparison with Other Options

| Feature | HF Jobs | Google Colab | Local |
|---------|---------|--------------|-------|
| Setup Time | Medium | Fast | Slow |
| Cost | Pay-per-use | Free (limited) | Hardware cost |
| GPU Options | Multiple | T4 only | Your hardware |
| Session Limits | None | 12 hours | None |
| Interruptions | None | Frequent | None |
| Integration | Excellent | Manual | Manual |
| Best For | Production training | Quick experiments | Full control |

---

## Additional Resources

- **HF Jobs Documentation**: https://huggingface.co/docs/huggingface_hub/guides/jobs
- **Pricing Calculator**: https://huggingface.co/pricing#jobs
- **Community Forum**: https://discuss.huggingface.co/
- **DocLayout-YOLO Repo**: https://github.com/nealcaren/DocLayout-YOLO

---

## Questions or Issues?

- Check job logs first: `huggingface-cli jobs logs JOB_ID`
- Review [HF Jobs docs](https://huggingface.co/docs/huggingface_hub/guides/jobs)
- Ask in [HF Community](https://discuss.huggingface.co/)
- See main [README.md](README.md) for training details

Happy training! 🚀

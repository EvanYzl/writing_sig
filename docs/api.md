# MSA-T OSV API Documentation

This document provides detailed API documentation for the MSA-T OSV framework.

## Table of Contents

- [Models](#models)
- [Data Loading](#data-loading)
- [Loss Functions](#loss-functions)
- [Metrics](#metrics)
- [Utilities](#utilities)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)

## Models

### MSATOSVModel

The main model class that combines CNN backbone, multi-scale attention, and Transformer.

```python
from msa_t_osv.models import MSATOSVModel

# Initialize model
model = MSATOSVModel(config)

# Forward pass
logits = model(images)  # images: torch.Tensor of shape (B, C, H, W)
```

**Parameters:**
- `config` (dict): Configuration dictionary containing model parameters

**Configuration Options:**
```yaml
model:
  backbone: "resnet50"  # CNN backbone architecture
  input_size: 224       # Input image size
  num_classes: 2        # Number of output classes
  attention:
    spatial: true       # Enable spatial attention
    channel: true       # Enable channel attention
    scale: true         # Enable scale attention
  transformer:
    num_layers: 6       # Number of transformer layers
    num_heads: 8        # Number of attention heads
    hidden_dim: 512     # Hidden dimension
  spp:
    levels: [1, 2, 4]   # SPP pyramid levels
```

### CNNBackbone

ResNet-based feature extractor with Spatial Pyramid Pooling.

```python
from msa_t_osv.models.backbone import CNNBackbone

backbone = CNNBackbone(
    backbone_name="resnet50",
    input_size=224,
    spp_levels=[1, 2, 4]
)

# Extract features
features = backbone(images)  # Returns multi-scale features
```

### MultiScaleAttention

Multi-scale attention module combining spatial, channel, and scale attention.

```python
from msa_t_osv.models.attention import MultiScaleAttention

attention = MultiScaleAttention(
    in_channels=2048,
    spatial=True,
    channel=True,
    scale=True
)

# Apply attention
attended_features = attention(features)
```

### TransformerEncoder

Transformer encoder for global feature modeling.

```python
from msa_t_osv.models.transformer import TransformerEncoder

transformer = TransformerEncoder(
    num_layers=6,
    num_heads=8,
    hidden_dim=512,
    dropout=0.1
)

# Encode features
encoded_features = transformer(features)
```

## Data Loading

### Dataset Classes

#### CEDARDataset

```python
from msa_t_osv.data.cedar import CEDARDataset

dataset = CEDARDataset(
    data_dir="/path/to/cedar",
    split="train",
    transform=transforms,
    augmentations=aug_config
)

# Get sample
sample = dataset[0]
# Returns: {"image": torch.Tensor, "label": int, "writer": int}
```

#### MCYTDataset

```python
from msa_t_osv.data.mcyt import MCYTDataset

dataset = MCYTDataset(
    data_dir="/path/to/mcyt",
    split="train",
    transform=transforms,
    augmentations=aug_config
)
```

#### GPDSDataset

```python
from msa_t_osv.data.gpds import GPDSDataset

dataset = GPDSDataset(
    data_dir="/path/to/gpds",
    split="train",
    transform=transforms,
    augmentations=aug_config
)
```

### Data Loader Factory

```python
from msa_t_osv.data import get_dataset

# Get dataset by name
dataset = get_dataset("cedar", config, split="train")

# Create dataloader
from torch.utils.data import DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

## Loss Functions

### CombinedLoss

Combines multiple loss functions for training.

```python
from msa_t_osv.losses import CombinedLoss

criterion = CombinedLoss(config)

# Compute loss
loss = criterion(logits, labels)
```

**Configuration:**
```yaml
loss:
  ce_weight: 1.0        # Cross-entropy weight
  triplet_weight: 0.5   # Triplet loss weight
  focal_weight: 0.3     # Focal loss weight
  label_smoothing: 0.1  # Label smoothing factor
  triplet_margin: 1.0   # Triplet loss margin
  focal_alpha: 1.0      # Focal loss alpha
  focal_gamma: 2.0      # Focal loss gamma
```

### Individual Loss Functions

```python
from msa_t_osv.losses import (
    CrossEntropyWithLabelSmoothing,
    TripletLoss,
    FocalLoss
)

# Cross-entropy with label smoothing
ce_loss = CrossEntropyWithLabelSmoothing(smoothing=0.1)
loss = ce_loss(logits, labels)

# Triplet loss
triplet_loss = TripletLoss(margin=1.0, mining="batch_hard")
loss = triplet_loss(features, labels)

# Focal loss
focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
loss = focal_loss(logits, labels)
```

## Metrics

### MetricTracker

Tracks and computes evaluation metrics.

```python
from msa_t_osv.metrics import MetricTracker

tracker = MetricTracker()

# Update with batch results
tracker.update(labels, scores)

# Compute final metrics
metrics = tracker.compute()
# Returns: {"eer": float, "accuracy": float, "auc": float, ...}
```

### Individual Metric Functions

```python
from msa_t_osv.metrics import (
    compute_eer,
    compute_far_frr,
    compute_accuracy,
    compute_auc,
    compute_writer_dependent_metrics
)

# Compute EER
eer = compute_eer(labels, scores)

# Compute FAR/FRR
far, frr = compute_far_frr(labels, scores, threshold=0.5)

# Compute accuracy
accuracy = compute_accuracy(labels, predictions)

# Compute AUC
auc = compute_auc(labels, scores)

# Compute writer-dependent metrics
writer_metrics = compute_writer_dependent_metrics(labels, scores, writers)
```

## Utilities

### Logger

```python
from msa_t_osv.utils.logger import setup_logger, get_logger

# Setup logger
logger = setup_logger(
    log_file="train.log",
    log_level="INFO"
)

# Get logger in modules
logger = get_logger(__name__)
logger.info("Training started")
```

### Seed Setting

```python
from msa_t_osv.utils.seed import set_seed

# Set random seed for reproducibility
set_seed(42, cuda_deterministic=True)
```

### Visualizer

```python
from msa_t_osv.utils.visualizer import Visualizer

visualizer = Visualizer(output_dir="visualizations")

# Plot ROC curve
visualizer.plot_roc_curve(
    labels, scores,
    save_path="roc_curve.png"
)

# Plot confusion matrix
visualizer.plot_confusion_matrix(
    labels, predictions,
    save_path="confusion_matrix.png"
)

# Plot score distributions
visualizer.plot_score_distributions(
    labels, scores,
    save_path="score_distributions.png"
)

# Plot t-SNE visualization
visualizer.plot_tsne(
    features, labels,
    save_path="tsne.png"
)
```

## Training

### Training Script

```bash
# Basic training
python -m msa_t_osv train --config configs/cedar.yaml

# With custom output directory
python -m msa_t_osv train --config configs/cedar.yaml --output_dir outputs/cedar

# Resume training
python -m msa_t_osv train --config configs/cedar.yaml --resume checkpoint.pth
```

### Training Configuration

```yaml
training:
  num_epochs: 100
  batch_size: 32
  num_workers: 4
  pin_memory: true
  use_amp: true          # Automatic mixed precision
  grad_clip: 1.0         # Gradient clipping
  use_ema: true          # Exponential moving average
  ema_decay: 0.9999
  save_freq: 10          # Save checkpoint every N epochs
  
  optimizer:
    type: "adamw"
    lr: 0.001
    weight_decay: 0.01
    betas: [0.9, 0.999]
    eps: 1e-8
    
  scheduler:
    type: "cosine"
    min_lr: 0.00001
```

## Evaluation

### Evaluation Script

```bash
# Basic evaluation
python -m msa_t_osv evaluate --config configs/cedar.yaml --checkpoint best.pth

# With custom output directory
python -m msa_t_osv evaluate --config configs/cedar.yaml --checkpoint best.pth --output_dir results/

# Skip visualizations
python -m msa_t_osv evaluate --config configs/cedar.yaml --checkpoint best.pth --no_vis
```

### Evaluation Configuration

```yaml
evaluation:
  batch_size: 64
  num_workers: 4
  pin_memory: true
  compute_writer_metrics: true
```

## Inference

### Inference Script

```bash
# Single image
python -m msa_t_osv inference --config configs/cedar.yaml --checkpoint best.pth --input signature.png

# Directory of images
python -m msa_t_osv inference --config configs/cedar.yaml --checkpoint best.pth --input signatures/ --output results.json

# With custom threshold
python -m msa_t_osv inference --config configs/cedar.yaml --checkpoint best.pth --input signature.png --threshold 0.3
```

### SignatureVerifier Class

```python
from msa_t_osv.inference import SignatureVerifier

# Initialize verifier
verifier = SignatureVerifier(
    checkpoint_path="best.pth",
    config=config,
    device="cuda"
)

# Verify single signature
result = verifier.verify_signature("signature.png")
print(f"Decision: {result['decision']}")
print(f"Score: {result['score']:.4f}")
print(f"Confidence: {result['confidence']:.4f}")

# Verify batch of signatures
results = verifier.verify_batch(["sig1.png", "sig2.png", "sig3.png"])
```

**Result Format:**
```python
{
    'decision': 'genuine',  # or 'forged'
    'is_genuine': True,
    'score': 0.1234,        # Forged probability
    'confidence': 0.8766,   # Prediction confidence
    'probabilities': {
        'genuine': 0.8766,
        'forged': 0.1234
    },
    'threshold': 0.5
}
```

### Inference Configuration

```yaml
inference:
  threshold: 0.5          # Decision threshold
  batch_size: 1           # Batch size for inference
```

## Configuration Files

### Complete Configuration Example

```yaml
# Global settings
seed: 42
cuda_deterministic: true
device: "cuda"

# Paths
paths:
  output_dir: "outputs"
  log_dir: "logs"
  checkpoint_dir: "checkpoints"
  vis_dir: "visualizations"

# Dataset configuration
dataset:
  name: "CEDAR"
  data_dir: "/path/to/cedar"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  augmentations:
    rotation: 10
    scale: [0.9, 1.1]
    brightness: 0.2
    contrast: 0.2
    horizontal_flip: false

# Data preprocessing
data:
  input_size: 224
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

# Model configuration
model:
  backbone: "resnet50"
  input_size: 224
  num_classes: 2
  attention:
    spatial: true
    channel: true
    scale: true
  transformer:
    num_layers: 6
    num_heads: 8
    hidden_dim: 512
    dropout: 0.1
  spp:
    levels: [1, 2, 4]

# Training configuration
training:
  num_epochs: 100
  batch_size: 32
  num_workers: 4
  pin_memory: true
  use_amp: true
  grad_clip: 1.0
  use_ema: true
  ema_decay: 0.9999
  save_freq: 10
  
  optimizer:
    type: "adamw"
    lr: 0.001
    weight_decay: 0.01
    betas: [0.9, 0.999]
    eps: 1e-8
    
  scheduler:
    type: "cosine"
    min_lr: 0.00001

# Loss configuration
loss:
  ce_weight: 1.0
  triplet_weight: 0.5
  focal_weight: 0.3
  label_smoothing: 0.1
  triplet_margin: 1.0
  focal_alpha: 1.0
  focal_gamma: 2.0

# Evaluation configuration
evaluation:
  batch_size: 64
  num_workers: 4
  pin_memory: true
  compute_writer_metrics: true

# Inference configuration
inference:
  threshold: 0.5
  batch_size: 1

# Logging configuration
logging:
  log_freq: 100
  tensorboard: true
```

## Error Handling

The framework includes comprehensive error handling:

```python
try:
    result = verifier.verify_signature("invalid_image.txt")
except FileNotFoundError:
    print("Image file not found")
except ValueError as e:
    print(f"Invalid image format: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Tips

1. **GPU Memory**: Use gradient accumulation for large models
2. **Data Loading**: Increase `num_workers` for faster data loading
3. **Mixed Precision**: Enable `use_amp` for faster training
4. **Batch Size**: Adjust based on available GPU memory
5. **Checkpointing**: Regular checkpointing prevents loss of progress

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or enable gradient accumulation
2. **Slow Training**: Check data loading bottleneck, increase `num_workers`
3. **Poor Performance**: Verify dataset paths and preprocessing
4. **Reproducibility**: Ensure `cuda_deterministic` is set correctly

### Debug Mode

Enable debug logging for detailed information:

```python
logger = setup_logger(log_level="DEBUG")
```

For more detailed information, see the individual module documentation and examples in the `examples/` directory. 
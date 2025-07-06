"""Basic usage example for MSA-T OSV."""

import torch
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from msa_t_osv.models import MSATOSVModel
from msa_t_osv.losses import CombinedLoss
from msa_t_osv.metrics import MetricTracker
from msa_t_osv.utils.logger import setup_logger


def main():
    """Basic usage example."""
    
    # Setup logging
    logger = setup_logger(log_level="INFO")
    logger.info("Starting basic usage example")
    
    # Sample configuration
    config = {
        "model": {
            "backbone": "resnet50",
            "input_size": 224,
            "num_classes": 2,
            "attention": {
                "spatial": True,
                "channel": True,
                "scale": True
            },
            "transformer": {
                "num_layers": 6,
                "num_heads": 8,
                "hidden_dim": 512,
                "dropout": 0.1
            },
            "spp": {
                "levels": [1, 2, 4]
            }
        },
        "loss": {
            "ce_weight": 1.0,
            "triplet_weight": 0.5,
            "focal_weight": 0.3,
            "label_smoothing": 0.1,
            "triplet_margin": 1.0,
            "focal_alpha": 1.0,
            "focal_gamma": 2.0
        }
    }
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = MSATOSVModel(config)
    model = model.to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create loss function
    criterion = CombinedLoss(config["loss"])
    logger.info("Loss function created")
    
    # Create metric tracker
    metric_tracker = MetricTracker()
    logger.info("Metric tracker created")
    
    # Sample data
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    labels = torch.randint(0, 2, (batch_size,)).to(device)
    
    logger.info(f"Sample batch: {images.shape}, labels: {labels.shape}")
    
    # Forward pass
    model.train()
    logits = model(images)
    logger.info(f"Model output shape: {logits.shape}")
    
    # Compute loss
    loss = criterion(logits, labels)
    logger.info(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    logger.info("Backward pass completed")
    
    # Evaluation mode
    model.eval()
    with torch.no_grad():
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        scores = probs[:, 1]  # Probability of being forged
        
        # Update metrics
        metric_tracker.update(labels, scores)
        
        # Compute metrics
        metrics = metric_tracker.compute()
        logger.info(f"Metrics: {metrics}")
    
    logger.info("Basic usage example completed successfully!")


if __name__ == "__main__":
    main() 
"""Advanced usage example for MSA-T OSV."""

import torch
import yaml
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from msa_t_osv.models import MSATOSVModel
from msa_t_osv.losses import CombinedLoss
from msa_t_osv.metrics import MetricTracker
from msa_t_osv.utils.logger import setup_logger
from msa_t_osv.utils.visualizer import Visualizer
from msa_t_osv.utils.seed import set_seed


def create_advanced_config():
    """Create an advanced configuration for demonstration."""
    return {
        "seed": 42,
        "cuda_deterministic": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        
        "paths": {
            "output_dir": "advanced_outputs",
            "log_dir": "advanced_logs",
            "checkpoint_dir": "advanced_checkpoints",
            "vis_dir": "advanced_visualizations"
        },
        
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
                "num_layers": 8,
                "num_heads": 12,
                "hidden_dim": 768,
                "dropout": 0.1
            },
            "spp": {
                "levels": [1, 2, 4, 8]
            }
        },
        
        "loss": {
            "ce_weight": 1.0,
            "triplet_weight": 0.7,
            "focal_weight": 0.5,
            "label_smoothing": 0.1,
            "triplet_margin": 1.2,
            "focal_alpha": 0.25,
            "focal_gamma": 3.0
        }
    }


def demonstrate_model_analysis(model, config, logger):
    """Demonstrate model analysis capabilities."""
    logger.info("=== Model Analysis ===")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Analyze parameter distribution
    param_groups = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            module_name = name.split('.')[0]
            if module_name not in param_groups:
                param_groups[module_name] = 0
            param_groups[module_name] += param.numel()
    
    logger.info("Parameter distribution by module:")
    for module, count in param_groups.items():
        percentage = (count / trainable_params) * 100
        logger.info(f"  {module}: {count:,} ({percentage:.1f}%)")
    
    # Memory usage estimation
    batch_size = 4
    input_size = config["model"]["input_size"]
    estimated_memory = batch_size * 3 * input_size * input_size * 4  # 4 bytes per float32
    logger.info(f"Estimated memory per batch: {estimated_memory / 1024 / 1024:.1f} MB")


def demonstrate_loss_analysis(criterion, config, logger):
    """Demonstrate loss function analysis."""
    logger.info("=== Loss Function Analysis ===")
    
    # Create sample data
    batch_size = 8
    logits = torch.randn(batch_size, 2)
    labels = torch.randint(0, 2, (batch_size,))
    features = torch.randn(batch_size, 512)
    
    # Compute individual losses
    ce_loss = criterion.ce_loss(logits, labels)
    triplet_loss = criterion.triplet_loss(features, labels)
    focal_loss = criterion.focal_loss(logits, labels)
    combined_loss = criterion(logits, labels, features)
    
    logger.info(f"Cross-entropy loss: {ce_loss.item():.4f}")
    logger.info(f"Triplet loss: {triplet_loss.item():.4f}")
    logger.info(f"Focal loss: {focal_loss.item():.4f}")
    logger.info(f"Combined loss: {combined_loss.item():.4f}")
    
    # Analyze loss weights
    weights = config["loss"]
    logger.info("Loss weights:")
    for name, weight in weights.items():
        if 'weight' in name:
            logger.info(f"  {name}: {weight}")


def demonstrate_metrics_analysis(logger):
    """Demonstrate metrics analysis capabilities."""
    logger.info("=== Metrics Analysis ===")
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic signature verification data
    genuine_scores = np.random.normal(0.2, 0.1, n_samples // 2)
    forged_scores = np.random.normal(0.8, 0.1, n_samples // 2)
    
    labels = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    scores = np.concatenate([genuine_scores, forged_scores])
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    labels = labels[indices]
    scores = scores[indices]
    
    # Create metric tracker
    tracker = MetricTracker()
    tracker.update(labels, scores)
    metrics = tracker.compute()
    
    logger.info("Computed metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    
    # Analyze different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    logger.info("Performance at different thresholds:")
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        logger.info(f"  Threshold {threshold}: Accuracy = {accuracy:.4f}")


def demonstrate_visualization(visualizer, logger):
    """Demonstrate visualization capabilities."""
    logger.info("=== Visualization Demo ===")
    
    # Create synthetic data for visualization
    np.random.seed(42)
    n_samples = 500
    
    # Generate data with clear separation
    genuine_scores = np.random.normal(0.2, 0.08, n_samples // 2)
    forged_scores = np.random.normal(0.8, 0.08, n_samples // 2)
    
    labels = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    scores = np.concatenate([genuine_scores, forged_scores])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    labels = labels[indices]
    scores = scores[indices]
    
    # Create visualizations
    output_dir = Path("advanced_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # ROC curve
        visualizer.plot_roc_curve(
            labels, scores,
            save_path=output_dir / "advanced_roc.png"
        )
        logger.info("Created ROC curve")
        
        # Score distributions
        visualizer.plot_score_distributions(
            labels, scores,
            save_path=output_dir / "advanced_distributions.png"
        )
        logger.info("Created score distributions")
        
        # Confusion matrix
        predictions = (scores >= 0.5).astype(int)
        visualizer.plot_confusion_matrix(
            labels, predictions,
            save_path=output_dir / "advanced_confusion.png"
        )
        logger.info("Created confusion matrix")
        
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")


def demonstrate_training_simulation(model, criterion, config, logger):
    """Demonstrate training simulation."""
    logger.info("=== Training Simulation ===")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-6
    )
    
    # Simulate training for a few steps
    model.train()
    losses = []
    learning_rates = []
    
    for step in range(10):
        # Generate synthetic batch
        batch_size = 4
        images = torch.randn(batch_size, 3, config["model"]["input_size"], config["model"]["input_size"])
        labels = torch.randint(0, 2, (batch_size,))
        
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        
        # Forward pass
        logits = model(images)
        
        # Get features for triplet loss (simplified)
        features = torch.randn(batch_size, 512)
        if torch.cuda.is_available():
            features = features.cuda()
        
        # Compute loss
        loss = criterion(logits, labels, features)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Record metrics
        losses.append(loss.item())
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        if step % 2 == 0:
            logger.info(f"Step {step}: Loss = {loss.item():.4f}, LR = {learning_rates[-1]:.6f}")
    
    logger.info(f"Average loss: {np.mean(losses):.4f}")
    logger.info(f"Final learning rate: {learning_rates[-1]:.6f}")


def main():
    """Advanced usage demonstration."""
    
    # Setup
    logger = setup_logger(log_level="INFO")
    logger.info("Starting advanced usage demonstration")
    
    # Set random seed
    set_seed(42, True)
    
    # Create advanced configuration
    config = create_advanced_config()
    logger.info(f"Using device: {config['device']}")
    
    # Create model
    model = MSATOSVModel(config)
    if torch.cuda.is_available():
        model = model.cuda()
    logger.info("Model created successfully")
    
    # Create loss function
    criterion = CombinedLoss(config["loss"])
    logger.info("Loss function created successfully")
    
    # Create visualizer
    visualizer = Visualizer(config["paths"]["vis_dir"])
    
    # Demonstrate various capabilities
    demonstrate_model_analysis(model, config, logger)
    demonstrate_loss_analysis(criterion, config, logger)
    demonstrate_metrics_analysis(logger)
    demonstrate_visualization(visualizer, logger)
    demonstrate_training_simulation(model, criterion, config, logger)
    
    logger.info("Advanced usage demonstration completed!")


if __name__ == "__main__":
    main() 
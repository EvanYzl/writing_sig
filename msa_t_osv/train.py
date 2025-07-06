"""Training script for MSA-T OSV."""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msa_t_osv.models import MSATOSVModel
from msa_t_osv.data import get_dataset
from msa_t_osv.losses import CombinedLoss
from msa_t_osv.metrics import MetricTracker, compute_all_metrics
from msa_t_osv.utils.logger import setup_logger, get_logger, MetricLogger, TensorBoardLogger
from msa_t_osv.utils.seed import set_seed
from msa_t_osv.utils.visualizer import Visualizer


class EMA:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """Initialize EMA.
        
        Args:
            model: Model to track
            decay: EMA decay rate
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self, model: nn.Module):
        """Update EMA parameters.
        
        Args:
            model: Current model
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self, model: nn.Module):
        """Apply EMA parameters to model.
        
        Args:
            model: Model to update
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self, model: nn.Module):
        """Restore original parameters.
        
        Args:
            model: Model to restore
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: Dict[str, Any]):
    """Create training and validation dataloaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get dataset name from config
    dataset_name = config.get("dataset", {}).get("name", "CEDAR").lower()
    
    # Create datasets
    train_dataset = get_dataset(dataset_name, config, split="train")
    val_dataset = get_dataset(dataset_name, config, split="val")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=config["evaluation"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
    )
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    logger: Any,
    metric_logger: MetricLogger,
    tensorboard_logger: TensorBoardLogger,
    ema: Optional[EMA] = None,
) -> Dict[str, float]:
    """Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch
        config: Configuration dictionary
        logger: Logger instance
        metric_logger: Metric logger
        tensorboard_logger: TensorBoard logger
        ema: Optional EMA instance
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    # Metric tracker
    metric_tracker = MetricTracker()
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        if config["training"]["use_amp"]:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config["training"]["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            
        optimizer.step()
        
        # Update EMA if enabled
        if ema is not None:
            ema.update(model)
            
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
            
        # Compute metrics
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            scores = probs[:, 1]  # Probability of being forged
            
            metric_tracker.update(labels, scores)
            
        # Update running statistics
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Log to TensorBoard
        if batch_idx % config["logging"]["log_freq"] == 0:
            step = epoch * len(train_loader) + batch_idx
            tensorboard_logger.log_scalar('train/loss', loss.item(), step)
            tensorboard_logger.log_scalar('train/lr', optimizer.param_groups[0]["lr"], step)
            
    # Compute final metrics
    metrics = metric_tracker.compute()
    avg_loss = total_loss / total_samples
    
    # Log metrics
    log_metrics = {
        'loss': avg_loss,
        **metrics
    }
    
    metric_logger.log(log_metrics, epoch * len(train_loader), epoch)
    
    for name, value in log_metrics.items():
        tensorboard_logger.log_scalar(f'train/{name}', value, epoch)
        
    return log_metrics


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    logger: Any,
    metric_logger: MetricLogger,
    tensorboard_logger: TensorBoardLogger,
    ema: Optional[EMA] = None,
) -> Dict[str, float]:
    """Validate for one epoch.
    
    Args:
        model: Model to validate
        val_loader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch
        config: Configuration dictionary
        logger: Logger instance
        metric_logger: Metric logger
        tensorboard_logger: TensorBoard logger
        ema: Optional EMA instance
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    # Metric tracker
    metric_tracker = MetricTracker()
    
    # Apply EMA if enabled
    if ema is not None:
        ema.apply_shadow(model)
        
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation {epoch}"):
            # Move data to device
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            if config["training"]["use_amp"]:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                
            # Compute metrics
            probs = F.softmax(logits, dim=1)
            scores = probs[:, 1]  # Probability of being forged
            
            metric_tracker.update(labels, scores)
            
            # Update running statistics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
    # Restore original parameters if EMA was used
    if ema is not None:
        ema.restore(model)
        
    # Compute final metrics
    metrics = metric_tracker.compute()
    avg_loss = total_loss / total_samples
    
    # Log metrics
    log_metrics = {
        'loss': avg_loss,
        **metrics
    }
    
    metric_logger.log(log_metrics, epoch * len(val_loader), epoch)
    
    for name, value in log_metrics.items():
        tensorboard_logger.log_scalar(f'val/{name}', value, epoch)
        
    return log_metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    checkpoint_dir: str,
    is_best: bool = False,
):
    """Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        metrics: Current metrics
        config: Configuration
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'best_eer.pth'
        torch.save(checkpoint, best_path)
        
    # Keep only recent checkpoints
    if config["training"]["save_freq"] > 0:
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > 5:  # Keep only 5 most recent
            for checkpoint_file in checkpoints[:-5]:
                checkpoint_file.unlink()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train MSA-T OSV model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set output directory
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
        
    # Set random seed
    set_seed(config["seed"], config["cuda_deterministic"])
    
    # Setup device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    
    # Setup logging
    log_dir = Path(config["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        log_file=log_dir / "train.log",
        log_level="INFO"
    )
    
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {config['paths']['output_dir']}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    model = MSATOSVModel(config)
    model = model.to(device)
    
    # Create loss function
    criterion = CombinedLoss(config["loss"])
    
    # Create optimizer
    optimizer_config = config["training"]["optimizer"]
    optimizer = AdamW(
        model.parameters(),
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"],
        betas=optimizer_config["betas"],
        eps=optimizer_config["eps"],
    )
    
    # Create scheduler
    scheduler_config = config["training"]["scheduler"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["num_epochs"],
        eta_min=scheduler_config["min_lr"],
    )
    
    # Create EMA if enabled
    ema = None
    if config["training"]["use_ema"]:
        ema = EMA(model, config["training"]["ema_decay"])
        
    # Create loggers
    metric_logger = MetricLogger(config["paths"]["log_dir"])
    tensorboard_logger = TensorBoardLogger(config["paths"]["log_dir"])
    
    # Create visualizer
    visualizer = Visualizer(config["paths"]["vis_dir"])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_eer = float('inf')
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_eer = checkpoint['metrics'].get('eer', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}")
        
    # Training loop
    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        logger.info(f"Epoch {epoch}/{config['training']['num_epochs']}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, config, logger, metric_logger, tensorboard_logger, ema
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch,
            config, logger, metric_logger, tensorboard_logger, ema
        )
        
        # Log epoch summary
        logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                   f"Val Loss: {val_metrics['loss']:.4f}, Val EER: {val_metrics['eer']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics['eer'] < best_eer
        if is_best:
            best_eer = val_metrics['eer']
            logger.info(f"New best EER: {best_eer:.4f}")
            
        if epoch % config["training"]["save_freq"] == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                config, config["paths"]["checkpoint_dir"], is_best
            )
            
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, config["training"]["num_epochs"] - 1,
        val_metrics, config, config["paths"]["checkpoint_dir"], False
    )
    
    # Save training summary
    metric_logger.save_summary()
    tensorboard_logger.close()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main() 
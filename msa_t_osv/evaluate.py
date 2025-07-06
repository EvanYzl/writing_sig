"""Evaluation script for MSA-T OSV."""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msa_t_osv.models import MSATOSVModel
from msa_t_osv.data import get_dataset
from msa_t_osv.metrics import MetricTracker, compute_all_metrics, compute_writer_dependent_metrics
from msa_t_osv.utils.logger import setup_logger, get_logger
from msa_t_osv.utils.seed import set_seed
from msa_t_osv.utils.visualizer import Visualizer


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


def load_model(checkpoint_path: str, config: Dict[str, Any], device: torch.device) -> MSATOSVModel:
    """Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # Create model
    model = MSATOSVModel(config)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def create_test_dataloader(config: Dict[str, Any]) -> DataLoader:
    """Create test dataloader.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Test dataloader
    """
    # Get dataset name from config
    dataset_name = config.get("dataset", {}).get("name", "CEDAR").lower()
    
    # Create test dataset
    test_dataset = get_dataset(dataset_name, config, split="test")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=config["evaluation"]["num_workers"],
        pin_memory=config["evaluation"]["pin_memory"],
    )
    
    return test_loader


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    logger: Any,
) -> Dict[str, Any]:
    """Evaluate model on test set.
    
    Args:
        model: Model to evaluate
        test_loader: Test dataloader
        device: Device to evaluate on
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Dictionary containing evaluation results
    """
    model.eval()
    
    # Lists to store predictions and ground truth
    all_scores = []
    all_labels = []
    all_writers = []
    all_predictions = []
    
    # Metric tracker
    metric_tracker = MetricTracker()
    
    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            writers = batch.get("writer", None)
            
            # Forward pass
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            scores = probs[:, 1]  # Probability of being forged
            predictions = torch.argmax(logits, dim=1)
            
            # Store results
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            
            if writers is not None:
                all_writers.extend(writers.cpu().numpy())
            
            # Update metric tracker
            metric_tracker.update(labels, scores)
    
    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_writers = np.array(all_writers) if all_writers else None
    
    # Compute metrics
    metrics = metric_tracker.compute()
    
    # Compute additional metrics
    additional_metrics = compute_all_metrics(all_labels, all_scores)
    metrics.update(additional_metrics)
    
    # Compute writer-dependent metrics if writer information is available
    if all_writers is not None:
        writer_metrics = compute_writer_dependent_metrics(all_labels, all_scores, all_writers)
        metrics.update(writer_metrics)
    
    # Store all results
    results = {
        'metrics': metrics,
        'scores': all_scores,
        'labels': all_labels,
        'predictions': all_predictions,
        'writers': all_writers,
    }
    
    return results


def save_results(results: Dict[str, Any], output_dir: str, config: Dict[str, Any]):
    """Save evaluation results.
    
    Args:
        results: Evaluation results
        output_dir: Output directory
        config: Configuration dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    metrics_file = output_dir / "evaluation_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    
    # Save raw results as numpy arrays
    np.save(output_dir / "scores.npy", results['scores'])
    np.save(output_dir / "labels.npy", results['labels'])
    np.save(output_dir / "predictions.npy", results['predictions'])
    if results['writers'] is not None:
        np.save(output_dir / "writers.npy", results['writers'])
    
    # Save configuration
    config_file = output_dir / "evaluation_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_visualizations(
    results: Dict[str, Any],
    output_dir: str,
    config: Dict[str, Any],
    visualizer: Visualizer,
):
    """Create evaluation visualizations.
    
    Args:
        results: Evaluation results
        output_dir: Output directory
        config: Configuration dictionary
        visualizer: Visualizer instance
    """
    output_dir = Path(output_dir)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot ROC curve
    roc_fig = visualizer.plot_roc_curve(
        results['labels'],
        results['scores'],
        save_path=vis_dir / "roc_curve.png"
    )
    
    # Plot confusion matrix
    cm_fig = visualizer.plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        save_path=vis_dir / "confusion_matrix.png"
    )
    
    # Plot score distributions
    score_fig = visualizer.plot_score_distributions(
        results['labels'],
        results['scores'],
        save_path=vis_dir / "score_distributions.png"
    )
    
    # Plot t-SNE if enough samples
    if len(results['scores']) > 100:
        try:
            tsne_fig = visualizer.plot_tsne(
                results['scores'].reshape(-1, 1),
                results['labels'],
                save_path=vis_dir / "tsne_visualization.png"
            )
        except Exception as e:
            print(f"Could not create t-SNE plot: {e}")
    
    # Plot writer-dependent metrics if available
    if results['writers'] is not None:
        writer_eer_fig = visualizer.plot_writer_metrics(
            results['writers'],
            results['labels'],
            results['scores'],
            save_path=vis_dir / "writer_metrics.png"
        )


def print_summary(results: Dict[str, Any], logger: Any):
    """Print evaluation summary.
    
    Args:
        results: Evaluation results
        logger: Logger instance
    """
    metrics = results['metrics']
    
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total samples: {len(results['labels'])}")
    logger.info(f"Genuine samples: {np.sum(results['labels'] == 0)}")
    logger.info(f"Forged samples: {np.sum(results['labels'] == 1)}")
    logger.info("")
    logger.info("PERFORMANCE METRICS:")
    logger.info(f"  EER: {metrics['eer']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"  AUC: {metrics['auc']:.4f}")
    logger.info("")
    logger.info("ERROR RATES:")
    logger.info(f"  FAR: {metrics['far']:.4f}")
    logger.info(f"  FRR: {metrics['frr']:.4f}")
    
    if 'writer_eer_mean' in metrics:
        logger.info("")
        logger.info("WRITER-DEPENDENT METRICS:")
        logger.info(f"  Mean Writer EER: {metrics['writer_eer_mean']:.4f}")
        logger.info(f"  Std Writer EER: {metrics['writer_eer_std']:.4f}")
        logger.info(f"  Min Writer EER: {metrics['writer_eer_min']:.4f}")
        logger.info(f"  Max Writer EER: {metrics['writer_eer_max']:.4f}")
    
    logger.info("=" * 50)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate MSA-T OSV model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--no_vis', action='store_true', help='Skip visualization')
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
        log_file=log_dir / "evaluate.log",
        log_level="INFO"
    )
    
    logger.info(f"Starting evaluation with config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {config['paths']['output_dir']}")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    logger.info("Model loaded successfully")
    
    # Create test dataloader
    test_loader = create_test_dataloader(config)
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create visualizer
    visualizer = Visualizer(config["paths"]["vis_dir"])
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device, config, logger)
    
    # Print summary
    print_summary(results, logger)
    
    # Save results
    save_results(results, config["paths"]["output_dir"], config)
    logger.info(f"Results saved to {config['paths']['output_dir']}")
    
    # Create visualizations
    if not args.no_vis:
        create_visualizations(results, config["paths"]["output_dir"], config, visualizer)
        logger.info("Visualizations created")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main() 
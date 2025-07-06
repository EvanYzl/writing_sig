"""Logging utilities for MSA-T OSV."""

import logging
import sys
import os
from typing import Optional, Dict, Any
from datetime import datetime
import json
from pathlib import Path


def setup_logger(
    name: str = "msa_t_osv",
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Setup logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Optional custom format string
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    if format_string is None:
        format_string = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file is not None:
        # Create log directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


def get_logger(name: str = "msa_t_osv") -> logging.Logger:
    """Get logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class MetricLogger:
    """Logger for tracking training metrics."""
    
    def __init__(self, log_dir: str, name: str = "metrics"):
        """Initialize metric logger.
        
        Args:
            log_dir: Directory to save logs
            name: Name for the log file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_file = self.log_dir / f"{name}_{timestamp}.json"
        self.csv_file = self.log_dir / f"{name}_{timestamp}.csv"
        
        self.metrics = []
        self.csv_header_written = False
        
    def log(self, metrics: Dict[str, Any], step: int, epoch: Optional[int] = None):
        """Log metrics for a step.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step/iteration
            epoch: Optional epoch number
        """
        # Add metadata
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }
        if epoch is not None:
            log_entry["epoch"] = epoch
            
        # Add metrics
        log_entry.update(metrics)
        
        # Store in memory
        self.metrics.append(log_entry)
        
        # Write to JSON file (append mode)
        with open(self.json_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        # Write to CSV file
        self._write_csv(log_entry)
        
    def _write_csv(self, log_entry: Dict[str, Any]):
        """Write metrics to CSV file.
        
        Args:
            log_entry: Dictionary containing metrics
        """
        import csv
        
        # Get all keys
        keys = list(log_entry.keys())
        
        # Write header if first time
        if not self.csv_header_written:
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
            self.csv_header_written = True
            
        # Write data
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writerow(log_entry)
            
    def get_best_metric(self, metric_name: str, mode: str = "min") -> Dict[str, Any]:
        """Get the best metric value and corresponding step.
        
        Args:
            metric_name: Name of the metric
            mode: "min" or "max"
            
        Returns:
            Dictionary with best value and step
        """
        if not self.metrics:
            return {}
            
        # Filter entries with the metric
        valid_entries = [m for m in self.metrics if metric_name in m]
        if not valid_entries:
            return {}
            
        # Find best
        if mode == "min":
            best_entry = min(valid_entries, key=lambda x: x[metric_name])
        else:
            best_entry = max(valid_entries, key=lambda x: x[metric_name])
            
        return {
            "value": best_entry[metric_name],
            "step": best_entry["step"],
            "epoch": best_entry.get("epoch"),
            "entry": best_entry,
        }
        
    def save_summary(self):
        """Save a summary of all logged metrics."""
        summary_file = self.log_dir / "summary.json"
        
        if not self.metrics:
            return
            
        # Compute summary statistics
        summary = {
            "total_steps": len(self.metrics),
            "metrics": {},
        }
        
        # Get all metric names
        metric_names = set()
        for entry in self.metrics:
            metric_names.update(k for k in entry.keys() 
                              if k not in ["step", "epoch", "timestamp"])
                              
        # Compute statistics for each metric
        for metric_name in metric_names:
            values = [m[metric_name] for m in self.metrics if metric_name in m]
            if values and all(isinstance(v, (int, float)) for v in values):
                import numpy as np
                summary["metrics"][metric_name] = {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "final": float(values[-1]),
                }
                
        # Save summary
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)


class TensorBoardLogger:
    """Wrapper for TensorBoard logging."""
    
    def __init__(self, log_dir: str, name: str = "default"):
        """Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            name: Run name
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(log_dir, name))
            self.enabled = True
        except ImportError:
            get_logger().warning("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
            self.enabled = False
            
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value.
        
        Args:
            tag: Metric name
            value: Metric value
            step: Current step
        """
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)
            
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars.
        
        Args:
            main_tag: Main tag for grouping
            tag_scalar_dict: Dictionary of tag -> value
            step: Current step
        """
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
            
    def log_image(self, tag: str, image: Any, step: int):
        """Log an image.
        
        Args:
            tag: Image tag
            image: Image tensor or numpy array
            step: Current step
        """
        if self.enabled and self.writer:
            self.writer.add_image(tag, image, step)
            
    def log_histogram(self, tag: str, values: Any, step: int):
        """Log a histogram.
        
        Args:
            tag: Histogram tag
            values: Values to plot
            step: Current step
        """
        if self.enabled and self.writer:
            self.writer.add_histogram(tag, values, step)
            
    def log_model_graph(self, model: Any, input_data: Any):
        """Log model graph.
        
        Args:
            model: PyTorch model
            input_data: Sample input for the model
        """
        if self.enabled and self.writer:
            try:
                self.writer.add_graph(model, input_data)
            except Exception as e:
                get_logger().warning(f"Failed to log model graph: {e}")
                
    def close(self):
        """Close the TensorBoard writer."""
        if self.enabled and self.writer:
            self.writer.close() 
"""Evaluation metrics for signature verification."""

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from typing import Tuple, Dict, List, Optional, Union
import warnings


def compute_eer(
    y_true: Union[np.ndarray, torch.Tensor],
    y_scores: Union[np.ndarray, torch.Tensor],
    pos_label: int = 1,
) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER).
    
    EER is the point where False Acceptance Rate equals False Rejection Rate.
    
    Args:
        y_true: True labels (0 for genuine, 1 for forged)
        y_scores: Prediction scores (higher means more likely to be forged)
        pos_label: Label of positive class
        
    Returns:
        Tuple of (EER value, threshold at EER)
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
        
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=pos_label)
    
    # Compute False Rejection Rate (FRR) = 1 - TPR
    frr = 1 - tpr
    
    # Find the point where FAR = FRR
    # We interpolate to find the exact EER
    eer_threshold = None
    eer = None
    
    # Find the points where the curves cross
    diff = np.abs(fpr - frr)
    min_index = np.argmin(diff)
    
    # Interpolate to find exact EER
    if min_index == 0:
        eer = fpr[min_index]
        eer_threshold = thresholds[min_index]
    else:
        # Linear interpolation between adjacent points
        if fpr[min_index] > frr[min_index]:
            # FAR is higher, so we interpolate with previous point
            if min_index > 0:
                x1, y1 = fpr[min_index - 1], frr[min_index - 1]
                x2, y2 = fpr[min_index], frr[min_index]
            else:
                x1, y1 = fpr[min_index], frr[min_index]
                x2, y2 = fpr[min_index + 1], frr[min_index + 1]
        else:
            # FRR is higher, so we interpolate with next point
            if min_index < len(fpr) - 1:
                x1, y1 = fpr[min_index], frr[min_index]
                x2, y2 = fpr[min_index + 1], frr[min_index + 1]
            else:
                x1, y1 = fpr[min_index - 1], frr[min_index - 1]
                x2, y2 = fpr[min_index], frr[min_index]
                
        # Find intersection point
        if x2 - y2 != x1 - y1:
            eer = (y1 - x1) / ((x2 - x1) - (y2 - y1)) * (x2 - x1) + x1
            
            # Interpolate threshold
            if min_index < len(thresholds) - 1:
                t1, t2 = thresholds[min_index], thresholds[min_index + 1]
                alpha = (eer - fpr[min_index]) / (fpr[min_index + 1] - fpr[min_index])
                eer_threshold = t1 + alpha * (t2 - t1)
            else:
                eer_threshold = thresholds[min_index]
        else:
            eer = diff[min_index]
            eer_threshold = thresholds[min_index]
            
    return float(eer), float(eer_threshold)


def compute_far(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute False Acceptance Rate (FAR).
    
    FAR = Number of forged signatures accepted as genuine / Total number of forged signatures
    
    Args:
        y_true: True labels (0 for genuine, 1 for forged)
        y_pred: Predicted labels (0 for genuine, 1 for forged)
        
    Returns:
        FAR value
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    # Find forged signatures (label = 1)
    forged_mask = y_true == 1
    
    if forged_mask.sum() == 0:
        warnings.warn("No forged signatures in the dataset")
        return 0.0
        
    # Count forged signatures incorrectly classified as genuine
    false_acceptances = ((y_pred == 0) & forged_mask).sum()
    
    # Compute FAR
    far = false_acceptances / forged_mask.sum()
    
    return float(far)


def compute_frr(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute False Rejection Rate (FRR).
    
    FRR = Number of genuine signatures rejected / Total number of genuine signatures
    
    Args:
        y_true: True labels (0 for genuine, 1 for forged)
        y_pred: Predicted labels (0 for genuine, 1 for forged)
        
    Returns:
        FRR value
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    # Find genuine signatures (label = 0)
    genuine_mask = y_true == 0
    
    if genuine_mask.sum() == 0:
        warnings.warn("No genuine signatures in the dataset")
        return 0.0
        
    # Count genuine signatures incorrectly classified as forged
    false_rejections = ((y_pred == 1) & genuine_mask).sum()
    
    # Compute FRR
    frr = false_rejections / genuine_mask.sum()
    
    return float(frr)


def compute_accuracy(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute classification accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy value
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    return float(accuracy_score(y_true, y_pred))


def compute_all_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_scores: Union[np.ndarray, torch.Tensor],
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Compute all metrics for signature verification.
    
    Args:
        y_true: True labels (0 for genuine, 1 for forged)
        y_scores: Prediction scores (higher means more likely to be forged)
        threshold: Optional threshold for classification. If None, use EER threshold
        
    Returns:
        Dictionary containing all metrics
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
        
    # Compute EER and threshold
    eer, eer_threshold = compute_eer(y_true, y_scores)
    
    # Use provided threshold or EER threshold
    if threshold is None:
        threshold = eer_threshold
        
    # Convert scores to predictions using threshold
    y_pred = (y_scores >= threshold).astype(int)
    
    # Compute other metrics
    far = compute_far(y_true, y_pred)
    frr = compute_frr(y_true, y_pred)
    acc = compute_accuracy(y_true, y_pred)
    
    # Compute AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "far": far,
        "frr": frr,
        "accuracy": acc,
        "auc": roc_auc,
        "threshold": threshold,
    }


def compute_writer_dependent_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_scores: Union[np.ndarray, torch.Tensor],
    writer_ids: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """Compute writer-dependent metrics.
    
    This computes metrics for each writer separately and then averages them.
    
    Args:
        y_true: True labels (0 for genuine, 1 for forged)
        y_scores: Prediction scores
        writer_ids: Writer/user IDs for each sample
        
    Returns:
        Dictionary containing writer-dependent metrics
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    if isinstance(writer_ids, torch.Tensor):
        writer_ids = writer_ids.cpu().numpy()
        
    unique_writers = np.unique(writer_ids)
    writer_metrics = []
    
    for writer_id in unique_writers:
        # Get samples for this writer
        writer_mask = writer_ids == writer_id
        writer_y_true = y_true[writer_mask]
        writer_y_scores = y_scores[writer_mask]
        
        # Skip if writer has no samples or only one class
        if len(writer_y_true) == 0 or len(np.unique(writer_y_true)) < 2:
            continue
            
        # Compute metrics for this writer
        try:
            writer_metric = compute_all_metrics(writer_y_true, writer_y_scores)
            writer_metrics.append(writer_metric)
        except Exception as e:
            warnings.warn(f"Could not compute metrics for writer {writer_id}: {e}")
            continue
            
    # Average metrics across writers
    if not writer_metrics:
        warnings.warn("No valid writer metrics computed")
        return {}
        
    avg_metrics = {}
    for key in writer_metrics[0].keys():
        values = [m[key] for m in writer_metrics]
        avg_metrics[f"writer_dependent_{key}"] = np.mean(values)
        avg_metrics[f"writer_dependent_{key}_std"] = np.std(values)
        
    return avg_metrics


class MetricTracker:
    """Class to track metrics during training/validation."""
    
    def __init__(self, metrics: List[str] = ["eer", "far", "frr", "accuracy"]):
        """Initialize metric tracker.
        
        Args:
            metrics: List of metrics to track
        """
        self.metrics = metrics
        self.reset()
        
    def reset(self):
        """Reset all tracked values."""
        self.y_true = []
        self.y_scores = []
        self.writer_ids = []
        
    def update(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_scores: Union[np.ndarray, torch.Tensor],
        writer_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """Update with new predictions.
        
        Args:
            y_true: True labels
            y_scores: Prediction scores
            writer_ids: Optional writer IDs
        """
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_scores, torch.Tensor):
            y_scores = y_scores.cpu().numpy()
        if writer_ids is not None and isinstance(writer_ids, torch.Tensor):
            writer_ids = writer_ids.cpu().numpy()
            
        self.y_true.extend(y_true)
        self.y_scores.extend(y_scores)
        
        if writer_ids is not None:
            self.writer_ids.extend(writer_ids)
            
    def compute(self) -> Dict[str, float]:
        """Compute all tracked metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        if not self.y_true:
            warnings.warn("No data to compute metrics")
            return {}
            
        y_true = np.array(self.y_true)
        y_scores = np.array(self.y_scores)
        
        # Compute standard metrics
        metrics = compute_all_metrics(y_true, y_scores)
        
        # Compute writer-dependent metrics if writer IDs available
        if self.writer_ids:
            writer_ids = np.array(self.writer_ids)
            writer_metrics = compute_writer_dependent_metrics(y_true, y_scores, writer_ids)
            metrics.update(writer_metrics)
            
        return metrics 
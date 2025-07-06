"""Visualization utilities for MSA-T OSV."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import cv2
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


def plot_roc_curve(
    y_true: Union[np.ndarray, torch.Tensor],
    y_scores: Union[np.ndarray, torch.Tensor],
    save_path: Optional[str] = None,
    title: str = "ROC Curve",
    show_eer: bool = True,
) -> Dict[str, float]:
    """Plot ROC curve.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        save_path: Optional path to save the plot
        title: Plot title
        show_eer: Whether to show EER point
        
    Returns:
        Dictionary with AUC and EER values
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
        
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    # Show EER if requested
    eer_value = None
    if show_eer:
        from ..metrics import compute_eer
        eer_value, eer_threshold = compute_eer(y_true, y_scores)
        
        # Find EER point on curve
        eer_fpr = eer_value
        eer_tpr = 1 - eer_value
        
        plt.scatter([eer_fpr], [eer_tpr], color='red', s=100, 
                   label=f'EER = {eer_value:.3f}', zorder=5)
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {"auc": roc_auc, "eer": eer_value}


def plot_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    class_names: List[str] = ["Genuine", "Forged"],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> np.ndarray:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Optional path to save the plot
        title: Plot title
        normalize: Whether to normalize the matrix
        
    Returns:
        Confusion matrix
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
        
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm


class Visualizer:
    """Main visualization class for MSA-T OSV."""
    
    def __init__(self, save_dir: str = "./visualizations"):
        """Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def visualize_attention(
        self,
        image: torch.Tensor,
        attention_map: torch.Tensor,
        save_name: str = "attention.png",
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Visualize attention map overlaid on image.
        
        Args:
            image: Input image tensor (C, H, W)
            attention_map: Attention map (H', W')
            save_name: Name for saved file
            alpha: Overlay transparency
            
        Returns:
            Visualization as numpy array
        """
        # Convert image to numpy
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.shape[0] == 3:  # CHW to HWC
                image = np.transpose(image, (1, 2, 0))
                
        # Normalize image to [0, 1]
        if image.max() > 1:
            image = image / 255.0
            
        # Convert attention map to numpy
        if isinstance(attention_map, torch.Tensor):
            attention_map = attention_map.cpu().numpy()
            
        # Resize attention map to match image size
        if attention_map.shape != image.shape[:2]:
            attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
            
        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Create heatmap
        heatmap = plt.get_cmap('jet')(attention_map)[:, :, :3]
        
        # Overlay
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
            
        overlay = (1 - alpha) * image + alpha * heatmap
        overlay = np.clip(overlay, 0, 1)
        
        # Save
        save_path = self.save_dir / save_name
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(attention_map, cmap='jet')
        plt.title("Attention Map")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return overlay
        
    def visualize_features_tsne(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        save_name: str = "tsne.png",
        perplexity: int = 30,
        n_iter: int = 1000,
    ) -> np.ndarray:
        """Visualize features using t-SNE.
        
        Args:
            features: Feature tensor (N, D)
            labels: Label tensor (N,)
            save_name: Name for saved file
            perplexity: t-SNE perplexity
            n_iter: Number of iterations
            
        Returns:
            t-SNE embeddings
        """
        # Convert to numpy
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
            
        # Compute t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        embeddings = tsne.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                            c=labels, cmap='coolwarm', alpha=0.7, s=50)
        plt.colorbar(scatter, label='Class')
        plt.title('t-SNE Visualization of Features')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return embeddings
        
    def visualize_grad_cam(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        target_layer: torch.nn.Module,
        class_idx: Optional[int] = None,
        save_name: str = "gradcam.png",
    ) -> np.ndarray:
        """Visualize Grad-CAM.
        
        Args:
            model: PyTorch model
            image: Input image tensor (1, C, H, W)
            target_layer: Target layer for Grad-CAM
            class_idx: Target class index
            save_name: Name for saved file
            
        Returns:
            Grad-CAM heatmap
        """
        # Hook to capture gradients and activations
        gradients = []
        activations = []
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
            
        def forward_hook(module, input, output):
            activations.append(output)
            
        # Register hooks
        backward_handle = target_layer.register_backward_hook(backward_hook)
        forward_handle = target_layer.register_forward_hook(forward_hook)
        
        # Forward pass
        model.eval()
        output = model(image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        # Backward pass
        model.zero_grad()
        output[0, class_idx].backward()
        
        # Get gradients and activations
        grad = gradients[0].cpu().data.numpy()[0]
        act = activations[0].cpu().data.numpy()[0]
        
        # Compute Grad-CAM
        weights = np.mean(grad, axis=(1, 2))
        cam = np.zeros(act.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * act[i]
            
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        # Remove hooks
        backward_handle.remove()
        forward_handle.remove()
        
        # Visualize
        image_np = image[0].cpu().numpy()
        if image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))
            
        overlay = self.visualize_attention(
            torch.tensor(image_np), 
            torch.tensor(cam),
            save_name=save_name
        )
        
        return cam
        
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_name: str = "training_history.png",
    ):
        """Plot training history.
        
        Args:
            history: Dictionary of metric lists
            save_name: Name for saved file
        """
        n_metrics = len(history)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
            
        for ax, (metric_name, values) in zip(axes, history.items()):
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} over epochs')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_metric_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_name: str = "metric_comparison.png",
    ):
        """Plot comparison of metrics across different models/datasets.
        
        Args:
            metrics_dict: Dictionary of model/dataset -> metrics
            save_name: Name for saved file
        """
        # Extract metric names
        metric_names = list(next(iter(metrics_dict.values())).keys())
        n_models = len(metrics_dict)
        n_metrics = len(metric_names)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(n_metrics)
        width = 0.8 / n_models
        
        for i, (model_name, metrics) in enumerate(metrics_dict.items()):
            values = [metrics[m] for m in metric_names]
            ax.bar(x + i * width, values, width, label=model_name)
            
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Metric Comparison')
        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 
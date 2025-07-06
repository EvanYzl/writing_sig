"""Loss functions for MSA-T OSV."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1):
        """Initialize label smoothing cross entropy.
        
        Args:
            smoothing: Label smoothing factor
        """
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing cross entropy.
        
        Args:
            pred: Predictions of shape (B, num_classes)
            target: Ground truth labels of shape (B,)
            
        Returns:
            Loss value
        """
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            # Create smoothed target distribution
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class TripletLoss(nn.Module):
    """Triplet loss for metric learning."""
    
    def __init__(self, margin: float = 0.2, p: int = 2, mining: str = "batch_hard"):
        """Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
            p: Norm degree for distance calculation
            mining: Mining strategy ('batch_hard', 'batch_all', 'none')
        """
        super().__init__()
        self.margin = margin
        self.p = p
        self.mining = mining
        
    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between embeddings.
        
        Args:
            embeddings: Embeddings of shape (B, D)
            
        Returns:
            Pairwise distances of shape (B, B)
        """
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        
        # Compute pairwise distances
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = F.relu(distances)  # Ensure non-negative
        
        # Add epsilon for numerical stability
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16
        
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)
        
        return distances
        
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        anchors: Optional[torch.Tensor] = None,
        positives: Optional[torch.Tensor] = None,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute triplet loss.
        
        Args:
            embeddings: Embeddings of shape (B, D)
            labels: Labels of shape (B,)
            anchors: Optional pre-selected anchor embeddings
            positives: Optional pre-selected positive embeddings
            negatives: Optional pre-selected negative embeddings
            
        Returns:
            Loss value
        """
        if anchors is not None and positives is not None and negatives is not None:
            # Use pre-selected triplets
            ap_distances = F.pairwise_distance(anchors, positives, p=self.p)
            an_distances = F.pairwise_distance(anchors, negatives, p=self.p)
            losses = F.relu(ap_distances - an_distances + self.margin)
            return losses.mean()
            
        # Compute pairwise distances
        distances = self._pairwise_distances(embeddings)
        
        if self.mining == "batch_hard":
            return self._batch_hard_triplet_loss(distances, labels)
        elif self.mining == "batch_all":
            return self._batch_all_triplet_loss(distances, labels)
        else:
            raise ValueError(f"Unknown mining strategy: {self.mining}")
            
    def _batch_hard_triplet_loss(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Batch hard triplet loss.
        
        For each anchor, select the hardest positive and negative.
        
        Args:
            distances: Pairwise distances of shape (B, B)
            labels: Labels of shape (B,)
            
        Returns:
            Loss value
        """
        batch_size = distances.size(0)
        
        # Create masks for valid positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # For each anchor, find hardest positive (largest distance)
        anchor_positive_dist = distances.unsqueeze(2)
        mask = labels_equal.unsqueeze(2)
        anchor_positive_dist = anchor_positive_dist * mask.float()
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)
        
        # For each anchor, find hardest negative (smallest distance)
        anchor_negative_dist = distances.unsqueeze(1)
        mask = labels_not_equal.unsqueeze(1)
        
        # Add max value to invalid positions
        max_val = distances.max()
        anchor_negative_dist = anchor_negative_dist + (~mask).float() * max_val
        hardest_negative_dist, _ = anchor_negative_dist.min(2, keepdim=True)
        
        # Compute triplet loss
        triplet_loss = hardest_positive_dist - hardest_negative_dist + self.margin
        triplet_loss = F.relu(triplet_loss)
        
        # Count valid triplets
        valid_triplets = (triplet_loss > 1e-16).float()
        num_positive_triplets = valid_triplets.sum()
        
        if num_positive_triplets == 0:
            return torch.tensor(0.0, device=distances.device)
            
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        
        return triplet_loss
        
    def _batch_all_triplet_loss(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Batch all triplet loss.
        
        Consider all valid triplets.
        
        Args:
            distances: Pairwise distances of shape (B, B)
            labels: Labels of shape (B,)
            
        Returns:
            Loss value
        """
        batch_size = distances.size(0)
        
        # Create masks for valid triplets
        indices_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        indices_not_equal = ~indices_equal
        i_not_equal_j = ~torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        i_not_equal_k = i_not_equal_j
        distinct_indices = i_not_equal_j.unsqueeze(2) & i_not_equal_k.unsqueeze(1)
        
        label_equal = indices_equal.unsqueeze(2) & indices_not_equal.unsqueeze(1)
        valid_triplets = distinct_indices & label_equal
        
        # Compute all anchor-positive and anchor-negative distances
        anchor_positive = distances.unsqueeze(2)
        anchor_negative = distances.unsqueeze(1)
        
        # Compute triplet loss
        triplet_loss = anchor_positive - anchor_negative + self.margin
        triplet_loss = triplet_loss * valid_triplets.float()
        triplet_loss = F.relu(triplet_loss)
        
        # Count valid triplets
        num_positive_triplets = (triplet_loss > 1e-16).float().sum()
        
        if num_positive_triplets == 0:
            return torch.tensor(0.0, device=distances.device)
            
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        
        return triplet_loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """Initialize focal loss.
        
        Args:
            alpha: Weighting factor for class imbalance
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            pred: Predictions of shape (B, num_classes)
            target: Ground truth labels of shape (B,)
            
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Apply focal term
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            if self.alpha >= 0:
                alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
                focal_loss = alpha_t * focal_loss
                
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss function for MSA-T OSV."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize combined loss.
        
        Args:
            config: Loss configuration containing:
                - ce_weight: Weight for cross entropy loss
                - triplet_weight: Weight for triplet loss
                - triplet_margin: Margin for triplet loss
                - label_smoothing: Label smoothing factor
                - use_focal_loss: Whether to use focal loss
                - focal_gamma: Gamma for focal loss
                - focal_alpha: Alpha for focal loss
        """
        super().__init__()
        self.config = config
        
        # Loss weights
        self.ce_weight = config.get("ce_weight", 1.0)
        self.triplet_weight = config.get("triplet_weight", 0.1)
        
        # Cross entropy loss
        if config.get("use_focal_loss", False):
            self.ce_loss = FocalLoss(
                alpha=config.get("focal_alpha", 0.25),
                gamma=config.get("focal_gamma", 2.0),
            )
        else:
            label_smoothing = config.get("label_smoothing", 0.1)
            if label_smoothing > 0:
                self.ce_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
            else:
                self.ce_loss = nn.CrossEntropyLoss()
                
        # Triplet loss
        self.triplet_loss = TripletLoss(
            margin=config.get("triplet_margin", 0.2),
            mining="batch_hard",
        )
        
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            logits: Classification logits of shape (B, num_classes)
            labels: Ground truth labels of shape (B,)
            embeddings: Optional embeddings for triplet loss of shape (B, D)
            return_components: If True, return individual loss components
            
        Returns:
            Total loss or dictionary of loss components
        """
        losses = {}
        
        # Cross entropy loss
        ce_loss = self.ce_loss(logits, labels)
        losses["ce_loss"] = ce_loss
        
        # Triplet loss if embeddings provided
        if embeddings is not None and self.triplet_weight > 0:
            triplet_loss = self.triplet_loss(embeddings, labels)
            losses["triplet_loss"] = triplet_loss
        else:
            losses["triplet_loss"] = torch.tensor(0.0, device=logits.device)
            
        # Combined loss
        total_loss = self.ce_weight * losses["ce_loss"] + self.triplet_weight * losses["triplet_loss"]
        losses["total_loss"] = total_loss
        
        if return_components:
            return losses
        else:
            return total_loss 
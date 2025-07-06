"""Tests for loss functions."""

import pytest
import torch
import torch.nn.functional as F
from msa_t_osv.losses import (
    CrossEntropyWithLabelSmoothing,
    TripletLoss,
    FocalLoss,
    CombinedLoss
)


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    batch_size = 8
    num_classes = 2
    feature_dim = 512
    
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    features = torch.randn(batch_size, feature_dim)
    
    return logits, labels, features


@pytest.fixture
def loss_config():
    """Sample loss configuration."""
    return {
        "ce_weight": 1.0,
        "triplet_weight": 0.5,
        "focal_weight": 0.3,
        "label_smoothing": 0.1,
        "triplet_margin": 1.0,
        "focal_alpha": 1.0,
        "focal_gamma": 2.0
    }


def test_cross_entropy_with_label_smoothing(sample_data):
    """Test cross-entropy loss with label smoothing."""
    logits, labels, _ = sample_data
    
    criterion = CrossEntropyWithLabelSmoothing(smoothing=0.1)
    loss = criterion(logits, labels)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_cross_entropy_without_smoothing(sample_data):
    """Test cross-entropy loss without label smoothing."""
    logits, labels, _ = sample_data
    
    criterion = CrossEntropyWithLabelSmoothing(smoothing=0.0)
    loss = criterion(logits, labels)
    
    # Should be equivalent to standard cross-entropy
    expected_loss = F.cross_entropy(logits, labels)
    
    assert torch.allclose(loss, expected_loss, atol=1e-6)


def test_triplet_loss(sample_data):
    """Test triplet loss."""
    _, labels, features = sample_data
    
    criterion = TripletLoss(margin=1.0, mining="batch_hard")
    loss = criterion(features, labels)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_triplet_loss_different_margins(sample_data):
    """Test triplet loss with different margins."""
    _, labels, features = sample_data
    
    margins = [0.5, 1.0, 2.0]
    losses = []
    
    for margin in margins:
        criterion = TripletLoss(margin=margin, mining="batch_hard")
        loss = criterion(features, labels)
        losses.append(loss.item())
    
    # Losses should be different for different margins
    assert len(set(losses)) > 1


def test_triplet_loss_mining_strategies(sample_data):
    """Test triplet loss with different mining strategies."""
    _, labels, features = sample_data
    
    strategies = ["batch_hard", "batch_all"]
    losses = []
    
    for strategy in strategies:
        criterion = TripletLoss(margin=1.0, mining=strategy)
        loss = criterion(features, labels)
        losses.append(loss.item())
    
    # All strategies should produce valid losses
    assert all(l >= 0 for l in losses)


def test_focal_loss(sample_data):
    """Test focal loss."""
    logits, labels, _ = sample_data
    
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    loss = criterion(logits, labels)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_focal_loss_different_gamma(sample_data):
    """Test focal loss with different gamma values."""
    logits, labels, _ = sample_data
    
    gammas = [0.0, 1.0, 2.0, 5.0]
    losses = []
    
    for gamma in gammas:
        criterion = FocalLoss(alpha=1.0, gamma=gamma)
        loss = criterion(logits, labels)
        losses.append(loss.item())
    
    # Losses should be different for different gamma values
    assert len(set(losses)) > 1


def test_focal_loss_alpha(sample_data):
    """Test focal loss with different alpha values."""
    logits, labels, _ = sample_data
    
    alphas = [0.25, 0.5, 1.0]
    losses = []
    
    for alpha in alphas:
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        loss = criterion(logits, labels)
        losses.append(loss.item())
    
    # Losses should be different for different alpha values
    assert len(set(losses)) > 1


def test_combined_loss(sample_data, loss_config):
    """Test combined loss function."""
    logits, labels, features = sample_data
    
    criterion = CombinedLoss(loss_config)
    loss = criterion(logits, labels, features)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_combined_loss_weights(sample_data):
    """Test combined loss with different weight configurations."""
    logits, labels, features = sample_data
    
    # Test with only cross-entropy
    config1 = {
        "ce_weight": 1.0,
        "triplet_weight": 0.0,
        "focal_weight": 0.0,
        "label_smoothing": 0.1,
        "triplet_margin": 1.0,
        "focal_alpha": 1.0,
        "focal_gamma": 2.0
    }
    
    # Test with only triplet loss
    config2 = {
        "ce_weight": 0.0,
        "triplet_weight": 1.0,
        "focal_weight": 0.0,
        "label_smoothing": 0.1,
        "triplet_margin": 1.0,
        "focal_alpha": 1.0,
        "focal_gamma": 2.0
    }
    
    criterion1 = CombinedLoss(config1)
    criterion2 = CombinedLoss(config2)
    
    loss1 = criterion1(logits, labels, features)
    loss2 = criterion2(logits, labels, features)
    
    # Losses should be different
    assert not torch.allclose(loss1, loss2, atol=1e-6)


def test_combined_loss_gradient_flow(sample_data, loss_config):
    """Test gradient flow through combined loss."""
    logits, labels, features = sample_data
    
    # Make inputs require gradients
    logits.requires_grad_(True)
    features.requires_grad_(True)
    
    criterion = CombinedLoss(loss_config)
    loss = criterion(logits, labels, features)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    assert logits.grad is not None
    assert features.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert not torch.isnan(features.grad).any()


def test_loss_consistency(sample_data, loss_config):
    """Test loss consistency across multiple calls."""
    logits, labels, features = sample_data
    
    criterion = CombinedLoss(loss_config)
    
    # Multiple forward passes should produce consistent results
    loss1 = criterion(logits, labels, features)
    loss2 = criterion(logits, labels, features)
    
    assert torch.allclose(loss1, loss2, atol=1e-6)


def test_loss_with_different_batch_sizes(loss_config):
    """Test loss functions with different batch sizes."""
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        logits = torch.randn(batch_size, 2)
        labels = torch.randint(0, 2, (batch_size,))
        features = torch.randn(batch_size, 512)
        
        criterion = CombinedLoss(loss_config)
        loss = criterion(logits, labels, features)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)


def test_loss_edge_cases():
    """Test loss functions with edge cases."""
    # Single sample
    logits = torch.randn(1, 2)
    labels = torch.randint(0, 2, (1,))
    features = torch.randn(1, 512)
    
    config = {
        "ce_weight": 1.0,
        "triplet_weight": 0.5,
        "focal_weight": 0.3,
        "label_smoothing": 0.1,
        "triplet_margin": 1.0,
        "focal_alpha": 1.0,
        "focal_gamma": 2.0
    }
    
    criterion = CombinedLoss(config)
    loss = criterion(logits, labels, features)
    
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__]) 
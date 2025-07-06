"""Tests for model components."""

import pytest
import torch
import torch.nn as nn
from msa_t_osv.models import MSATOSVModel
from msa_t_osv.models.backbone import CNNBackbone
from msa_t_osv.models.attention import MultiScaleAttention
from msa_t_osv.models.transformer import TransformerEncoder


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
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
        }
    }


@pytest.fixture
def sample_batch():
    """Sample batch for testing."""
    return torch.randn(2, 3, 224, 224)


def test_cnn_backbone(sample_batch):
    """Test CNN backbone."""
    backbone = CNNBackbone(
        backbone_name="resnet50",
        input_size=224,
        spp_levels=[1, 2, 4]
    )
    
    features = backbone(sample_batch)
    
    assert isinstance(features, dict)
    assert "spp_features" in features
    assert "multi_scale_features" in features


def test_multi_scale_attention(sample_batch):
    """Test multi-scale attention module."""
    # First get features from backbone
    backbone = CNNBackbone(
        backbone_name="resnet50",
        input_size=224,
        spp_levels=[1, 2, 4]
    )
    features = backbone(sample_batch)
    
    attention = MultiScaleAttention(
        in_channels=2048,
        spatial=True,
        channel=True,
        scale=True
    )
    
    attended_features = attention(features["multi_scale_features"])
    
    assert isinstance(attended_features, torch.Tensor)
    assert attended_features.shape[0] == sample_batch.shape[0]


def test_transformer_encoder():
    """Test transformer encoder."""
    batch_size = 2
    seq_len = 196  # 14x14 feature map
    hidden_dim = 512
    
    features = torch.randn(batch_size, seq_len, hidden_dim)
    
    transformer = TransformerEncoder(
        num_layers=6,
        num_heads=8,
        hidden_dim=hidden_dim,
        dropout=0.1
    )
    
    encoded_features = transformer(features)
    
    assert encoded_features.shape == features.shape


def test_msa_t_osv_model(sample_config, sample_batch):
    """Test complete MSA-T OSV model."""
    model = MSATOSVModel(sample_config)
    
    # Test forward pass
    logits = model(sample_batch)
    
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (sample_batch.shape[0], 2)
    
    # Test model parameters
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0


def test_model_device_moving(sample_config, sample_batch):
    """Test moving model to different devices."""
    model = MSATOSVModel(sample_config)
    
    if torch.cuda.is_available():
        # Move to GPU
        model = model.cuda()
        sample_batch = sample_batch.cuda()
        
        logits = model(sample_batch)
        assert logits.device.type == "cuda"
        
        # Move back to CPU
        model = model.cpu()
        sample_batch = sample_batch.cpu()
        
        logits = model(sample_batch)
        assert logits.device.type == "cpu"


def test_model_gradient_flow(sample_config, sample_batch):
    """Test gradient flow through the model."""
    model = MSATOSVModel(sample_config)
    
    # Forward pass
    logits = model(sample_batch)
    
    # Backward pass
    loss = nn.CrossEntropyLoss()(logits, torch.randint(0, 2, (sample_batch.shape[0],)))
    loss.backward()
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


def test_model_output_consistency(sample_config, sample_batch):
    """Test model output consistency."""
    model = MSATOSVModel(sample_config)
    model.eval()
    
    with torch.no_grad():
        logits1 = model(sample_batch)
        logits2 = model(sample_batch)
    
    # Outputs should be identical for same input
    assert torch.allclose(logits1, logits2, atol=1e-6)


def test_model_with_different_batch_sizes(sample_config):
    """Test model with different batch sizes."""
    model = MSATOSVModel(sample_config)
    
    batch_sizes = [1, 4, 8]
    
    for batch_size in batch_sizes:
        sample_batch = torch.randn(batch_size, 3, 224, 224)
        logits = model(sample_batch)
        
        assert logits.shape == (batch_size, 2)


if __name__ == "__main__":
    pytest.main([__file__]) 
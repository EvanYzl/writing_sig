"""Tests for evaluation metrics."""

import pytest
import torch
import numpy as np
from msa_t_osv.metrics import (
    MetricTracker,
    compute_eer,
    compute_far_frr,
    compute_accuracy,
    compute_auc,
    compute_precision_recall_f1,
    compute_writer_dependent_metrics
)


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 1000
    n_genuine = 600
    n_forged = 400
    
    # Genuine signatures (lower scores)
    genuine_scores = np.random.normal(0.2, 0.1, n_genuine)
    genuine_labels = np.zeros(n_genuine)
    
    # Forged signatures (higher scores)
    forged_scores = np.random.normal(0.8, 0.1, n_forged)
    forged_labels = np.ones(n_forged)
    
    # Combine
    scores = np.concatenate([genuine_scores, forged_scores])
    labels = np.concatenate([genuine_labels, forged_labels])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    scores = scores[indices]
    labels = labels[indices]
    
    return labels, scores


@pytest.fixture
def sample_writer_data():
    """Sample data with writer information."""
    np.random.seed(42)
    
    n_writers = 10
    n_samples_per_writer = 50
    
    all_labels = []
    all_scores = []
    all_writers = []
    
    for writer_id in range(n_writers):
        # Genuine samples
        genuine_scores = np.random.normal(0.2, 0.1, n_samples_per_writer // 2)
        genuine_labels = np.zeros(n_samples_per_writer // 2)
        
        # Forged samples
        forged_scores = np.random.normal(0.8, 0.1, n_samples_per_writer // 2)
        forged_labels = np.ones(n_samples_per_writer // 2)
        
        # Combine
        writer_scores = np.concatenate([genuine_scores, forged_scores])
        writer_labels = np.concatenate([genuine_labels, forged_labels])
        writer_ids = np.full(n_samples_per_writer, writer_id)
        
        all_labels.extend(writer_labels)
        all_scores.extend(writer_scores)
        all_writers.extend(writer_ids)
    
    return np.array(all_labels), np.array(all_scores), np.array(all_writers)


def test_metric_tracker(sample_data):
    """Test MetricTracker class."""
    labels, scores = sample_data
    
    tracker = MetricTracker()
    
    # Update in batches
    batch_size = 100
    for i in range(0, len(labels), batch_size):
        batch_labels = labels[i:i+batch_size]
        batch_scores = scores[i:i+batch_size]
        tracker.update(batch_labels, batch_scores)
    
    # Compute metrics
    metrics = tracker.compute()
    
    # Check that all expected metrics are present
    expected_metrics = ['eer', 'accuracy', 'auc', 'precision', 'recall', 'f1_score', 'far', 'frr']
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)
        assert not np.isnan(metrics[metric])


def test_compute_eer(sample_data):
    """Test EER computation."""
    labels, scores = sample_data
    
    eer = compute_eer(labels, scores)
    
    assert isinstance(eer, float)
    assert 0.0 <= eer <= 1.0
    assert not np.isnan(eer)


def test_compute_eer_edge_cases():
    """Test EER computation with edge cases."""
    # All genuine
    labels = np.zeros(100)
    scores = np.random.uniform(0, 0.5, 100)
    eer = compute_eer(labels, scores)
    assert isinstance(eer, float)
    
    # All forged
    labels = np.ones(100)
    scores = np.random.uniform(0.5, 1.0, 100)
    eer = compute_eer(labels, scores)
    assert isinstance(eer, float)
    
    # Perfect separation
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    eer = compute_eer(labels, scores)
    assert isinstance(eer, float)


def test_compute_far_frr(sample_data):
    """Test FAR/FRR computation."""
    labels, scores = sample_data
    
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        far, frr = compute_far_frr(labels, scores, threshold)
        
        assert isinstance(far, float)
        assert isinstance(frr, float)
        assert 0.0 <= far <= 1.0
        assert 0.0 <= frr <= 1.0
        assert not np.isnan(far)
        assert not np.isnan(frr)


def test_compute_accuracy(sample_data):
    """Test accuracy computation."""
    labels, scores = sample_data
    
    # Test with different thresholds
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        accuracy = compute_accuracy(labels, predictions)
        
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        assert not np.isnan(accuracy)


def test_compute_auc(sample_data):
    """Test AUC computation."""
    labels, scores = sample_data
    
    auc = compute_auc(labels, scores)
    
    assert isinstance(auc, float)
    assert 0.0 <= auc <= 1.0
    assert not np.isnan(auc)


def test_compute_precision_recall_f1(sample_data):
    """Test precision, recall, and F1 computation."""
    labels, scores = sample_data
    
    # Test with different thresholds
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        precision, recall, f1 = compute_precision_recall_f1(labels, predictions)
        
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(f1, float)
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= f1 <= 1.0
        assert not np.isnan(precision)
        assert not np.isnan(recall)
        assert not np.isnan(f1)


def test_compute_writer_dependent_metrics(sample_writer_data):
    """Test writer-dependent metrics computation."""
    labels, scores, writers = sample_writer_data
    
    metrics = compute_writer_dependent_metrics(labels, scores, writers)
    
    # Check that all expected metrics are present
    expected_metrics = [
        'writer_eer_mean', 'writer_eer_std', 'writer_eer_min', 'writer_eer_max',
        'writer_accuracy_mean', 'writer_accuracy_std',
        'writer_auc_mean', 'writer_auc_std'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)
        assert not np.isnan(metrics[metric])


def test_metric_tracker_reset(sample_data):
    """Test MetricTracker reset functionality."""
    labels, scores = sample_data
    
    tracker = MetricTracker()
    
    # First update
    tracker.update(labels[:100], scores[:100])
    metrics1 = tracker.compute()
    
    # Reset
    tracker.reset()
    
    # Second update with different data
    tracker.update(labels[100:200], scores[100:200])
    metrics2 = tracker.compute()
    
    # Metrics should be different
    assert metrics1['eer'] != metrics2['eer']


def test_metric_tracker_empty():
    """Test MetricTracker with empty data."""
    tracker = MetricTracker()
    
    # Compute without any updates
    metrics = tracker.compute()
    
    # Should return default values
    assert isinstance(metrics, dict)
    assert 'eer' in metrics


def test_metric_tracker_single_class():
    """Test MetricTracker with single class data."""
    tracker = MetricTracker()
    
    # All genuine
    labels = np.zeros(100)
    scores = np.random.uniform(0, 0.5, 100)
    
    tracker.update(labels, scores)
    metrics = tracker.compute()
    
    assert isinstance(metrics, dict)
    assert not np.isnan(metrics['eer'])


def test_metric_consistency(sample_data):
    """Test metric consistency across multiple computations."""
    labels, scores = sample_data
    
    # Compute metrics multiple times
    results = []
    for _ in range(5):
        tracker = MetricTracker()
        tracker.update(labels, scores)
        metrics = tracker.compute()
        results.append(metrics['eer'])
    
    # Results should be consistent
    for i in range(1, len(results)):
        assert abs(results[i] - results[0]) < 1e-6


def test_metric_with_different_data_sizes():
    """Test metrics with different data sizes."""
    sizes = [10, 50, 100, 500]
    
    for size in sizes:
        labels = np.random.randint(0, 2, size)
        scores = np.random.uniform(0, 1, size)
        
        # Test EER
        eer = compute_eer(labels, scores)
        assert isinstance(eer, float)
        assert not np.isnan(eer)
        
        # Test AUC
        auc = compute_auc(labels, scores)
        assert isinstance(auc, float)
        assert not np.isnan(auc)


def test_metric_edge_cases():
    """Test metrics with edge cases."""
    # Single sample
    labels = np.array([0])
    scores = np.array([0.5])
    
    eer = compute_eer(labels, scores)
    assert isinstance(eer, float)
    
    # All same scores
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    
    eer = compute_eer(labels, scores)
    assert isinstance(eer, float)


if __name__ == "__main__":
    pytest.main([__file__]) 
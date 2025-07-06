"""MSA-T OSV: Multi-Scale Attention and Transformer-based Offline Signature Verification."""

__version__ = "0.1.0"
__author__ = "MSA-T OSV Contributors"
__license__ = "MIT"

from . import data, models, utils
from .metrics import compute_eer, compute_far, compute_frr, compute_accuracy
from .train import train_model
from .evaluate import evaluate_model

__all__ = [
    "data",
    "models", 
    "utils",
    "compute_eer",
    "compute_far",
    "compute_frr",
    "compute_accuracy",
    "train_model",
    "evaluate_model",
] 
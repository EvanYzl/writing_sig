"""Inference script for MSA-T OSV."""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import json
import cv2
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msa_t_osv.models import MSATOSVModel
from msa_t_osv.utils.logger import setup_logger, get_logger
from msa_t_osv.utils.seed import set_seed
from msa_t_osv.utils.visualizer import Visualizer


class SignatureVerifier:
    """Signature verification class for inference."""
    
    def __init__(self, checkpoint_path: str, config: Dict[str, Any], device: torch.device):
        """Initialize signature verifier.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration dictionary
            device: Device to run inference on
        """
        self.config = config
        self.device = device
        
        # Load model
        self.model = MSATOSVModel(config)
        self.model = self.model.to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Setup logger
        self.logger = get_logger(__name__)
        
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """Preprocess image for inference.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed tensor
        """
        # Load image if path is provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a path, numpy array, or PIL Image")
        
        # Get preprocessing parameters
        input_size = self.config["model"]["input_size"]
        mean = self.config["data"]["normalize"]["mean"]
        std = self.config["data"]["normalize"]["std"]
        
        # Resize image
        image = image.resize((input_size, input_size), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # HWC to CHW and scale to [0, 1]
        
        # Normalize
        for i in range(3):
            image_tensor[i] = (image_tensor[i] - mean[i]) / std[i]
            
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
        
    def verify_signature(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Verify a single signature.
        
        Args:
            image: Input signature image
            
        Returns:
            Dictionary containing verification results
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)
            scores = probs[:, 1]  # Probability of being forged
            predictions = torch.argmax(logits, dim=1)
            
        # Convert to numpy
        score = scores.cpu().numpy()[0]
        prediction = predictions.cpu().numpy()[0]
        probability = probs.cpu().numpy()[0]
        
        # Determine result
        is_genuine = prediction == 0
        confidence = probability[prediction]
        
        # Get threshold from config
        threshold = self.config.get("inference", {}).get("threshold", 0.5)
        
        # Determine final decision
        if score < threshold:
            decision = "genuine"
        else:
            decision = "forged"
            
        results = {
            'decision': decision,
            'is_genuine': is_genuine,
            'score': float(score),
            'confidence': float(confidence),
            'probabilities': {
                'genuine': float(probability[0]),
                'forged': float(probability[1])
            },
            'threshold': threshold
        }
        
        return results
        
    def verify_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> List[Dict[str, Any]]:
        """Verify a batch of signatures.
        
        Args:
            images: List of input signature images
            
        Returns:
            List of verification results
        """
        results = []
        
        for image in tqdm(images, desc="Verifying signatures"):
            try:
                result = self.verify_signature(image)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing image: {e}")
                results.append({
                    'error': str(e),
                    'decision': 'error',
                    'is_genuine': None,
                    'score': None,
                    'confidence': None,
                    'probabilities': None,
                    'threshold': None
                })
                
        return results


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


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save inference results.
    
    Args:
        results: List of inference results
        output_path: Path to save results
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def print_summary(results: List[Dict[str, Any]]):
    """Print inference summary.
    
    Args:
        results: List of inference results
    """
    total = len(results)
    genuine = sum(1 for r in results if r.get('is_genuine', False))
    forged = sum(1 for r in results if r.get('is_genuine') == False)
    errors = sum(1 for r in results if 'error' in r)
    
    print("=" * 50)
    print("INFERENCE SUMMARY")
    print("=" * 50)
    print(f"Total signatures: {total}")
    print(f"Genuine signatures: {genuine}")
    print(f"Forged signatures: {forged}")
    print(f"Errors: {errors}")
    print("=" * 50)
    
    if total > 0:
        print(f"Genuine rate: {genuine/total:.2%}")
        print(f"Forged rate: {forged/total:.2%}")
        if errors > 0:
            print(f"Error rate: {errors/total:.2%}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Inference with MSA-T OSV model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--threshold', type=float, help='Custom threshold for decision')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override threshold if provided
    if args.threshold is not None:
        config["inference"]["threshold"] = args.threshold
    
    # Set random seed
    set_seed(config["seed"], config["cuda_deterministic"])
    
    # Setup device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    
    # Setup logging
    logger = setup_logger(log_level="INFO")
    
    logger.info(f"Starting inference with config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Device: {device}")
    
    # Create verifier
    verifier = SignatureVerifier(args.checkpoint, config, device)
    logger.info("Model loaded successfully")
    
    # Prepare input
    input_path = Path(args.input)
    images = []
    
    if input_path.is_file():
        # Single image
        images = [str(input_path)]
    elif input_path.is_dir():
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        images = [str(f) for f in input_path.iterdir() 
                 if f.suffix.lower() in image_extensions]
        images.sort()
    else:
        raise ValueError(f"Input path does not exist: {args.input}")
    
    if not images:
        raise ValueError(f"No images found in: {args.input}")
    
    logger.info(f"Found {len(images)} images")
    
    # Run inference
    results = verifier.verify_batch(images)
    
    # Print summary
    print_summary(results)
    
    # Save results
    if args.output:
        save_results(results, args.output)
        logger.info(f"Results saved to {args.output}")
    else:
        # Print results to console
        for i, (image_path, result) in enumerate(zip(images, results)):
            print(f"\nImage {i+1}: {Path(image_path).name}")
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Decision: {result['decision']}")
                print(f"  Score: {result['score']:.4f}")
                print(f"  Confidence: {result['confidence']:.4f}")
                print(f"  Genuine probability: {result['probabilities']['genuine']:.4f}")
                print(f"  Forged probability: {result['probabilities']['forged']:.4f}")
    
    logger.info("Inference completed!")


if __name__ == "__main__":
    main() 
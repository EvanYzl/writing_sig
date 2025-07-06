"""Main entry point for MSA-T OSV."""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from msa_t_osv.train import main as train_main
from msa_t_osv.evaluate import main as evaluate_main
from msa_t_osv.inference import main as inference_main


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='MSA-T OSV: Multi-Scale Attention and Transformer for Offline Signature Verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python -m msa_t_osv train --config configs/cedar.yaml --output_dir outputs/cedar

  # Evaluate a trained model
  python -m msa_t_osv evaluate --config configs/cedar.yaml --checkpoint outputs/cedar/best_eer.pth

  # Run inference on a single image
  python -m msa_t_osv inference --config configs/cedar.yaml --checkpoint outputs/cedar/best_eer.pth --input signature.png

  # Run inference on a directory of images
  python -m msa_t_osv inference --config configs/cedar.yaml --checkpoint outputs/cedar/best_eer.pth --input signatures/ --output results.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', type=str, required=True, help='Path to config file')
    train_parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    train_parser.add_argument('--output_dir', type=str, help='Output directory')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--config', type=str, required=True, help='Path to config file')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--output_dir', type=str, help='Output directory')
    eval_parser.add_argument('--no_vis', action='store_true', help='Skip visualization')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference on images')
    inference_parser.add_argument('--config', type=str, required=True, help='Path to config file')
    inference_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    inference_parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    inference_parser.add_argument('--output', type=str, help='Output JSON file')
    inference_parser.add_argument('--threshold', type=float, help='Custom threshold for decision')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Set up sys.argv for train script
        sys.argv = ['train.py']
        if args.config:
            sys.argv.extend(['--config', args.config])
        if args.resume:
            sys.argv.extend(['--resume', args.resume])
        if args.output_dir:
            sys.argv.extend(['--output_dir', args.output_dir])
        train_main()
        
    elif args.command == 'evaluate':
        # Set up sys.argv for evaluate script
        sys.argv = ['evaluate.py']
        if args.config:
            sys.argv.extend(['--config', args.config])
        if args.checkpoint:
            sys.argv.extend(['--checkpoint', args.checkpoint])
        if args.output_dir:
            sys.argv.extend(['--output_dir', args.output_dir])
        if args.no_vis:
            sys.argv.append('--no_vis')
        evaluate_main()
        
    elif args.command == 'inference':
        # Set up sys.argv for inference script
        sys.argv = ['inference.py']
        if args.config:
            sys.argv.extend(['--config', args.config])
        if args.checkpoint:
            sys.argv.extend(['--checkpoint', args.checkpoint])
        if args.input:
            sys.argv.extend(['--input', args.input])
        if args.output:
            sys.argv.extend(['--output', args.output])
        if args.threshold:
            sys.argv.extend(['--threshold', str(args.threshold)])
        inference_main()
        
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Main script for training the AI face detection model.
"""
import os
import sys
import logging
import argparse
from pathlib import Path

# Ensure the src directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import run_training_pipeline
from src.config import get_config, MODEL_CONFIGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train AI face detection model')
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        default='efficientnet',
        choices=list(MODEL_CONFIGS.keys()),
        help='Model architecture to use'
    )
    
    # Data configuration
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Base data directory (default: "data" in current directory)'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        nargs=2,
        default=(224, 224),
        help='Input image size (width, height)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    # GCP configuration
    parser.add_argument(
        '--use-gcp',
        action='store_true',
        help='Use Google Cloud Storage for data'
    )
    
    parser.add_argument(
        '--gcp-bucket',
        type=str,
        default=None,
        help='GCP bucket name'
    )
    
    # Training configuration
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for models and logs'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the training pipeline."""
    # Parse command line arguments
    args = parse_args()
    
    # Get the configuration
    config = get_config(args.model)
    
    # Override configuration with command line arguments
    if args.data_dir:
        config['data_config']['data_dir'] = args.data_dir
    
    config['data_config']['img_size'] = tuple(args.img_size)
    config['data_config']['batch_size'] = args.batch_size
    config['model_config']['input_shape'] = (*args.img_size, 3)
    
    if args.use_gcp:
        config['data_config']['use_gcp'] = True
    
    if args.gcp_bucket:
        config['data_config']['gcp_bucket_name'] = args.gcp_bucket
    
    config['training_config']['epochs'] = args.epochs
    
    if args.output_dir:
        config['training_config']['output_dir'] = args.output_dir
    
    # Ensure output directory exists
    output_dir = Path(config['training_config']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log configuration
    logger.info(f"Using model: {args.model}")
    logger.info(f"Image size: {config['data_config']['img_size']}")
    logger.info(f"Batch size: {config['data_config']['batch_size']}")
    logger.info(f"Epochs: {config['training_config']['epochs']}")
    logger.info(f"Using GCP: {config['data_config']['use_gcp']}")
    if config['data_config']['use_gcp']:
        logger.info(f"GCP bucket: {config['data_config']['gcp_bucket_name']}")
    logger.info(f"Output directory: {config['training_config']['output_dir']}")
    
    # Run the training pipeline
    try:
        results = run_training_pipeline(
            data_config=config['data_config'],
            model_config=config['model_config'],
            training_config=config['training_config']
        )
        
        # Log results
        eval_results = results.get('evaluation', {})
        accuracy = eval_results.get('accuracy', 0)
        roc_auc = eval_results.get('roc_auc', 0)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main()) 
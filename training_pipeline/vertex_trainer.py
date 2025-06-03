#!/usr/bin/env python3
"""
Vertex AI Training Script for AI Face Detection Model.
This script is designed to run inside a Docker container on Google Cloud Vertex AI.
"""
import os
import argparse
import logging
import json
import yaml
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from google.cloud import storage

# Add src directory to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from src.config import get_config
from src.data import DataProcessor
from src.train import run_training_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments passed by Vertex AI."""
    parser = argparse.ArgumentParser(description='Train AI face detection model on Vertex AI')
    
    # GCP-related args
    parser.add_argument(
        '--data-bucket-name',
        type=str,
        default=os.environ.get('DATA_BUCKET_NAME', 'distinguish-ai-faces-dataset'),
        help='GCS bucket name for dataset'
    )
    
    parser.add_argument(
        '--model-bucket-name',
        type=str,
        default=os.environ.get('MODEL_BUCKET_NAME', 'distinguish-ai-faces-model'),
        help='GCS bucket name for model output'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default=os.environ.get('MODEL_DIR', 'models'),
        help='GCS directory for model output'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=os.environ.get('DATA_DIR', 'data'),
        help='GCS directory for input data'
    )
    
    # Model training args
    parser.add_argument(
        '--model',
        type=str,
        default='efficientnet',
        choices=['efficientnet', 'resnet50', 'mobilenet', 'scratch'],
        help='Model architecture to use'
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
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0001,
        help='Learning rate for optimizer'
    )
    
    return parser.parse_args()


def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
        
        blob.download_to_filename(destination_file_name)
        logger.info(f"Downloaded {source_blob_name} to {destination_file_name}")
    except Exception as e:
        logger.error(f"Error downloading from GCS: {str(e)}")
        raise


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)
        logger.info(f"Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}")
    except Exception as e:
        logger.error(f"Error uploading to GCS: {str(e)}")
        raise


def load_config_from_yaml(yaml_file):
    """Load configuration from YAML file."""
    try:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config from YAML: {str(e)}")
        return None


def setup_gcp_paths(args):
    """Set up GCP paths for data and model."""
    # Define GCS paths
    gcs_data_bucket = args.data_bucket_name
    gcs_model_bucket = args.model_bucket_name
    gcs_data_dir = args.data_dir
    gcs_model_dir = args.model_dir
    
    # Generate unique output directory name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.model}_{timestamp}"
    gcs_output_path = f"{gcs_model_dir}/{model_name}"
    
    # Local paths for temporary storage
    local_data_dir = 'temp_data'
    local_output_dir = 'temp_output'
    
    # Create local directories
    os.makedirs(local_data_dir, exist_ok=True)
    os.makedirs(local_output_dir, exist_ok=True)
    
    return {
        'gcs_data_bucket': gcs_data_bucket,
        'gcs_model_bucket': gcs_model_bucket,
        'gcs_data_dir': gcs_data_dir,
        'gcs_output_path': gcs_output_path,
        'local_data_dir': local_data_dir,
        'local_output_dir': local_output_dir
    }


def main():
    """Main function to run the training pipeline on Vertex AI."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Print TensorFlow version for debugging
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"GPU available: {tf.config.list_physical_devices('GPU')}")
        
        # Try to load config from YAML if exists
        yaml_config = load_config_from_yaml('vertex_ai_config.yaml')
        if yaml_config:
            logger.info("Loaded configuration from YAML file")
            # Override args with YAML config if provided
            if 'training_params' in yaml_config:
                params = yaml_config['training_params']
                args.model = params.get('model_type', args.model)
                args.epochs = params.get('epochs', args.epochs)
                args.batch_size = params.get('batch_size', args.batch_size)
                args.img_size = params.get('img_size', args.img_size)
                args.learning_rate = params.get('learning_rate', args.learning_rate)
                
            # Override bucket names if in YAML config
            if 'container' in yaml_config and 'env_vars' in yaml_config['container']:
                for env_var in yaml_config['container']['env_vars']:
                    if env_var['name'] == 'DATA_BUCKET_NAME':
                        args.data_bucket_name = env_var['value']
                    elif env_var['name'] == 'MODEL_BUCKET_NAME':
                        args.model_bucket_name = env_var['value']
        
        # Set up GCP paths
        paths = setup_gcp_paths(args)
        
        # Get training configuration using our existing config module
        config = get_config(args.model)
        
        # Override with command line arguments
        config['data_config']['img_size'] = tuple(args.img_size)
        config['data_config']['batch_size'] = args.batch_size
        config['model_config']['input_shape'] = (*args.img_size, 3)
        config['training_config']['epochs'] = args.epochs
        config['training_config']['output_dir'] = paths['local_output_dir']
        
        # Enable GCP for data
        config['data_config']['use_gcp'] = True
        config['data_config']['gcp_bucket_name'] = paths['gcs_data_bucket']
        
        # Modify optimizer learning rate if specified
        if hasattr(args, 'learning_rate') and args.learning_rate is not None:
            config['model_config']['learning_rate'] = args.learning_rate
        
        # Log configuration
        logger.info(f"Training with model: {args.model}")
        logger.info(f"Image size: {config['data_config']['img_size']}")
        logger.info(f"Batch size: {config['data_config']['batch_size']}")
        logger.info(f"Epochs: {config['training_config']['epochs']}")
        logger.info(f"Data GCS bucket: {paths['gcs_data_bucket']}")
        logger.info(f"Model GCS bucket: {paths['gcs_model_bucket']}")
        
        # Run the training pipeline
        logger.info("Starting training pipeline...")
        results = run_training_pipeline(
            data_config=config['data_config'],
            model_config=config['model_config'],
            training_config=config['training_config']
        )
        
        # Upload results and model to GCS
        logger.info(f"Training completed. Uploading results to GCS: {paths['gcs_output_path']}")
        
        # Upload the model file
        best_model_path = os.path.join(paths['local_output_dir'], 'checkpoints', 'best_model.h5')
        final_model_path = os.path.join(paths['local_output_dir'], 'final_model.h5')
        
        if os.path.exists(best_model_path):
            upload_to_gcs(
                paths['gcs_model_bucket'],
                best_model_path,
                f"{paths['gcs_output_path']}/best_model.h5"
            )
            
        if os.path.exists(final_model_path):
            upload_to_gcs(
                paths['gcs_model_bucket'],
                final_model_path,
                f"{paths['gcs_output_path']}/final_model.h5"
            )
        
        # Upload evaluation results
        eval_results_path = os.path.join(paths['local_output_dir'], 'evaluation', 'evaluation_results.json')
        if os.path.exists(eval_results_path):
            upload_to_gcs(
                paths['gcs_model_bucket'],
                eval_results_path,
                f"{paths['gcs_output_path']}/evaluation_results.json"
            )
        
        # Save metadata about the training job
        metadata = {
            'model_type': args.model,
            'img_size': args.img_size,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'timestamp': datetime.now().isoformat(),
            'tf_version': tf.__version__,
            'evaluation': results.get('evaluation', {})
        }
        
        with open('metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        upload_to_gcs(
            paths['gcs_model_bucket'],
            'metadata.json',
            f"{paths['gcs_output_path']}/metadata.json"
        )
        
        logger.info("Training job completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main()) 
#!/usr/bin/env python3
"""
Script to create and run a custom training job on Vertex AI.
"""
import os
import argparse
import yaml
import logging
from datetime import datetime

from google.cloud import aiplatform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
STAGING_BUCKET = "gs://distinguish-ai-faces-dataset"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run training job on Vertex AI')
    
    parser.add_argument(
        '--config',
        type=str,
        default='vertex_ai_config.yaml',
        help='Path to Vertex AI configuration YAML file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['efficientnet', 'resnet50', 'mobilenet', 'scratch'],
        help='Override model architecture to use'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size for training'
    )
    
    parser.add_argument(
        '--job-name',
        type=str,
        help='Custom job name (default: model_type + timestamp)'
    )
    
    parser.add_argument(
        '--staging-bucket',
        type=str,
        default=STAGING_BUCKET,
        help='Staging bucket for Vertex AI (default: gs://distinguish-ai-faces-dataset)'
    )
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise


def initialize_vertex_ai(project_id, region, staging_bucket):
    """Initialize Vertex AI SDK."""
    try:
        # Ensure staging bucket has the proper format
        if not staging_bucket.startswith("gs://"):
            staging_bucket = f"gs://{staging_bucket}"
        
        # Initialize the SDK with project, region, and staging bucket
        aiplatform.init(
            project=project_id, 
            location=region,
            staging_bucket=staging_bucket
        )
        logger.info(f"Initialized Vertex AI SDK for project {project_id} in {region}")
        logger.info(f"Using staging bucket: {staging_bucket}")
    except Exception as e:
        logger.error(f"Error initializing Vertex AI: {str(e)}")
        raise


def create_custom_training_job(config, args):
    """Create and run a custom training job on Vertex AI."""
    
    # Override config with command-line arguments if provided
    if args.model:
        config['training_params']['model_type'] = args.model
    
    if args.epochs:
        config['training_params']['epochs'] = args.epochs
    
    if args.batch_size:
        config['training_params']['batch_size'] = args.batch_size
    
    # Create job display name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config['training_params']['model_type']
    
    if args.job_name:
        job_display_name = args.job_name
    else:
        job_display_name = f"ai_face_detection_{model_type}_{timestamp}"
    
    # Machine configuration - doğrudan yapılandırma dosyasından alınır, varsayılan değer yok
    machine_config = config['machine_config']
    machine_type = machine_config.get('machine_type')
    
    # Bir makine tipi belirtilmemişse, en küçük makineyi kullan
    if not machine_type:
        machine_type = 'n1-standard-1'
        logger.warning(f"No machine type specified in config, using {machine_type}")
    
    # GPU parametrelerini kontrol et - eğer yapılandırmada yoksa kullanma
    accelerator_type = machine_config.get('accelerator_type', None)
    accelerator_count = machine_config.get('accelerator_count', 0)
    
    # Container configuration
    container_config = config['container']
    image_uri = container_config.get('image_uri')
    
    # Create environment variables from config
    env_vars = []
    if 'env_vars' in container_config:
        env_vars = container_config['env_vars']
    
    # Add training parameters as environment variables
    for key, value in config['training_params'].items():
        # Skip img_size as it's handled separately
        if key != 'img_size':
            env_vars.append({
                'name': key.upper(),
                'value': str(value)
            })
    
    # Handle img_size specially
    if 'img_size' in config['training_params']:
        img_width, img_height = config['training_params']['img_size']
        env_vars.append({
            'name': 'IMG_WIDTH',
            'value': str(img_width)
        })
        env_vars.append({
            'name': 'IMG_HEIGHT',
            'value': str(img_height)
        })
    
    # Extract bucket names from env_vars or use defaults
    data_bucket_name = "distinguish-ai-faces-dataset"
    model_bucket_name = "distinguish-ai-faces-model"
    
    for env_var in env_vars:
        if env_var['name'] == 'DATA_BUCKET_NAME':
            data_bucket_name = env_var['value']
        elif env_var['name'] == 'MODEL_BUCKET_NAME':
            model_bucket_name = env_var['value']
    
    # Create command arguments
    command = [
        "python", 
        "vertex_trainer.py",
        f"--model={model_type}",
        f"--epochs={config['training_params']['epochs']}",
        f"--batch-size={config['training_params']['batch_size']}",
        f"--img-size={config['training_params']['img_size'][0]}",
        f"{config['training_params']['img_size'][1]}",
        f"--data-bucket-name={data_bucket_name}",
        f"--model-bucket-name={model_bucket_name}",
    ]
    
    if 'learning_rate' in config['training_params']:
        command.append(f"--learning-rate={config['training_params']['learning_rate']}")
    
    # Use model bucket for output URI prefix
    output_uri_prefix = config.get('output_uri_prefix', f"gs://{model_bucket_name}/vertex_output")
    
    # Ensure staging bucket has the proper format
    staging_bucket = args.staging_bucket
    if not staging_bucket.startswith("gs://"):
        staging_bucket = f"gs://{staging_bucket}"
    
    # Log configuration details
    logger.info(f"Creating custom training job: {job_display_name}")
    logger.info(f"Using image: {image_uri}")
    logger.info(f"Using machine type: {machine_type}")
    
    if accelerator_type and accelerator_count > 0:
        logger.info(f"Using accelerator: {accelerator_count} {accelerator_type}")
    else:
        logger.info("Running without GPU acceleration")
    
    logger.info(f"Data bucket: {data_bucket_name}")
    logger.info(f"Model bucket: {model_bucket_name}")
    logger.info(f"Output URI: {output_uri_prefix}")
    logger.info(f"Staging bucket: {staging_bucket}")
    
    try:
        # Create machine spec for the job
        machine_spec = {
            "machine_type": machine_type,
        }
        
        # GPU yapılandırması varsa ekle
        if accelerator_type and accelerator_count > 0:
            machine_spec["accelerator_type"] = accelerator_type
            machine_spec["accelerator_count"] = accelerator_count
        
        # Create and run the custom job
        custom_job = aiplatform.CustomJob(
            display_name=job_display_name,
            worker_pool_specs=[
                {
                    "machine_spec": machine_spec,
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": image_uri,
                        "command": command,
                        "env": env_vars,
                    }
                }
            ],
            staging_bucket=staging_bucket  # Directly specify in constructor
        )
        
        # Run the custom job
        logger.info("Starting custom training job...")
        custom_job.run(sync=True)
        
        logger.info(f"Custom training job {job_display_name} completed!")
        return custom_job
    
    except Exception as e:
        logger.error(f"Error creating or running custom job: {str(e)}")
        # Provide more detailed troubleshooting information
        logger.error("Check that:")
        logger.error(f"1. The staging bucket '{staging_bucket}' exists and you have access to it")
        logger.error(f"2. The Docker image '{image_uri}' exists in Container Registry")
        logger.error(f"3. Your service account has proper permissions")
        raise


def main():
    """Main function to run a custom training job on Vertex AI."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Load configuration from YAML
        config = load_config(args.config)
        
        # Log configuration
        logger.info(f"Using region: {config['region']}")
        logger.info(f"Using machine type: {config['machine_config'].get('machine_type', 'n1-standard-1')}")
        
        # Initialize Vertex AI
        initialize_vertex_ai(
            project_id=config['project_id'],
            region=config['region'],
            staging_bucket=args.staging_bucket
        )
        
        # Create and run custom training job
        job = create_custom_training_job(config, args)
        
        logger.info("Vertex AI training job completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error running Vertex AI training job: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main()) 
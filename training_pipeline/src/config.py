"""
Configuration module for AI face detection training.
"""
import os
from typing import Dict, Any

# Base directory (current working directory by default)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Proje kimliğini ortam değişkenine ayarlayın
os.environ['GOOGLE_CLOUD_PROJECT'] = 'wingie-devops-project'

# GCP configuration
GCP_CONFIG = {
    'use_gcp': False,  # Set to True to use GCP
    'bucket_name': 'distinguish-ai-faces-dataset',  # Bucket adını doğrudan belirtin, ortam değişkenine bağlı olmamasını sağlayın
    'ai_folder': 'ai-faces',
    'human_folder': 'human-faces'
}

# Data configuration
DATA_CONFIG = {
    'img_size': (224, 224),  # Input image size
    'batch_size': 32,        # Batch size for training
    'val_split': 0.2,        # Validation split ratio
    'test_split': 0.1,       # Test split ratio
    'use_gcp': GCP_CONFIG['use_gcp'],
    'gcp_bucket_name': GCP_CONFIG['bucket_name'],
    'gcp_ai_folder': GCP_CONFIG['ai_folder'],
    'gcp_human_folder': GCP_CONFIG['human_folder'],
    'data_dir': os.path.join(BASE_DIR, 'data'),
    'local_ai_dir': 'ai_faces',
    'local_human_dir': 'human_faces'
}

# Model configurations
MODEL_CONFIGS = {
    # Custom CNN from scratch
    'scratch': {
        'model_type': 'scratch',
        'input_shape': (*DATA_CONFIG['img_size'], 3),
        'dropout_rate': 0.5,
        'patience': 5,
        'min_delta': 0.01
    },
    
    # EfficientNetB0 pre-trained model
    'efficientnet': {
        'model_type': 'pretrained',
        'base_model_name': 'EfficientNetB0',
        'input_shape': (*DATA_CONFIG['img_size'], 3),
        'dropout_rate': 0.6,  # Increased for better regularization
        'freeze_base': False,  # Allow fine-tuning of later layers
        'patience': 8,  # Increased patience for more training time
        'min_delta': 0.005  # Decreased for finer improvements
    },
    
    # ResNet50 pre-trained model
    'resnet50': {
        'model_type': 'pretrained',
        'base_model_name': 'ResNet50',
        'input_shape': (*DATA_CONFIG['img_size'], 3),
        'dropout_rate': 0.6,  # Increased for better regularization
        'freeze_base': False,  # Allow fine-tuning of later layers
        'patience': 8,  # Increased patience for more training time
        'min_delta': 0.005  # Decreased for finer improvements
    },
    
    # MobileNetV2 pre-trained model (smaller, faster)
    'mobilenet': {
        'model_type': 'pretrained',
        'base_model_name': 'MobileNetV2',
        'input_shape': (*DATA_CONFIG['img_size'], 3),
        'dropout_rate': 0.6,  # Increased for better regularization
        'freeze_base': False,  # Allow fine-tuning of later layers
        'patience': 8,  # Increased patience for more training time
        'min_delta': 0.005  # Decreased for finer improvements
    }
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 50,  # Increased from 20 for longer training
    'output_dir': os.path.join(BASE_DIR, 'outputs_advanced')  # New output directory for advanced training
}


def get_config(model_name: str = 'efficientnet') -> Dict[str, Any]:
    """
    Get the configuration for the specified model.
    
    Args:
        model_name: Name of the model configuration to use
        
    Returns:
        Dictionary with combined configuration
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")
    
    return {
        'data_config': DATA_CONFIG,
        'model_config': MODEL_CONFIGS[model_name],
        'training_config': TRAINING_CONFIG
    } 
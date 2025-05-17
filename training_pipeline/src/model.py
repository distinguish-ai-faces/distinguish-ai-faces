"""
Model definition for AI face detection.
Contains model architecture using TensorFlow/Keras.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, applications, regularizers
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Optional, Dict, Any, Union


def build_from_scratch(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.5
) -> models.Model:
    """
    Build a CNN model from scratch for AI face detection.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate/2),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate/2),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate/2),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


def build_from_pretrained(
    base_model_name: str = 'EfficientNetB0',
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.5,
    freeze_base: bool = True
) -> models.Model:
    """
    Build a model using a pre-trained base for AI face detection.
    
    Args:
        base_model_name: Name of the pre-trained model
        input_shape: Input image shape (height, width, channels)
        dropout_rate: Dropout rate for regularization
        freeze_base: Whether to freeze the base model layers
        
    Returns:
        Compiled Keras model
    """
    # Define the input tensor
    inputs = layers.Input(shape=input_shape)
    
    # Pre-trained base model
    if base_model_name == 'EfficientNetB0':
        base_model = applications.EfficientNetB0(
            include_top=False, 
            weights='imagenet',
            input_tensor=inputs
        )
    elif base_model_name == 'ResNet50':
        base_model = applications.ResNet50(
            include_top=False, 
            weights='imagenet',
            input_tensor=inputs
        )
    elif base_model_name == 'MobileNetV2':
        base_model = applications.MobileNetV2(
            include_top=False, 
            weights='imagenet',
            input_tensor=inputs
        )
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")
    
    # Freeze the base model if specified, or just freeze the early layers
    if freeze_base:
        # Freeze all layers in the base model
        base_model.trainable = False
    else:
        # Fine-tuning approach: Freeze early layers, but allow training of later layers
        # This allows for model fine-tuning while still preserving learned low-level features
        
        # For EfficientNetB0
        if base_model_name == 'EfficientNetB0':
            # Freeze the first 60% of layers, train the last 40% (daha fazla katman açılıyor)
            total_layers = len(base_model.layers)
            freeze_until = int(total_layers * 0.6)
            
            # Make the model trainable first
            base_model.trainable = True
            
            # Then freeze early layers
            for layer in base_model.layers[:freeze_until]:
                layer.trainable = False
                
            print(f"EfficientNetB0: Freezing {freeze_until} layers out of {total_layers} total layers")
        
        # For ResNet50
        elif base_model_name == 'ResNet50':
            # ResNet50 has 175 layers. Freeze the first 120 layers
            base_model.trainable = True
            for layer in base_model.layers[:100]:  # Daha az katman donduruldu
                layer.trainable = False
                
            print(f"ResNet50: Freezing 100 layers out of {len(base_model.layers)} total layers")
        
        # For MobileNetV2
        elif base_model_name == 'MobileNetV2':
            # MobileNetV2 has 154 layers. Freeze the first 100 layers
            base_model.trainable = True
            for layer in base_model.layers[:80]:  # Daha az katman donduruldu
                layer.trainable = False
                
            print(f"MobileNetV2: Freezing 80 layers out of {len(base_model.layers)} total layers")
    
    # Add custom layers on top
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    
    # Expanded custom layers with more capacity and regularization
    # Increased dropout rate and L2 regularization ekleniyor
    x = layers.Dense(1536, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate + 0.1)(x)  # Dropout oranını artırıyoruz
    
    x = layers.Dense(768, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate + 0.1)(x)
    
    x = layers.Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Combine the base model and custom layers
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Use a smaller learning rate if we're fine-tuning the base model
    lr = 1e-5 if not freeze_base else 1e-4
    
    # Compile model with extra metrics
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.AUC(), 
            tf.keras.metrics.Precision(), 
            tf.keras.metrics.Recall(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives()
        ]
    )
    
    return model


def get_model(
    model_type: str = 'pretrained',
    config: Optional[Dict[str, Any]] = None
) -> models.Model:
    """
    Get a model based on the specified type and configuration.
    
    Args:
        model_type: 'scratch' for a model built from scratch, 'pretrained' for a model using a pre-trained base
        config: Model configuration parameters
        
    Returns:
        Compiled Keras model
    """
    config = config or {}
    
    if model_type.lower() == 'scratch':
        return build_from_scratch(
            input_shape=config.get('input_shape', (224, 224, 3)),
            dropout_rate=config.get('dropout_rate', 0.5)
        )
    elif model_type.lower() == 'pretrained':
        return build_from_pretrained(
            base_model_name=config.get('base_model_name', 'EfficientNetB0'),
            input_shape=config.get('input_shape', (224, 224, 3)),
            dropout_rate=config.get('dropout_rate', 0.5),
            freeze_base=config.get('freeze_base', True)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 
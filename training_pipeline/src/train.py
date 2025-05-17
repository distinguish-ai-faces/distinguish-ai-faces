"""
Training module for AI face detection models.
"""
import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau,
    TensorBoard,
    LearningRateScheduler
)
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_curve, 
    auc, 
    precision_recall_curve
)

from .model import get_model
from .data import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_callbacks(
    checkpoint_dir: Union[str, Path],
    tensorboard_dir: Union[str, Path] = None,
    patience: int = 5,
    min_delta: float = 0.01
) -> List[tf.keras.callbacks.Callback]:
    """
    Get training callbacks.
    
    Args:
        checkpoint_dir: Directory to save model checkpoints
        tensorboard_dir: Directory for TensorBoard logs
        patience: Patience for early stopping
        min_delta: Minimum delta for early stopping and LR reduction
        
    Returns:
        List of Keras callbacks
    """
    # Ensure directories exist
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Learning rate schedule fonksiyonu
    def lr_schedule(epoch, lr):
        # İlk 5 epoch'ta sabit öğrenme oranı
        if epoch < 5:
            return lr
        # Sonraki epoch'larda kademeli olarak azalma
        else:
            decay_rate = 0.95
            decay_step = 1
            return lr * (decay_rate ** (epoch // decay_step))
    
    callbacks = [
        # Model checkpoint
        ModelCheckpoint(
            filepath=str(checkpoint_dir / 'model_{epoch:02d}_{val_accuracy:.4f}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        # Best model checkpoint
        ModelCheckpoint(
            filepath=str(checkpoint_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        # Early stopping - daha fazla sabır ekledik
        EarlyStopping(
            monitor='val_loss',
            patience=patience + 3,  # Sabır artırıldı
            min_delta=min_delta,
            verbose=1,
            restore_best_weights=True
        ),
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=patience // 2,
            min_delta=min_delta,
            min_lr=1e-6,
            verbose=1
        ),
        # Learning rate scheduler
        LearningRateScheduler(lr_schedule, verbose=1)
    ]
    
    # Add TensorBoard if directory is provided
    if tensorboard_dir:
        tensorboard_dir = Path(tensorboard_dir)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks.append(
            TensorBoard(
                log_dir=str(tensorboard_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=False,
                update_freq='epoch',
                profile_batch=0
            )
        )
    
    return callbacks


def train_model(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_config: Dict[str, Any],
    output_dir: Union[str, Path],
    epochs: int = 20
) -> Tuple[Model, Dict[str, Any]]:
    """
    Train the model.
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        model_config: Model configuration
        output_dir: Output directory for saving models and logs
        epochs: Number of epochs to train
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Create output directories
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    tensorboard_dir = output_dir / 'logs'
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model
    model = get_model(
        model_type=model_config.get('model_type', 'pretrained'),
        config=model_config
    )
    
    # Get callbacks
    callbacks = get_callbacks(
        checkpoint_dir=checkpoint_dir,
        tensorboard_dir=tensorboard_dir,
        patience=model_config.get('patience', 5),
        min_delta=model_config.get('min_delta', 0.01)
    )
    
    # Train the model
    logger.info(f"Starting model training for {epochs} epochs")
    start_time = time.time()
    
    # Sınıf ağırlıklarını ayarla - dengesiz veri seti için faydalı
    class_weight = {0: 1.0, 1: 1.2}  # İnsan yüzlerine (1) daha fazla ağırlık
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weight  # Sınıf ağırlıklarını ekle
    )
    
    training_time = time.time() - start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    # Save the final model
    model.save(output_dir / 'final_model.h5')
    logger.info(f"Model saved to {output_dir / 'final_model.h5'}")
    
    # Convert history to dict if it's a Keras History object
    if hasattr(history, 'history'):
        history = history.history
    
    return model, history


def evaluate_model(
    model: Model,
    test_ds: tf.data.Dataset,
    output_dir: Union[str, Path] = None
) -> Dict[str, Any]:
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        test_ds: Test dataset
        output_dir: Output directory for saving evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Create output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Predict on test dataset
    logger.info("Evaluating model on test dataset")
    
    # Extract test data as numpy arrays
    test_images = []
    test_labels = []
    
    for images, labels in test_ds:
        test_images.append(images.numpy())
        test_labels.append(labels.numpy())
    
    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    # Get predictions
    y_pred = model.predict(test_images)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = np.mean(y_pred_classes == test_labels)
    
    # Generate classification report
    report = classification_report(test_labels, y_pred_classes, output_dict=True)
    logger.info(f"Classification report:\n{classification_report(test_labels, y_pred_classes)}")
    
    # Generate confusion matrix
    cm = confusion_matrix(test_labels, y_pred_classes)
    logger.info(f"Confusion matrix:\n{cm}")
    
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(test_labels, y_pred)
    roc_auc = auc(fpr, tpr)
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(test_labels, y_pred)
    pr_auc = auc(recall, precision)
    logger.info(f"PR AUC: {pr_auc:.4f}")
    
    # Save plots if output_dir is provided
    if output_dir:
        # ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(output_dir / 'roc_curve.png')
        plt.close()
        
        # Precision-Recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(output_dir / 'precision_recall_curve.png')
        plt.close()
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['AI', 'Human'])
        plt.yticks(tick_marks, ['AI', 'Human'])
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(output_dir / 'confusion_matrix.png')
        plt.close()
    
    # Return results as a dictionary
    return {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc)
    }


def save_evaluation_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path]
) -> None:
    """
    Save evaluation results to output directory.
    
    Args:
        results: Evaluation results
        output_dir: Output directory
    """
    import json
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results as JSON
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Evaluation results saved to {output_dir / 'evaluation_results.json'}")


def run_training_pipeline(
    data_config: Dict[str, Any],
    model_config: Dict[str, Any],
    training_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run the complete training pipeline.
    
    Args:
        data_config: Data configuration
        model_config: Model configuration
        training_config: Training configuration
        
    Returns:
        Dictionary with training results
    """
    # Initialize data processor
    data_processor = DataProcessor(
        img_size=data_config.get('img_size', (224, 224)),
        batch_size=data_config.get('batch_size', 32),
        val_split=data_config.get('val_split', 0.2),
        test_split=data_config.get('test_split', 0.1),
        use_gcp=data_config.get('use_gcp', False),
        gcp_bucket_name=data_config.get('gcp_bucket_name')
    )
    
    # Load dataset
    train_ds, val_ds, test_ds = data_processor.load_dataset(
        data_dir=data_config.get('data_dir'),
        gcp_ai_folder=data_config.get('gcp_ai_folder', 'ai-faces'),
        gcp_human_folder=data_config.get('gcp_human_folder', 'human-faces'),
        local_ai_dir=data_config.get('local_ai_dir', 'data/ai_faces'),
        local_human_dir=data_config.get('local_human_dir', 'data/human_faces')
    )
    
    # Train model
    output_dir = Path(training_config.get('output_dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, history = train_model(
        train_ds=train_ds,
        val_ds=val_ds,
        model_config=model_config,
        output_dir=output_dir,
        epochs=training_config.get('epochs', 20)
    )
    
    # Evaluate model
    eval_results = evaluate_model(
        model=model,
        test_ds=test_ds,
        output_dir=output_dir / 'evaluation'
    )
    
    # Save evaluation results
    save_evaluation_results(
        results=eval_results,
        output_dir=output_dir / 'evaluation'
    )
    
    # Return combined results
    return {
        'history': history,
        'evaluation': eval_results
    } 
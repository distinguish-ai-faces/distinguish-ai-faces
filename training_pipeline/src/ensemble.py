"""
Ensemble model for AI face detection.
This module combines multiple models to make more accurate predictions.
"""
import logging
import numpy as np
from typing import List, Union, Dict, Any, Tuple, Optional
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble predictor that combines multiple models for better predictions."""
    
    def __init__(
        self,
        model_paths: List[Union[str, Path]],
        img_size: Tuple[int, int] = (224, 224),
        weights: Optional[List[float]] = None
    ):
        """
        Initialize the ensemble predictor.
        
        Args:
            model_paths: List of paths to trained model files (.h5)
            img_size: Input image size expected by the models
            weights: Optional weights for each model (must sum to 1.0)
        """
        self.model_paths = [Path(path) for path in model_paths]
        self.img_size = img_size
        self.models = []
        
        # Validate and normalize weights if provided
        if weights:
            if len(weights) != len(model_paths):
                raise ValueError("Number of weights must match number of models")
            
            # Normalize weights to sum to 1.0
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            # Equal weights by default
            self.weights = [1.0 / len(model_paths) for _ in model_paths]
        
        # Load all models
        self._load_models()
        
        logger.info(f"Ensemble predictor initialized with {len(self.models)} models")
    
    def _load_models(self):
        """Load all models from file paths."""
        for path in self.model_paths:
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            
            try:
                model = load_model(str(path))
                self.models.append(model)
                logger.info(f"Model loaded successfully from {path}")
            except Exception as e:
                logger.error(f"Error loading model {path}: {str(e)}")
                raise
    
    def preprocess_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Preprocess an image for prediction.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load and resize image
            img = load_img(str(image_path), target_size=self.img_size)
            
            # Convert to array and normalize
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def predict_single(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Make a prediction for a single image using ensemble of models.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results
        """
        if not self.models:
            raise ValueError("No models loaded")
        
        # Preprocess the image
        img_array = self.preprocess_image(image_path)
        
        # Get predictions from each model
        predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(img_array)[0][0]
            weighted_pred = pred * self.weights[i]
            predictions.append(weighted_pred)
            logger.debug(f"Model {i+1} prediction: {pred:.4f} (weight: {self.weights[i]:.2f})")
        
        # Compute weighted average prediction
        final_pred = sum(predictions)
        
        # Determine class (AI = 0, Human = 1)
        pred_class = 'Human' if final_pred > 0.5 else 'AI'
        confidence = float(final_pred) if pred_class == 'Human' else 1 - float(final_pred)
        
        # Get individual model predictions
        individual_preds = [float(model.predict(img_array)[0][0]) for model in self.models]
        model_predictions = {f"model_{i+1}": pred for i, pred in enumerate(individual_preds)}
        
        logger.info(f"Ensemble prediction for {image_path}: {pred_class} (confidence: {confidence:.4f})")
        
        return {
            'image_path': str(image_path),
            'prediction': pred_class,
            'confidence': confidence,
            'raw_score': float(final_pred),
            'model_predictions': model_predictions
        }
    
    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of dictionaries with prediction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {image_path}: {str(e)}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        
        return results


def create_ensemble(
    base_output_dir: Union[str, Path] = "outputs",
    model_versions: List[str] = ["v1", "v2", "v3"],
    model_file: str = "best_model.h5",
    custom_weights: Optional[List[float]] = None
) -> EnsemblePredictor:
    """
    Create an ensemble predictor from multiple model versions.
    
    Args:
        base_output_dir: Base directory where model outputs are stored
        model_versions: List of model version folders
        model_file: Model file name to use (default: best_model.h5)
        custom_weights: Optional custom weights for each model
        
    Returns:
        EnsemblePredictor instance
    """
    base_dir = Path(base_output_dir)
    model_paths = []
    
    # Build model paths
    for version in model_versions:
        if version.startswith("v"):
            # Version folder like "outputs_v1"
            output_folder = f"{base_dir.stem}_{version}"
            model_path = base_dir.parent / output_folder / "checkpoints" / model_file
        else:
            # Default output folder
            model_path = base_dir / "checkpoints" / model_file
        
        model_paths.append(model_path)
    
    # Create ensemble predictor
    return EnsemblePredictor(model_paths, weights=custom_weights) 
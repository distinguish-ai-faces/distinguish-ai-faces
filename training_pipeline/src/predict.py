"""
Prediction module for AI face detection.
"""
import os
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Predictor:
    """Class for making predictions with a trained model."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        img_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model file (.h5)
            img_size: Input image size expected by the model
        """
        self.model_path = Path(model_path)
        self.img_size = img_size
        self.model = None
        
        # Load the model
        self._load_model()
        
        logger.info(f"Predictor initialized with model: {model_path}")
        logger.info(f"Expected image size: {img_size}")
    
    def _load_model(self):
        """Load the model from file."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = load_model(str(self.model_path))
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
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
        Make a prediction for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess the image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        prediction = self.model.predict(img_array)[0][0]
        
        # Determine class (AI = 0, Human = 1)
        pred_class = 'Human' if prediction > 0.5 else 'AI'
        confidence = float(prediction) if pred_class == 'Human' else 1 - float(prediction)
        
        logger.info(f"Prediction for {image_path}: {pred_class} (confidence: {confidence:.4f})")
        
        return {
            'image_path': str(image_path),
            'prediction': pred_class,
            'confidence': confidence,
            'raw_score': float(prediction)
        }
    
    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of dictionaries with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
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
    
    def visualize_prediction(
        self,
        image_path: Union[str, Path],
        show_confidence: bool = True,
        save_path: Optional[Union[str, Path]] = None
    ):
        """
        Visualize the prediction for a single image.
        
        Args:
            image_path: Path to the image file
            show_confidence: Whether to show confidence score
            save_path: Path to save the visualization (if None, display only)
        """
        # Make prediction
        result = self.predict_single(image_path)
        
        # Load original image
        img = load_img(str(image_path))
        
        # Set up the plot
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        
        # Set title based on prediction
        pred_class = result['prediction']
        confidence = result['confidence']
        
        title = f"Prediction: {pred_class}"
        if show_confidence:
            title += f" (Confidence: {confidence:.2f})"
        
        # Set color based on prediction (green for Human, red for AI)
        color = 'green' if pred_class == 'Human' else 'red'
        
        plt.title(title, fontsize=14, color=color)
        plt.axis('off')
        
        # Save or show
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
        
        return result


def predict_image(
    model_path: Union[str, Path],
    image_path: Union[str, Path],
    img_size: Tuple[int, int] = (224, 224),
    visualize: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to predict a single image.
    
    Args:
        model_path: Path to the trained model file (.h5)
        image_path: Path to the image file
        img_size: Input image size expected by the model
        visualize: Whether to visualize the prediction
        
    Returns:
        Dictionary with prediction results
    """
    predictor = Predictor(model_path=model_path, img_size=img_size)
    
    if visualize:
        return predictor.visualize_prediction(image_path)
    else:
        return predictor.predict_single(image_path)


def batch_predict(
    model_path: Union[str, Path],
    image_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    img_size: Tuple[int, int] = (224, 224),
    file_extensions: List[str] = ['.jpg', '.jpeg', '.png']
) -> List[Dict[str, Any]]:
    """
    Predict all images in a directory.
    
    Args:
        model_path: Path to the trained model file (.h5)
        image_dir: Directory containing images
        output_dir: Directory to save visualizations (if None, no visualizations saved)
        img_size: Input image size expected by the model
        file_extensions: List of file extensions to consider
        
    Returns:
        List of dictionaries with prediction results
    """
    image_dir = Path(image_dir)
    
    # Find all image files
    image_paths = []
    for ext in file_extensions:
        image_paths.extend(list(image_dir.glob(f"*{ext}")))
    
    logger.info(f"Found {len(image_paths)} images in {image_dir}")
    
    if not image_paths:
        logger.warning(f"No images found in {image_dir} with extensions {file_extensions}")
        return []
    
    # Initialize predictor
    predictor = Predictor(model_path=model_path, img_size=img_size)
    
    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    results = []
    for img_path in image_paths:
        result = predictor.predict_single(img_path)
        results.append(result)
        
        # Save visualization if output directory provided
        if output_dir:
            output_path = output_dir / f"{img_path.stem}_pred.jpg"
            predictor.visualize_prediction(img_path, save_path=output_path)
    
    # Generate summary
    ai_count = sum(1 for r in results if r['prediction'] == 'AI')
    human_count = sum(1 for r in results if r['prediction'] == 'Human')
    
    logger.info(f"Prediction complete. Found {ai_count} AI faces and {human_count} human faces.")
    
    return results 
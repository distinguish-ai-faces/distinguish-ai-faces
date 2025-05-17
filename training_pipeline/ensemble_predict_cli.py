#!/usr/bin/env python3
"""
Command-line interface for making ensemble predictions with multiple trained models.
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Optional

# Ensure the src directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ensemble import EnsemblePredictor, create_ensemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make ensemble predictions using multiple models')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image',
        type=str,
        help='Path to a single image for prediction'
    )
    input_group.add_argument(
        '--dir',
        type=str,
        help='Directory containing images for batch prediction'
    )
    
    # Model paths (either direct paths or use model_versions with base_dir)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--model-paths',
        type=str,
        nargs='+',
        help='Paths to the trained models (.h5 files)'
    )
    model_group.add_argument(
        '--model-versions',
        type=str,
        nargs='+',
        help='Model version folders to use (e.g. v1 v2 v3)'
    )
    
    # Optional base directory
    parser.add_argument(
        '--base-dir',
        type=str,
        default='outputs',
        help='Base directory for model outputs (used with --model-versions)'
    )
    
    # Model file name (used with --model-versions)
    parser.add_argument(
        '--model-file',
        type=str,
        default='best_model.h5',
        help='Model file name to use (default: best_model.h5)'
    )
    
    # Weights for models
    parser.add_argument(
        '--weights',
        type=float,
        nargs='+',
        help='Weights for each model (must match number of models)'
    )
    
    # Image size
    parser.add_argument(
        '--img-size',
        type=int,
        nargs=2,
        default=(224, 224),
        help='Input image size (width, height)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save prediction visualizations'
    )
    
    parser.add_argument(
        '--json',
        type=str,
        help='Path to save JSON results'
    )
    
    return parser.parse_args()


def get_image_paths(dir_path: str) -> List[str]:
    """Get all image paths from a directory."""
    dir_path = Path(dir_path)
    extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    
    for ext in extensions:
        image_paths.extend([str(p) for p in dir_path.glob(f'*{ext}')])
    
    return image_paths


def visualize_prediction(result, output_dir: Optional[str] = None):
    """
    Visualize ensemble prediction result.
    
    Args:
        result: Prediction result dictionary
        output_dir: Output directory for visualization
    """
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing.image import load_img
    
    # Load the image
    img = load_img(result['image_path'])
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis('off')
    
    # Display model predictions
    plt.subplot(1, 2, 2)
    
    # Extract model predictions
    model_names = sorted(result['model_predictions'].keys())
    model_values = [result['model_predictions'][name] for name in model_names]
    
    # Plot bar chart
    bars = plt.bar(model_names, model_values)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    
    # Color bars based on prediction
    for i, bar in enumerate(bars):
        bar.set_color('green' if model_values[i] > 0.5 else 'red')
    
    # Add final ensemble prediction
    plt.title(f"Ensemble Prediction: {result['prediction']} ({result['confidence']:.2f})")
    plt.ylim(0, 1)
    plt.ylabel("Human Face Probability")
    
    # Save or show
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        img_name = Path(result['image_path']).stem
        output_path = output_dir / f"{img_name}_ensemble.png"
        plt.savefig(output_path)
        logger.info(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function to run the prediction CLI."""
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Create ensemble predictor
        if args.model_paths:
            # Use direct model paths
            ensemble = EnsemblePredictor(
                model_paths=args.model_paths,
                img_size=tuple(args.img_size),
                weights=args.weights
            )
        else:
            # Use model versions with base directory
            ensemble = create_ensemble(
                base_output_dir=args.base_dir,
                model_versions=args.model_versions,
                model_file=args.model_file,
                custom_weights=args.weights
            )
        
        logger.info(f"Ensemble predictor created with {len(ensemble.models)} models")
        
        # Process based on input type
        if args.image:
            # Single image prediction
            logger.info(f"Predicting single image: {args.image}")
            
            result = ensemble.predict_single(args.image)
            
            # Visualize prediction
            if args.output_dir:
                visualize_prediction(result, args.output_dir)
            else:
                visualize_prediction(result)
            
            # Print result
            logger.info(f"Ensemble prediction: {result['prediction']} (Confidence: {result['confidence']:.4f})")
            
            # Save results as JSON if requested
            if args.json:
                with open(args.json, 'w') as f:
                    json.dump(result, f, indent=4)
                logger.info(f"Results saved to {args.json}")
                
        else:
            # Batch prediction
            logger.info(f"Batch predicting images in directory: {args.dir}")
            
            image_paths = get_image_paths(args.dir)
            if not image_paths:
                logger.warning(f"No images found in {args.dir}")
                return 1
                
            logger.info(f"Found {len(image_paths)} images")
            
            results = ensemble.predict_batch(image_paths)
            
            # Visualize each prediction
            if args.output_dir:
                for result in results:
                    visualize_prediction(result, args.output_dir)
            
            # Print summary
            ai_count = sum(1 for r in results if r['prediction'] == 'AI')
            human_count = sum(1 for r in results if r['prediction'] == 'Human')
            
            logger.info(f"Ensemble prediction complete. Found {ai_count} AI faces and {human_count} human faces.")
            
            # Save results as JSON if requested
            if args.json:
                with open(args.json, 'w') as f:
                    json.dump(results, f, indent=4)
                logger.info(f"Results saved to {args.json}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main()) 
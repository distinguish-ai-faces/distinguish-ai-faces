#!/usr/bin/env python3
"""
Command-line interface for making predictions with the trained model.
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Ensure the src directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict_image, batch_predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict AI vs Human faces')
    
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
    
    # Model path
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the trained model (.h5 file)'
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
        help='Directory to save visualizations (for batch predictions)'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable visualization (for single image prediction)'
    )
    
    parser.add_argument(
        '--json',
        type=str,
        help='Path to save JSON results'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the prediction CLI."""
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Process based on input type
        if args.image:
            # Single image prediction
            logger.info(f"Predicting single image: {args.image}")
            
            result = predict_image(
                model_path=args.model,
                image_path=args.image,
                img_size=tuple(args.img_size),
                visualize=not args.no_visualize
            )
            
            # Print result
            logger.info(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.4f})")
            
            # Save results as JSON if requested
            if args.json:
                with open(args.json, 'w') as f:
                    json.dump(result, f, indent=4)
                logger.info(f"Results saved to {args.json}")
                
        else:
            # Batch prediction
            logger.info(f"Batch predicting images in directory: {args.dir}")
            
            results = batch_predict(
                model_path=args.model,
                image_dir=args.dir,
                output_dir=args.output_dir,
                img_size=tuple(args.img_size)
            )
            
            # Print summary
            ai_count = sum(1 for r in results if r['prediction'] == 'AI')
            human_count = sum(1 for r in results if r['prediction'] == 'Human')
            
            logger.info(f"Prediction complete. Found {ai_count} AI faces and {human_count} human faces.")
            
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
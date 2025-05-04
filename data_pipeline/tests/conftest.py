"""
Configuration file for pytest in the tests directory.
This file adjusts import paths to make relative imports work correctly.
"""
import os
import sys
from pathlib import Path

# Get the parent directory (data_pipeline)
PARENT_DIR = Path(__file__).parent.parent.absolute()

# Add the parent directory to Python path
sys.path.insert(0, str(PARENT_DIR))

# This enables the relative imports in test files to work correctly 
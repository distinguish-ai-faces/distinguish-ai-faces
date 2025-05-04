"""
Configuration file for pytest.
This file is automatically loaded by pytest before running tests.
"""
import os
import sys
from pathlib import Path

# Get the current directory (where conftest.py is located)
BASE_DIR = Path(__file__).parent.absolute()

# Add the parent directory to Python path to enable imports from 'src'
sys.path.insert(0, str(BASE_DIR.parent))  # Parent directory for absolute imports
sys.path.insert(0, str(BASE_DIR))  # Current directory for relative imports

# This allows imports like "from src.xxx import yyy" to work in test files
# even though they are using relative imports like "from ..src.xxx import yyy" 
"""
AI Face Data Pipeline.

A pipeline for scraping, processing, and preparing AI-generated face images.
"""

__version__ = "0.1.0"

# Make src submodules accessible
from data_pipeline.src import scrape_faces, preprocess, gcp_storage 
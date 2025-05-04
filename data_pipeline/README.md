# Data Pipeline for AI Face Detection

## Overview
This pipeline is designed to scrape, process, and prepare AI-generated face images for analysis. It includes tools for downloading images, preprocessing them, and ensuring data quality.

## Features
- Automated scraping of AI-generated faces
- Image preprocessing (resizing, watermark removal, face centering)
- Duplicate detection and removal
- Configurable output formats and quality settings

## Installation
1. Clone the repository
2. Install dependencies:
```bash
make install-dev
```

## Usage
Run the pipeline with default settings:
```bash
python main.py
```

Common options:
- `--count`: Number of images to download (default: 5)
- `--raw-dir`: Directory for raw images (default: "img")
- `--processed-dir`: Directory for processed images (default: "img/processed")
- `--width/--height`: Output image dimensions (default: 512x512)
- `--format`: Output image format (jpg/png)
- `--quality`: JPEG quality (0-100)

## Development
- Run tests: `make test`
- Run linting: `make lint`
- Format code: `make format`
- Clean build files: `make clean`
 
# Distinguish AI Faces 

```bash
distinguish-ai-faces/
│
├── .github/
│   └── workflows/
│       ├── build-dataset-image.yml   # builds and pushes data pipeline image
│       └── build-train-image.yml     # builds and pushes training image
│
├── data_pipeline/                    # Data collection and preprocessing
│   ├── scrape_faces.py               # Web scraping from sources
│   ├── preprocess.py                 # Image resizing, normalization, etc.
│   ├── upload_to_gcs.py              # Upload to Google Cloud Storage
│   ├── requirements.txt
│   └── README.md
│
├── training_pipeline/               # Model development and training
│   ├── train.py                      # Training loop and validation
│   ├── model.py                      # CNN model architecture
│   ├── config.py                     # Parameters (batch size, epochs, etc.)
│   ├── Dockerfile                    # Containerized training definition
│   ├── requirements.txt
│   └── README.md
│
├── deploy_model.md                  # Instructions for Vertex AI training jobs
├── setup_env.md                     # GCP setup, IAM roles, buckets, etc.
├── .gitignore
└── README.md                        # This file
```
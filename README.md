# TryOnDiffusion

## Directory structure

```code
TryOnDiffusion/
|
|-- data/                          # Directory to store datasets
|   |-- raw/                       # Raw, unprocessed data
|   |-- processed/                 # Preprocessed data and datasets ready for training/testing
|
|-- models/                        # Directory for model architectures
|   |-- unet.py                    # UNet model definitions
|   |-- parallel_unet.py           # Parallel-UNet model definitions
|   |-- efficient_unet.py          # Efficient-UNet model definitions
|
|-- utils/                         # Utilities and helper functions
|   |-- preprocessing.py           # Preprocessing functions and utilities
|   |-- postprocessing.py          # Postprocessing functions and utilities
|   |-- metrics.py                 # Evaluation metrics (e.g., FID, KID)
|   |-- visualization.py           # Functions for visualizing results, attention maps, etc.
|
|-- checkpoints/                   # Directory to save model checkpoints during training
|
|-- logs/                          # Directory for logging, tensorboard logs, etc.
|
|-- experiments/                   # Scripts for various experiments, hyperparameter tuning, etc.
|
|-- main.py                        # Main training and evaluation script
|-- config.py                      # Configuration file for hyperparameters, paths, etc.
|-- requirements.txt               # List of dependencies and required libraries
|-- README.md                      # Project documentation and instructions

```

## Run

```
python main.py --training-batch_size 64 --data-train_dir "new/path/to/data"
```

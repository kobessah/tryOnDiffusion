import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Load configuration")

    # Training
    parser.add_argument("--training-batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--training-learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--training-num_epochs", type=int, default=100, help="Number of training epochs")

    # Data
    parser.add_argument("--data-train_dir", type=str, default="path/to/train/data", help="Path to the training data")

    # Model
    parser.add_argument("--model-in_channels", type=int, default=3, help="Number of input channels for the model")
    parser.add_argument("--model-out_channels", type=int, default=3, help="Number of output channels for the model")

    # Checkpoints
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")

    args = parser.parse_args()
    return args
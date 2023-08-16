from models.unet import UNet
from utils.preprocessing import parse_human, segment_garment, generate_clothing_agnostic_rgb
from utils.postprocessing import postprocess_output
import torch

from config import get_args

def main():
    # Dataset loading and preparation
    # TODO: Load the dataset, split into train/test
    
    # Model instantiation
    model = UNet(in_channels=3, out_channels=3)
    
    # Training loop
    # TODO: Define optimizer, loss, and training loop
    
    # Evaluation
    # TODO: Load test data, forward pass through the model, compute metrics

    return

if __name__ == "__main__":
    args = get_args()
    print(args)

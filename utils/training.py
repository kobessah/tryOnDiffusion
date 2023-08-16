# utils/training.py

import torch

def train_epoch(model, dataloader, optimizer, criterion, device="cuda"):
    """
    Training loop for a single epoch.
    """
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        # TODO: Implement the training loop (forward pass, compute loss, backward pass, optimizer step)
        pass
    
    return total_loss / len(dataloader)

def save_model(model, optimizer, epoch, save_path):
    """
    Save model checkpoint.
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }, save_path)

def load_model(model, optimizer, load_path, device="cuda"):
    """
    Load model checkpoint.
    """
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return model, optimizer, epoch

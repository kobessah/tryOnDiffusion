import torch
from torch.utils.data import Dataset, DataLoader
from utils.preprocessing import parse_human, segment_garment, generate_clothing_agnostic_rgb
import os
from PIL import Image

class TryOnDataset(Dataset):
    """
    Dataset class for the virtual try-on task.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]  # Assuming .jpg format

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        
        # Placeholder for preprocessing
        parsing_map, keypoints = parse_human(image_path)
        segmented_garment = segment_garment(parsing_map)
        agnostic_rgb = generate_clothing_agnostic_rgb(image, parsing_map, keypoints)
        
        return {
            "image": image,
            "parsing_map": parsing_map,
            "keypoints": keypoints,
            "segmented_garment": segmented_garment,
            "agnostic_rgb": agnostic_rgb
        }

def get_dataloader(data_dir, batch_size=32, shuffle=True):
    """
    Utility function to get data loader.
    """
    dataset = TryOnDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

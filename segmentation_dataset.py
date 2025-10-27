import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegmentationDataset(Dataset):
    """
    A custom PyTorch Dataset for loading images and their corresponding segmentation masks.
    FINAL CORRECTED VERSION: Works with the 'masks' folder and '_segmentation.png' filenames.
    """
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        
        self.image_folder = os.path.join(self.base_dir, 'all_images')
        self.mask_folder = os.path.join(self.base_dir, 'masks')
        metadata_path = os.path.join(self.base_dir, 'HAM10000_metadata.csv')
        
        self.metadata_df = pd.read_csv(metadata_path)
        self.image_ids = self.metadata_df['image_id'].values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        image_path = os.path.join(self.image_folder, f'{image_id}.jpg')
        # CORRECTED: The mask filename has a '_segmentation' suffix.
        mask_path = os.path.join(self.mask_folder, f'{image_id}_segmentation.png')
        
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        mask[mask > 0] = 1.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        mask = mask.unsqueeze(0)
        
        return image, mask

def get_segmentation_transforms(image_size=256):
    """Defines image and mask transformations for segmentation."""
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

# --- Built-in Test ---
if __name__ == '__main__':
    print("Testing the FINAL CORRECTED SegmentationDataset...")
    
    BASE_DIR = 'data/ham10000'
    
    if not os.path.exists(os.path.join(BASE_DIR, 'masks')):
        print("\nERROR: The 'masks' directory was not found inside 'data/ham10000'.")
    else:
        _, test_transform = get_segmentation_transforms()
        
        dataset = SegmentationDataset(base_dir=BASE_DIR, transform=test_transform)
        
        print(f"Dataset size: {len(dataset)} samples.")
        
        print("Fetching one sample...")
        image, mask = dataset[0]
        
        print("Sample fetched successfully!")
        print(f"  - Image tensor shape: {image.shape}")
        print(f"  - Mask tensor shape: {mask.shape}")
        
        assert image.shape[0] == 3
        assert mask.shape[0] == 1
        assert image.shape[1] == mask.shape[1] and image.shape[2] == mask.shape[2]
        
        print("FINAL CORRECTED Segmentation Data Factory test passed!")


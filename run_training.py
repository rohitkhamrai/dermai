import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR100
from torchvision import transforms
import argparse

# Import our custom classes
from dataset import SkinLesionFusionDataset
from main_model import DualBrainModel
from gatekeeper_model import GatekeeperModel

# --- Configuration ---
MAX_EPOCHS_GATEKEEPER = 5
MAX_EPOCHS_MAIN = 25
BATCH_SIZE = 128 # Can use a larger batch size for the efficient Gatekeeper
NUM_WORKERS = 4
PATIENCE = 5

def get_transforms():
    """Defines the image transformations."""
    # These are standard transforms for ImageNet-pretrained models
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def run_gatekeeper_training():
    """Sets up and runs the training for the Gatekeeper model using an efficient data loading strategy."""
    print("\n--- PHASE 1: TRAINING THE GATEKEEPER MODEL ---")
    transform = get_transforms()

    # --- EFFICIENT DATA LOADING STRATEGY ---
    # 1. Load datasets without loading images into memory
    skin_dataset = SkinLesionFusionDataset(transform=transform)
    not_skin_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)

    # 2. Create lightweight wrapper datasets to assign the correct labels (1 for skin, 0 for not-skin)
    class LabeledDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, label):
            self.dataset = dataset
            self.label = label
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            # For skin_dataset, we get (rgb_img, dwt_img, label). We only want rgb_img.
            # For CIFAR100, we get (img, label).
            original_data = self.dataset[idx]
            image = original_data[0] # In both cases, the image is the first element
            return image, self.label

    labeled_skin = LabeledDataset(skin_dataset, 1)
    labeled_not_skin = LabeledDataset(not_skin_dataset, 0)

    # 3. Combine them using ConcatDataset, which is memory-efficient
    full_gatekeeper_dataset = ConcatDataset([labeled_skin, labeled_not_skin])
    print(f"Total gatekeeper dataset size: {len(full_gatekeeper_dataset)} images.")
    
    # 4. Split into train and validation
    train_size = int(0.8 * len(full_gatekeeper_dataset))
    val_size = len(full_gatekeeper_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_gatekeeper_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    model = GatekeeperModel()

    gatekeeper_checkpoint = ModelCheckpoint(
        dirpath='checkpoints',
        filename='gatekeeper-best',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=MAX_EPOCHS_GATEKEEPER,
        callbacks=[gatekeeper_checkpoint],
        logger=pl.loggers.TensorBoardLogger('logs/gatekeeper')
    )
    
    trainer.fit(model, train_loader, val_loader)
    print("\n--- Gatekeeper training complete! Best model saved to 'checkpoints/gatekeeper-best.ckpt' ---")


def run_main_model_training():
    """Sets up and runs the training for the main Dual-Brain model."""
    print("\n--- PHASE 2: TRAINING THE MAIN DUAL-BRAIN MODEL (ADVANCED) ---")
    transform = get_transforms()
    full_dataset = SkinLesionFusionDataset(transform=transform)
    
    indices = list(range(len(full_dataset)))
    labels = full_dataset.full_df['target'].values
    
    train_indices, val_indices, _, _ = train_test_split(indices, labels, test_size=0.2, random_state=42, stratify=labels)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # --- Weighted Random Sampler ---
    print("Calculating class weights for balanced training...")
    class_counts = full_dataset.full_df['target'].iloc[train_indices].value_counts().sort_index()
    class_weights = 1. / torch.tensor(class_counts.values, dtype=torch.float)
    sample_weights = class_weights[labels[train_indices]]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    print("WeightedRandomSampler created.")

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

    model = DualBrainModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints_v2',
        filename='best-model-v2',
        save_top_k=1,
        verbose=True,
        monitor='val_accuracy',
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=MAX_EPOCHS_MAIN,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=pl.loggers.TensorBoardLogger('logs/main_model_v2')
    )

    trainer.fit(model, train_loader, val_loader)
    print("\n--- Advanced main model training complete! Best model saved to 'checkpoints_v2' folder. ---")


if __name__ == '__main__':
    # --- NEW: Added argument parsing to remove the need for manual code edits ---
    parser = argparse.ArgumentParser(description="Run training for the skin lesion models.")
    parser.add_argument(
        '--phase', 
        type=str, 
        default='all', 
        choices=['gatekeeper', 'main', 'all'],
        help="Which training phase to run: 'gatekeeper', 'main', or 'all' (default)."
    )
    args = parser.parse_args()

    if args.phase == 'gatekeeper' or args.phase == 'all':
        run_gatekeeper_training()
    
    if args.phase == 'main' or args.phase == 'all':
        run_main_model_training()


import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from torchvision import transforms
import argparse

# Import our FINAL custom classes
from dataset import DefinitiveDataset
from main_model import DefinitiveExpertPanelModel 

# --- Configuration ---
MAX_EPOCHS = 40 # Train for longer on this complex model and large dataset
BATCH_SIZE = 32
NUM_WORKERS = 4
PATIENCE = 7 

def get_transforms():
    """Defines the image transformations."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # All models expect 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def run_definitive_training():
    """Sets up and runs the training for the definitive Expert Panel model on the combined dataset."""
    print("\n--- DEFINITIVE TRAINING ON COMBINED DATASET: THE MULTI-SPECTRUM DIAGNOSTIC PANEL ---")
    transform = get_transforms()
    full_dataset = DefinitiveDataset(transform=transform)
    
    indices = list(range(len(full_dataset)))
    labels = full_dataset.full_df['target'].values
    
    # Stratified split on the combined dataset
    train_indices, val_indices, _, _ = train_test_split(indices, labels, test_size=0.2, random_state=42, stratify=labels)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print("Calculating class weights for balanced training...")
    class_counts = full_dataset.full_df['target'].iloc[train_indices].value_counts().sort_index()
    class_weights = 1. / torch.tensor(class_counts.values, dtype=torch.float)
    sample_weights = class_weights[labels[train_indices]]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    print("WeightedRandomSampler created.")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        num_workers=NUM_WORKERS,
        drop_last=True # Ensure all batches have an even size for Mixup
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = DefinitiveExpertPanelModel(num_tabular_features=full_dataset.tabular_data.shape[1])

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints_definitive', # Save to a new folder
        filename='best-definitive-model',
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
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=pl.loggers.TensorBoardLogger('logs/definitive_panel')
    )

    trainer.fit(model, train_loader, val_loader)
    print("\n--- Definitive model training complete! Best model saved to 'checkpoints_definitive' folder. ---")


if __name__ == '__main__':
    run_definitive_training()


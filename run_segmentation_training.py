import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Import our new segmentation-specific classes
from segmentation_dataset import SegmentationDataset, get_segmentation_transforms
from segmentation_model import LesionSegmenter

# --- Configuration ---
BASE_DIR = 'data/ham10000'
IMAGE_SIZE = 256
BATCH_SIZE = 16 # Segmentation is memory-intensive, so we use a smaller batch size
NUM_WORKERS = 4
MAX_EPOCHS = 30
PATIENCE = 5 # For EarlyStopping

def run_segmentation_training():
    """
    The main function to set up and run the U-Net segmentation training.
    """
    print("--- PHASE 3: TRAINING THE U-NET SEGMENTATION MODEL ---")

    # --- 1. Set up Transforms and Dataset ---
    train_transform, val_transform = get_segmentation_transforms(image_size=IMAGE_SIZE)
    
    full_dataset = SegmentationDataset(base_dir=BASE_DIR, transform=train_transform)
    
    # We will split our dataset into training and validation sets (e.g., 80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Important: The validation set should only use validation transforms (no random flips etc.)
    val_dataset.dataset.transform = val_transform

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # --- 2. Set up DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- 3. Set up Model ---
    model = LesionSegmenter(learning_rate=1e-4)

    # --- 4. Set up Callbacks and Trainer ---
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints_segmentation',
        filename='best-segmenter',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
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
        logger=pl.loggers.TensorBoardLogger('logs/segmentation'),
        log_every_n_steps=10
    )

    # --- 5. Start Training ---
    print("\nStarting U-Net training...")
    trainer.fit(model, train_loader, val_loader)
    print("\n--- U-Net model training complete! Best model saved to 'checkpoints_segmentation' folder. ---")


if __name__ == '__main__':
    run_segmentation_training()

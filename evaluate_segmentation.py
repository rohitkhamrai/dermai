import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import argparse
import os
from tqdm import tqdm
import numpy as np

# NOTE: We have REMOVED the problematic 'torchmetrics' library dependency.

# Import our segmentation-specific classes
from segmentation_dataset import SegmentationDataset, get_segmentation_transforms
from segmentation_model import LesionSegmenter

# --- Configuration ---
CHECKPOINT_PATH = 'checkpoints_segmentation/best-segmenter.ckpt'
IMAGE_SIZE = 256
BATCH_SIZE = 16
BASE_DIR = 'data/ham10000' # Define the base directory for data

def calculate_dice_score(preds, masks, smooth=1e-6):
    """
    Calculates the Dice score for a batch of predictions and masks.
    This is our self-contained implementation, removing the need for torchmetrics.
    """
    # Flatten the tensors
    preds = preds.contiguous().view(preds.shape[0], -1)
    masks = masks.contiguous().view(masks.shape[0], -1)

    # Calculate intersection and union
    intersection = (preds * masks).sum(dim=1)
    union = preds.sum(dim=1) + masks.sum(dim=1)

    # Calculate Dice score
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice.mean()

def evaluate_segmenter(checkpoint_path, force_cpu=False):
    """
    Loads a trained U-Net model and evaluates its performance on the test set
    using our internal Dice score metric.
    """
    # --- 1. Set up device ---
    device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # --- 2. Load Model ---
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
        return

    print(f"Loading model from {checkpoint_path}...")
    model = LesionSegmenter.load_from_checkpoint(checkpoint_path, map_location=device).to(device)
    model.eval()

    # --- 3. Prepare Dataset ---
    print("Preparing test dataset...")
    _, val_transform = get_segmentation_transforms(image_size=IMAGE_SIZE)
    
    full_dataset = SegmentationDataset(base_dir=BASE_DIR, transform=val_transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    _, test_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Test set size: {len(test_dataset)} images.")

    # --- 4. Calculate Dice Score ---
    print("Running evaluation on the entire test set...")
    
    dice_scores = []
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)
            
            # Binarize predictions to 0 or 1
            preds = (preds > 0.5).float()
            
            # Calculate score for the current batch and append it
            score = calculate_dice_score(preds, masks)
            dice_scores.append(score.item())

    # Compute the final Dice score over all batches
    final_dice_score = np.mean(dice_scores)

    # --- 5. Display Report Card ---
    print("\n" + "="*50)
    print("      U-NET SEGMENTATION REPORT CARD")
    print("="*50)
    print(f"\nMetric: Average Dice Score")
    print("Description: Measures the overlap between the predicted mask and")
    print("             the true mask. (0 = no overlap, 1 = perfect overlap)")
    print("-" * 50)
    print(f"\n      Final Dice Score: {final_dice_score:.4f}")
    print("\n" + "="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net segmentation model.")
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH, help="Path to the model checkpoint file.")
    parser.add_argument('--cpu', action='store_true', help="Force CPU use.")
    args = parser.parse_args()
    
    evaluate_segmenter(args.checkpoint, args.cpu)


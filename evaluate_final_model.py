import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from torchvision import transforms
from tqdm import tqdm

# Import our FINAL model and dataset classes
from main_model import ExpertPanelModel
from dataset import MultiModalDataset

import warnings
warnings.filterwarnings("ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*")

def get_transforms():
    """Defines the image transformations for evaluation."""
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return val_transform

def evaluate_final_model(checkpoint_path, force_cpu=False):
    """
    Loads and evaluates the final ExpertPanelModel on the multi-modal test set.
    """
    device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
        return

    print(f"Loading final expert model from {checkpoint_path}...")
    
    # We need the number of tabular features to load the model
    # We can get this by creating a dummy dataset instance first
    dummy_dataset = MultiModalDataset(transform=get_transforms())
    num_tabular_features = dummy_dataset.tabular_data.shape[1]
    
    model = ExpertPanelModel.load_from_checkpoint(
        checkpoint_path, 
        map_location=device,
        num_tabular_features=num_tabular_features
    ).to(device)
    model.eval()

    print("Preparing multi-modal test dataset...")
    
    # Use the same dummy dataset to create our test set
    full_dataset = dummy_dataset
    indices = list(range(len(full_dataset)))
    labels = full_dataset.full_df['target'].values
    
    _, test_indices, _, _ = train_test_split(indices, labels, test_size=0.2, random_state=42, stratify=labels)

    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"Test set size: {len(test_dataset)} images.")

    print("Running predictions on the entire test set...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for rgb, dwt, canny, tabular, labels_batch in tqdm(test_loader, desc="Evaluating"):
            rgb = rgb.to(device)
            dwt = dwt.to(device)
            canny = canny.to(device)
            tabular = tabular.to(device)
            labels_batch = labels_batch.to(device)
            
            final_output, _, _, _ = model(rgb, dwt, canny, tabular)
            _, predicted = torch.max(final_output.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
    
    print("\n" + "="*50)
    print("      FINAL EXPERT PANEL - REPORT CARD (V3)")
    print("="*50)

    rev_lesion_dict = {v: k for k, v in full_dataset.lesion_map.items()}
    class_names = [rev_lesion_dict[i] for i in sorted(rev_lesion_dict.keys())]

    print("\n--- Classification Report ---\n")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
    print(report)

    print("\n--- Generating Final Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Final Expert Panel - Confusion Matrix (V3)')
    
    output_path = 'confusion_matrix_final_v3.png'
    plt.savefig(output_path)
    print(f"\nFinal Confusion Matrix saved to '{output_path}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the final expert panel model.")
    parser.add_argument('--checkpoint', type=str, default='checkpoints_final_v3/best-expert-panel-v3.ckpt', help="Path to the final model checkpoint file.")
    parser.add_argument('--cpu', action='store_true', help="Force CPU use.")
    args = parser.parse_args()
    evaluate_final_model(args.checkpoint, args.cpu)


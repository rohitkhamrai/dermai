import torch
import cv2
import numpy as np
import argparse
import os
from PIL import Image
from torchvision import transforms

# Import our U-Net model and hair removal utility
from segmentation_model import LesionSegmenter
from utils import remove_hair

# --- Configuration ---
IMAGE_SIZE = 256
CHECKPOINT_PATH = 'checkpoints_segmentation/best-segmenter.ckpt'

def predict_mask(image, model, device):
    """
    Takes a PIL image and a trained model, and returns the predicted probability mask.
    (Modified to accept a PIL image directly instead of a path)
    """
    original_size = image.size # (width, height)

    # --- Perform Hair Removal ---
    image_np = np.array(image)
    image_no_hair_np = remove_hair(image_np)
    image_no_hair = Image.fromarray(image_no_hair_np)
    
    # Preprocess the hairless image
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(image_no_hair).unsqueeze(0).to(device)

    # Make a prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Post-process the output mask
    predicted_mask_prob = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Resize the probability mask back to the original image size for the heatmap
    resized_prob_mask = cv2.resize(predicted_mask_prob, original_size, interpolation=cv2.INTER_LINEAR)
    
    return resized_prob_mask

def create_heatmap_overlay(image, prob_mask):
    """
    Overlays the probability mask on the original image as a heatmap.
    (Modified to accept a PIL image directly)
    """
    # Convert PIL image to OpenCV format
    original_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    heatmap = (prob_mask * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    alpha = 0.5
    overlaid_image = cv2.addWeighted(original_image, 1 - alpha, colored_heatmap, alpha, 0)
    
    overlaid_image_rgb = cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(overlaid_image_rgb)

def run_segmentation_on_image(image, model, device):
    """
    A wrapper function for the server to call.
    Takes a PIL image and returns a PIL image of the heatmap overlay.
    """
    prob_mask = predict_mask(image, model, device)
    heatmap_image = create_heatmap_overlay(image, prob_mask)
    return heatmap_image

# The main block is now for standalone testing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a heatmap for a skin lesion using a trained U-Net model.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image file.")
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH, help="Path to the model checkpoint file.")
    parser.add_argument('--cpu', action='store_true', help="Force CPU use.")
    args = parser.parse_args()
    
    # --- 1. Set up device and load model ---
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LesionSegmenter.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)

    # --- 2. Run prediction ---
    input_image = Image.open(args.image_path).convert("RGB")
    final_image = run_segmentation_on_image(input_image, model, device)
    
    output_filename = 'heatmap_output.png'
    final_image.save(output_filename)
    
    print(f"Highlighted heatmap saved as '{output_filename}'.")


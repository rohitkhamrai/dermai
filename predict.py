import torch
from torchvision import transforms
from PIL import Image
import numpy as np
# --- FIX: Add missing imports for cv2 and pywt ---
import cv2
import pywt

from utils import remove_hair

# This dictionary should be consistent with your training data
LESION_TYPE_DICT = {
    0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'
}

def get_dwt_image(image_array):
    """Calculates the DWT of an image and stacks the components."""
    image_array = np.array(image_array)
    img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    LL, (LH, HL, HH) = coeffs2
    
    # Normalize and stack coefficients to create a 3-channel image
    ll_norm = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lh_norm = cv2.normalize(LH, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hl_norm = cv2.normalize(HL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    dwt_stacked = np.stack([ll_norm, lh_norm, hl_norm], axis=-1)
    return dwt_stacked

def process_and_predict(image, model, device, is_gatekeeper=False):
    """
    Processes a single PIL image and returns the prediction from a given model.
    Handles both the gatekeeper and the main classifier.
    """
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_np = np.array(image)
    image_no_hair_np = remove_hair(image_np)
    image_no_hair_pil = Image.fromarray(image_no_hair_np)

    if is_gatekeeper:
        input_tensor = transform(image_no_hair_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            prediction = (predicted_idx.item() == 1) # 1 is 'skin'
            confidence = torch.softmax(output, dim=1).max().item()
            return prediction, f"{confidence*100:.2f}%"

    else: # Main classifier logic
        # Prepare RGB tensor
        rgb_tensor = transform(image_no_hair_pil).unsqueeze(0).to(device)
        
        # Prepare DWT tensor
        dwt_np = get_dwt_image(image_no_hair_np)
        dwt_pil = Image.fromarray(dwt_np)
        dwt_tensor = transform(dwt_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(rgb_tensor, dwt_tensor)
            confidence = torch.softmax(output, dim=1).max().item()
            _, predicted_idx = torch.max(output, 1)
            lesion_type = LESION_TYPE_DICT.get(predicted_idx.item(), "Unknown")
            return lesion_type, f"{confidence*100:.2f}%"


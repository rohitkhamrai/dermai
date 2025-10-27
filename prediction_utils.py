import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import pywt
from utils import remove_hair

# --- Helper Functions to Generate Data Modalities ---
def get_advanced_dwt_image(image_array):
    """Calculates DWT with a more advanced 'db4' wavelet."""
    img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    # Resize for consistent DWT output size
    img_gray_resized = cv2.resize(img_gray, (224, 224))
    coeffs2 = pywt.dwt2(img_gray_resized, 'db4')
    LL, (LH, HL, HH) = coeffs2
    ll_norm = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX)
    lh_norm = cv2.normalize(LH, None, 0, 255, cv2.NORM_MINMAX)
    hl_norm = cv2.normalize(HL, None, 0, 255, cv2.NORM_MINMAX)
    return np.stack([ll_norm, lh_norm, hl_norm], axis=-1).astype(np.uint8)

def get_canny_image(image_array):
    """Generates the Canny Edge Map."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 50, 150)
    return np.stack([canny, canny, canny], axis=-1).astype(np.uint8)

def get_clahe_image(image_array):
    """Generates the CLAHE enhanced image."""
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final_rgb.astype(np.uint8)

def run_segmentation_on_image(image_pil, model, device):
    """Generates a heatmap for a given PIL image."""
    image_np = np.array(image_pil)
    image_no_hair_np = remove_hair(image_np)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image_no_hair_np).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    resized_prob_mask = cv2.resize(prob_mask, image_pil.size, interpolation=cv2.INTER_LINEAR)
    heatmap = (resized_prob_mask * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original_image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    overlaid_image = cv2.addWeighted(original_image_bgr, 0.6, colored_heatmap, 0.4, 0)
    overlaid_image_rgb = cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlaid_image_rgb)

# --- The Single, Definitive Prediction Function ---
def predict_with_definitive_model(image: Image.Image, model, device, tabular_processor, lesion_map, age: int, sex: str, localization: str):
    """Runs prediction using the Definitive Expert Panel model."""
    # 1. Process Tabular Data (using provided arguments)
    age_np = np.array([[age]])
    age_scaled = tabular_processor['age_scaler'].transform(age_np)
    
    sex_np = np.array([[sex]])
    sex_encoded = tabular_processor['sex_encoder'].transform(sex_np)
    
    loc_np = np.array([[localization]])
    loc_encoded = tabular_processor['loc_encoder'].transform(loc_np)
    
    tabular_features = np.hstack([age_scaled, sex_encoded, loc_encoded])
    tabular_tensor = torch.tensor(tabular_features, dtype=torch.float32).to(device)

    # 2. Process Image Data
    image_np = np.array(image)
    image_no_hair_np = remove_hair(image_np)
    
    rgb_pil = Image.fromarray(image_no_hair_np)
    dwt_pil = Image.fromarray(get_advanced_dwt_image(image_no_hair_np))
    canny_pil = Image.fromarray(get_canny_image(image_no_hair_np))
    clahe_pil = Image.fromarray(get_clahe_image(image_no_hair_np))
    vit_pil = rgb_pil
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    rgb_tensor = transform(rgb_pil).unsqueeze(0).to(device)
    dwt_tensor = transform(dwt_pil).unsqueeze(0).to(device)
    canny_tensor = transform(canny_pil).unsqueeze(0).to(device)
    clahe_tensor = transform(clahe_pil).unsqueeze(0).to(device)
    vit_tensor = transform(vit_pil).unsqueeze(0).to(device)

    # 3. Make prediction
    with torch.no_grad():
        final_output = model(rgb_tensor, dwt_tensor, canny_tensor, clahe_tensor, vit_tensor, tabular_tensor)
        
    probabilities = F.softmax(final_output, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_label = lesion_map.get(predicted_idx.item(), "Unknown")
    return predicted_label, confidence.item()
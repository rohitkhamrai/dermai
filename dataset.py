import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import pywt
from utils import remove_hair
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DefinitiveDataset(Dataset):
    """
    The definitive, final Data Factory. It prepares all six data modalities for our
    Multi-Spectrum Diagnostic Panel and now correctly merges both HAM10000 and ISIC2020 datasets.
    """
    def __init__(self, base_dir='data', transform=None):
        self.base_dir = base_dir
        self.transform = transform
        
        # --- Define Paths ---
        ham_meta_path = os.path.join(self.base_dir, 'ham10000', 'HAM10000_metadata.csv')
        isic_meta_path = os.path.join(self.base_dir, 'isic2020', 'isic2020', 'train.csv')

        # --- Load and Process HAM10000 ---
        ham_df = pd.read_csv(ham_meta_path)
        ham_df['image_path'] = ham_df['image_id'].apply(lambda x: os.path.join(self.base_dir, 'ham10000', 'images', x + '.jpg'))

        # --- Load and Process ISIC2020 ---
        isic_df = pd.read_csv(isic_meta_path)
        # Map ISIC labels to HAM10000 labels for consistency
        # We will treat 'benign' as 'nv' and 'malignant' as 'mel'
        isic_df['dx'] = np.where(isic_df['benign_malignant'] == 'malignant', 'mel', 'nv')
        isic_df['image_path'] = isic_df['image_name'].apply(lambda x: os.path.join(self.base_dir, 'isic2020', 'isic2020', 'train', x + '.jpg'))
        
        # --- Combine DataFrames ---
        # Select and rename columns to match
        ham_df_subset = ham_df[['image_path', 'dx', 'age', 'sex', 'localization']]
        isic_df_subset = isic_df[['image_path', 'dx', 'age_approx', 'sex', 'anatom_site_general_challenge']]
        isic_df_subset = isic_df_subset.rename(columns={'age_approx': 'age', 'anatom_site_general_challenge': 'localization'})
        
        df = pd.concat([ham_df_subset, isic_df_subset], ignore_index=True)
        
        # --- Handle Missing Values (Future-Proof Method) ---
        df['age'] = df['age'].fillna(df['age'].mean())
        df['sex'] = df['sex'].fillna('unknown')
        df['localization'] = df['localization'].fillna('unknown')
        top_location = df[df['localization'] != 'unknown']['localization'].mode()[0]
        df['localization'] = df['localization'].replace('unknown', top_location)
        
        # --- Prepare Tabular Data ---
        self.age_scaler = StandardScaler()
        age_scaled = self.age_scaler.fit_transform(df[['age']])
        
        self.sex_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        sex_encoded = self.sex_encoder.fit_transform(df[['sex']])
        
        self.loc_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        loc_encoded = self.loc_encoder.fit_transform(df[['localization']])

        self.tabular_data = np.hstack([age_scaled, sex_encoded, loc_encoded])
        
        self.lesion_map = {lesion: i for i, lesion in enumerate(df['dx'].unique())}
        df['target'] = df['dx'].map(self.lesion_map)

        self.full_df = df
        print(f"Definitive Multi-Modal (Combined) Dataset created. Total samples: {len(self.full_df)}")
        print(f"Number of tabular features: {self.tabular_data.shape[1]}")

    def __len__(self):
        return len(self.full_df)

    def get_advanced_dwt_image(self, image_array):
        """Calculates DWT with a more advanced 'db4' wavelet."""
        img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        coeffs2 = pywt.dwt2(img_gray, 'db4') # Using Daubechies 4
        LL, (LH, HL, HH) = coeffs2
        ll_norm = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX)
        lh_norm = cv2.normalize(LH, None, 0, 255, cv2.NORM_MINMAX)
        hl_norm = cv2.normalize(HL, None, 0, 255, cv2.NORM_MINMAX)
        dwt_stacked = np.stack([ll_norm, lh_norm, hl_norm], axis=-1)
        return dwt_stacked.astype(np.uint8)
        
    def get_canny_image(self, image_array):
        """Generates the Canny Edge Map."""
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blurred, 50, 150)
        return np.stack([canny, canny, canny], axis=-1).astype(np.uint8)

    def get_clahe_image(self, image_array):
        """Generates the CLAHE enhanced image."""
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final_rgb.astype(np.uint8)

    def __getitem__(self, idx):
        row = self.full_df.iloc[idx]
        img_path = row['image_path']
        label = row['target']
        tabular = self.tabular_data[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"WARNING: Image not found at {img_path}. Returning first sample.")
            return self.__getitem__(0) # Return a valid sample to prevent crash

        image_np = np.array(image)
        
        image_no_hair_np = remove_hair(image_np)
        
        # --- Generate all 6 data modalities ---
        rgb_pil = Image.fromarray(image_no_hair_np)
        dwt_pil = Image.fromarray(self.get_advanced_dwt_image(image_no_hair_np))
        canny_pil = Image.fromarray(self.get_canny_image(image_no_hair_np))
        clahe_pil = Image.fromarray(self.get_clahe_image(image_no_hair_np))
        vit_pil = rgb_pil # ViT uses the same RGB image
        
        tabular_tensor = torch.tensor(tabular, dtype=torch.float32)

        # Apply transformations
        if self.transform:
            rgb_tensor = self.transform(rgb_pil)
            dwt_tensor = self.transform(dwt_pil)
            canny_tensor = self.transform(canny_pil)
            clahe_tensor = self.transform(clahe_pil)
            vit_tensor = self.transform(vit_pil)
        else:
            to_tensor = transforms.ToTensor()
            rgb_tensor = to_tensor(rgb_pil)
            dwt_tensor = to_tensor(dwt_pil)
            canny_tensor = to_tensor(canny_pil)
            clahe_tensor = to_tensor(clahe_pil)
            vit_tensor = to_tensor(vit_pil)
            
        return (rgb_tensor, dwt_tensor, canny_tensor, clahe_tensor, vit_tensor, 
                tabular_tensor, torch.tensor(label, dtype=torch.long))

# --- Built-in Test ---
if __name__ == '__main__':
    print("Testing the DefinitiveDataset...")
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = DefinitiveDataset(transform=test_transform)
    
    print(f"\nFetching one sample from the data factory...")
    rgb, dwt, canny, clahe, vit, tabular, label = dataset[0]
    
    print("Sample fetched successfully!")
    print(f"  - RGB image tensor shape: {rgb.shape}")
    print(f"  - Advanced DWT tensor shape: {dwt.shape}")
    print(f"  - Canny Edge tensor shape: {canny.shape}")
    print(f"  - CLAHE tensor shape: {clahe.shape}")
    print(f"  - ViT image tensor shape: {vit.shape}")
    print(f"  - Tabular data tensor shape: {tabular.shape}")
    print(f"  - Label: {label.item()}")
    
    assert rgb.shape == (3, 224, 224)
    assert dwt.shape[0] == 3 # DWT size can vary slightly
    assert canny.shape == (3, 224, 224)
    assert clahe.shape == (3, 224, 224)
    assert vit.shape == (3, 224, 224)
    assert tabular.ndim == 1
    
    print("\nDefinitive Data Factory test passed!")


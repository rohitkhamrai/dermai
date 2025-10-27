import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from timm.data.mixup import Mixup
from utils import FocalLoss
import torchmetrics
# --- FIX: Import the dataset class needed for the built-in test ---
from dataset import DefinitiveDataset

class DefinitiveExpertPanelModel(pl.LightningModule):
    """
    The definitive, final model architecture. It features six expert 'brains'
    and a Transformer-based fusion mechanism to force consensus.
    """
    def __init__(self, num_classes=7, num_tabular_features=22, learning_rate=1e-4, mixup_alpha=0.2):
        super().__init__()
        self.save_hyperparameters()

        # --- Brains (The Experts) ---
        self.brain_rgb = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.brain_dwt = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.brain_canny = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.brain_clahe = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.brain_vit = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        self.brain_tabular = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 192) # Match ViT's feature dimension
        )
        
        common_dim = 192 # ViT's dimension
        self.project_rgb = nn.Linear(self.brain_rgb.num_features, common_dim)
        self.project_dwt = nn.Linear(self.brain_dwt.num_features, common_dim)
        self.project_canny = nn.Linear(self.brain_canny.num_features, common_dim)
        self.project_clahe = nn.Linear(self.brain_clahe.num_features, common_dim)

        # --- Fusion Mechanism (The Moderated Debate) ---
        encoder_layer = nn.TransformerEncoderLayer(d_model=common_dim, nhead=6, dim_feedforward=768, dropout=0.1, batch_first=True)
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # --- Final Classifier (The Chief of Medicine's Decision) ---
        self.final_classifier = nn.Linear(common_dim, num_classes)
        
        self.criterion = FocalLoss()
        self.mixup = Mixup(mixup_alpha=mixup_alpha, cutmix_alpha=0., prob=0.5, num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, common_dim))

    def forward(self, rgb, dwt, canny, clahe, vit, tabular):
        # 1. Get initial features from all brains
        features_rgb = self.brain_rgb(rgb)
        features_dwt = self.brain_dwt(dwt)
        features_canny = self.brain_canny(canny)
        features_clahe = self.brain_clahe(clahe)
        features_vit = self.brain_vit(vit)
        features_tabular = self.brain_tabular(tabular)

        # 2. Project all image features to the common dimension
        rgb_proj = self.project_rgb(features_rgb)
        dwt_proj = self.project_dwt(features_dwt)
        canny_proj = self.project_canny(features_canny)
        clahe_proj = self.project_clahe(features_clahe)
        
        # 3. Form a sequence of "expert opinions"
        batch_size = rgb.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Sequence: [CLS, RGB, DWT, Canny, CLAHE, ViT, Tabular]
        token_sequence = torch.stack([
            cls_tokens, 
            rgb_proj.unsqueeze(1), 
            dwt_proj.unsqueeze(1), 
            canny_proj.unsqueeze(1), 
            clahe_proj.unsqueeze(1),
            features_vit.unsqueeze(1), 
            features_tabular.unsqueeze(1)
        ], dim=1).squeeze(2)

        # 4. Run the attention-based fusion
        transformer_output = self.fusion_transformer(token_sequence)
        
        # 5. The final, consensus-based feature is the output of the CLS token
        consensus_features = transformer_output[:, 0]
        
        # 6. Get the final panel decision
        final_output = self.final_classifier(consensus_features)
        
        return final_output

    def training_step(self, batch, batch_idx):
        rgb, dwt, canny, clahe, vit, tabular, targets = batch
        
        # Apply Mixup to images and targets
        rgb, targets_mixed = self.mixup(rgb, targets)
        dwt, _ = self.mixup(dwt, targets)
        canny, _ = self.mixup(canny, targets)
        clahe, _ = self.mixup(clahe, targets)
        vit, _ = self.mixup(vit, targets)

        final_output = self(rgb, dwt, canny, clahe, vit, tabular)
        
        loss = self.criterion(final_output, targets_mixed)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        rgb, dwt, canny, clahe, vit, tabular, targets = batch
        final_output = self(rgb, dwt, canny, clahe, vit, tabular)
        loss = self.criterion(final_output, targets)
        
        self.log('val_loss', loss, prog_bar=True)
        self.val_accuracy(final_output, targets)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

# --- Built-in Test ---
if __name__ == '__main__':
    print("Testing the DefinitiveExpertPanelModel...")
    # Get the number of tabular features from the dataset
    dataset_for_test = DefinitiveDataset()
    num_tabular_features = dataset_for_test.tabular_data.shape[1]

    model = DefinitiveExpertPanelModel(num_tabular_features=num_tabular_features)
    
    # Create dummy inputs for all modalities
    dummy_rgb = torch.randn(2, 3, 224, 224)
    # The DWT output is resized by the transform, so we can use a standard size here for the test
    dummy_dwt = torch.randn(2, 3, 224, 224) 
    dummy_canny = torch.randn(2, 3, 224, 224)
    dummy_clahe = torch.randn(2, 3, 224, 224)
    dummy_vit = torch.randn(2, 3, 224, 224)
    dummy_tabular = torch.randn(2, num_tabular_features)
    
    # Pass dummy inputs through the model
    final_output = model(dummy_rgb, dummy_dwt, dummy_canny, dummy_clahe, dummy_vit, dummy_tabular)
    print(f"Model final output shape: {final_output.shape}")

    assert final_output.shape == (2, 7)
    print("\nDefinitive Expert Panel Model test passed!")


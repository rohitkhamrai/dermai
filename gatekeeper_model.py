import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm

class GatekeeperModel(pl.LightningModule):
    """
    A simple binary classifier to determine if an image is of skin or not.
    Uses a lightweight, pre-trained MobileNetV3 for speed.
    ROBUST VERSION: Dynamically determines feature size to prevent shape errors.
    """
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=0)
        
        # --- FIX: Determine the number of features dynamically and robustly ---
        with torch.no_grad():
            # Create a dummy input tensor of the same size as our images
            dummy_input = torch.randn(1, 3, 224, 224)
            # Pass it through the backbone to see the output shape
            features = self.backbone(dummy_input)
            # The number of features is the second dimension (shape is [batch, features])
            num_features = features.shape[1]

        # Freeze the backbone layers so we only train the final layer
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Define our new classifier head using the dynamically found number of features
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2) # 2 output classes: 0 for 'not_skin', 1 for 'skin'
        )
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=self.hparams.learning_rate)

# --- Built-in Test ---
if __name__ == '__main__':
    print("Testing the GatekeeperModel...")
    model = GatekeeperModel()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"Dummy input shape: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")

    assert output.shape == (1, 2)
    print("Gatekeeper model test passed!")


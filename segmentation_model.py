import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

class LesionSegmenter(pl.LightningModule):
    """
    The PyTorch Lightning module for the U-Net segmentation model.
    FINAL CORRECTED VERSION: Fixes a data type mismatch in the loss function.
    """
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None # We will use BCEWithLogitsLoss, which prefers raw logits
        )

        self.loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        
        # --- FIX ---
        # The mask tensor must be converted to a float to match the output type
        masks = masks.float()

        dice_loss = self.loss_fn(outputs, masks)
        bce_loss_val = self.bce_loss(outputs, masks)
        loss = dice_loss + bce_loss_val
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        
        # --- FIX ---
        # The mask tensor must be converted to a float to match the output type
        masks = masks.float()

        dice_loss = self.loss_fn(outputs, masks)
        bce_loss_val = self.bce_loss(outputs, masks)
        val_loss = dice_loss + bce_loss_val

        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

# --- Built-in Test ---
if __name__ == '__main__':
    print("Testing the LesionSegmenter (U-Net) model...")
    # NOTE: The test doesn't use the loss function, so it would pass regardless.
    # The fix is for the actual training process.
    model = LesionSegmenter()
    
    dummy_input = torch.randn(2, 3, 256, 256)
    print(f"Dummy input shape: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")

    assert output.shape == (2, 1, 256, 256)
    print("U-Net model test passed!")


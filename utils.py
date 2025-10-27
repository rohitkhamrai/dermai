import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

def remove_hair(image: np.ndarray) -> np.ndarray:
    """
    Removes hair from a skin image using morphological operations.
    This is a crucial preprocessing step to reduce noise.
    Args:
        image (np.ndarray): The input image in BGR format (as read by OpenCV).
    Returns:
        np.ndarray: The image with hair removed.
    """
    # Convert image to grayscale for hair detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    
    # Apply blackhat filtering to find dark hair on a lighter background
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Intensify the hair detection
    _, thresholded = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Use inpainting to "paint over" the detected hair
    # The inpainting algorithm fills the detected regions based on surrounding pixels.
    inpainted_image = cv2.inpaint(image, thresholded, 3, cv2.INPAINT_TELEA)
    
    return inpainted_image

class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss to address class imbalance.
    This is our "Expert Trainer" method.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: model predictions (logits) of shape [N, C]
            targets: ground truth labels of shape [N]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

if __name__ == '__main__':
    # This block allows us to test the file directly.
    # It will not run when we import these functions elsewhere.
    print("Testing utility functions...")
    
    # Test the hair removal function
    print("Testing hair removal function...")
    # Create a dummy image with a black line (simulating hair)
    dummy_image = np.full((200, 200, 3), (180, 150, 120), dtype=np.uint8) # A skin-like color
    cv2.line(dummy_image, (10, 10), (190, 190), (0, 0, 0), 2) # Black line for hair
    
    # Save before and after for visual inspection
    cv2.imwrite("hair_before.png", dummy_image)
    inpainted = remove_hair(dummy_image)
    cv2.imwrite("hair_after.png", inpainted)
    print("Hair removal test complete. Check your file explorer for 'hair_before.png' and 'hair_after.png'.")

    # Test Focal Loss
    print("\nTesting Focal Loss...")
    loss_fn = FocalLoss()
    dummy_logits = torch.randn(16, 7) # Batch of 16, 7 classes
    dummy_labels = torch.randint(0, 7, (16,))
    loss = loss_fn(dummy_logits, dummy_labels)
    print(f"Focal Loss calculated successfully: {loss.item()}")


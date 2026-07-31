import cv2
import numpy as np
from rembg import remove

# Load test image
image = cv2.imread("/home/lexicon/Downloads/Watermark/Watermark-1.webp")
print(f"Image shape: {image.shape}")

# Remove background (this uses U2-Net internally)
result = remove(image)
print(f"Result shape: {result.shape}, dtype: {result.dtype}")

# Save result
cv2.imwrite("result_full.png", result)

# Extract just the alpha channel (this is your soft mask!)
if result.shape[2] == 4:
    alpha_mask = result[:, :, 3]  # RGBA format
    print(f"Alpha mask range: {alpha_mask.min()} to {alpha_mask.max()}")
    print(f"Alpha mask mean: {alpha_mask.mean()}")
    
    # Save the soft alpha mask
    cv2.imwrite("alpha_soft.png", alpha_mask)
    
    # Create thresholded versions
    cv2.imwrite("alpha_thresh_30.png", ((alpha_mask > 30) * 255).astype(np.uint8))
    cv2.imwrite("alpha_thresh_127.png", ((alpha_mask > 127) * 255).astype(np.uint8))
    
    print("Saved: alpha_soft.png, alpha_thresh_30.png, alpha_thresh_127.png")
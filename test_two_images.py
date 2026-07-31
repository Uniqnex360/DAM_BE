import sys
sys.path.append('/home/lexicon/Documents/Harisankar/Projects/DAM/DAM-Backend/app/services/image_processing/steps')

from watermark_removal import WatermarkRemovalStep
from PIL import Image
import numpy as np
import cv2
import os

print("Loading system...")
engine = WatermarkRemovalStep()
print("System loaded.\n")

test_dir = "/home/lexicon/Downloads/Watermark"
output_dir = "/home/lexicon/Documents/Harisankar/Projects/DAM/DAM-Backend/test_results"
os.makedirs(output_dir, exist_ok=True)

for i in [4, 5]:
    input_path = f"{test_dir}/Watermark-{i}.webp"
    
    print(f"{'='*60}")
    print(f"Testing: Watermark-{i}.webp")
    print('='*60)
    
    if not os.path.exists(input_path):
        print(f"File not found")
        continue
    
    image = Image.open(input_path).convert("RGB")
    print(f"Image size: {image.size}")
    
    mask = engine.auto_detect_mask(image)
    mask_np = np.array(mask)
    
    mask_path = f"{output_dir}/mask_{i}.png"
    cv2.imwrite(mask_path, mask_np)
    
    coverage = (mask_np > 0).mean()
    print(f"Mask coverage: {coverage*100:.2f}%")
    
    result = engine.process(image)
    result_path = f"{output_dir}/result_{i}.png"
    cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Saved: {result_path}")
    print()

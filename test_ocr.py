import cv2
import easyocr

# Initialize EasyOCR
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR loaded.\n")

# Test on all 5 images
for i in range(1, 6):
    image_path = f"/home/lexicon/Downloads/Watermark/Watermark-{i}.webp"
    
    print(f"{'='*60}")
    print(f"Image: Watermark-{i}.webp")
    print('='*60)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        continue
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Run EasyOCR
    result = reader.readtext(image_path)
    
    if result:
        print(f"Detected {len(result)} text regions:")
        for idx, (bbox, text, conf) in enumerate(result, 1):
            # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x1, y1 = int(min(x_coords)), int(min(y_coords))
            x2, y2 = int(max(x_coords)), int(max(y_coords))
            print(f"  {idx}. '{text}' (conf: {conf:.2f}) at ({x1},{y1})-({x2},{y2})")
    else:
        print("No text detected")
    
    print()

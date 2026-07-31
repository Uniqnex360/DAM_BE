import cv2
import easyocr
import numpy as np

class WatermarkDetector:
    def __init__(self):
        print("Loading EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR loaded.")
        self.watermark_keywords = [
            'aks sportswear', 'aksport', 'aks wear',
            'ling lux', 'linglux',
            'uncle slam', 'uncleslam',
            'watermark', 'sample', 'preview'
        ]
    
    def detect_watermarks(self, image_path, min_confidence=0.15, max_confidence=0.85):
        image = cv2.imread(image_path)
        if image is None:
            return []
        h, w = image.shape[:2]
        results = self.reader.readtext(image_path)
        watermarks = []
        for bbox, text, conf in results:
            text_lower = text.lower().strip()
            if not (min_confidence <= conf <= max_confidence):
                continue
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x1, y1 = int(min(x_coords)), int(min(y_coords))
            x2, y2 = int(max(x_coords)), int(max(y_coords))
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            image_area = w * h
            coverage = box_area / image_area
            if coverage < 0.01:
                continue
            is_known_watermark = any(kw in text_lower for kw in self.watermark_keywords)
            center_y = (y1 + y2) / 2
            is_bottom_or_center = center_y > h * 0.15
            is_watermark = is_known_watermark or (
                coverage > 0.05 and conf < 0.5 and is_bottom_or_center
            )
            if is_watermark:
                watermarks.append({
                    'text': text,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'area': box_area,
                    'coverage': coverage,
                    'is_known': is_known_watermark
                })
                print(f"  WATERMARK: '{text}' (conf: {conf:.2f}, area: {coverage:.1%})")
            else:
                print(f"  Product: '{text}' (conf: {conf:.2f}, area: {coverage:.1%})")
        return watermarks


def test_detector():
    detector = WatermarkDetector()
    for i in range(1, 6):
        image_path = f"/home/lexicon/Downloads/Watermark/Watermark-{i}.webp"
        print(f"\n{'='*60}")
        print(f"Image: Watermark-{i}.webp")
        print('='*60)
        watermarks = detector.detect_watermarks(image_path)
        if watermarks:
            print(f"\nFound {len(watermarks)} watermark(s)")
        else:
            print("\nNo watermarks detected")
        print()

if __name__ == "__main__":
    test_detector()

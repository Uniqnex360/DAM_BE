import os
import sys
import logging
import traceback
import urllib.request
import hashlib
from datetime import datetime
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from simple_lama_inpainting import SimpleLama

logger = logging.getLogger("WatermarkEngine")


class AutoTemplateBuilder:
    """Automatically builds and manages watermark templates from detections."""
    
    def __init__(self, template_dir="auto_templates/"):
        self.template_dir = template_dir
        os.makedirs(template_dir, exist_ok=True)
        self.template_index = self._load_existing_templates()
        logger.info(f"AutoTemplateBuilder initialized with {len(self.template_index)} existing templates")
    
    def _load_existing_templates(self):
        """Load existing templates and their metadata."""
        index = {}
        if os.path.exists(self.template_dir):
            for filename in os.listdir(self.template_dir):
                if filename.endswith('.png'):
                    filepath = os.path.join(self.template_dir, filename)
                    index[filename] = {
                        'path': filepath,
                        'created': datetime.fromtimestamp(os.path.getctime(filepath)),
                        'usage_count': 0
                    }
        return index
    
    def save_template(self, image, bbox, watermark_type="auto"):
        """Extract a watermark region and save it as a template."""
        x1, y1, x2, y2 = bbox
        
        # Add padding
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1], x2 + pad)
        y2 = min(image.shape[0], y2 + pad)
        
        # Extract region
        template = image[y1:y2, x1:x2]
        
        # Quality checks
        if not self._is_good_template(template):
            return None
        
        # Check if similar template already exists
        if self.find_similar_template(image, bbox):
            return None
        
        # Generate unique filename
        template_hash = hashlib.md5(template.tobytes()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{watermark_type}_{template_hash}_{timestamp}.png"
        filepath = os.path.join(self.template_dir, filename)
        
        cv2.imwrite(filepath, template)
        self.template_index[filename] = {
            'path': filepath,
            'created': datetime.now(),
            'usage_count': 0
        }
        
        logger.info(f"✓ Auto-saved template: {filename}")
        return filepath
    
    def _is_good_template(self, template):
        """Check if template is worth saving."""
        if template.shape[0] < 50 or template.shape[1] < 50:
            return False
        if template.shape[0] > 800 or template.shape[1] > 800:
            return False
        
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        std = np.std(gray)
        if std < 20:
            return False
        
        return True
    
    def find_similar_template(self, image, bbox):
        """Check if this watermark matches an existing template."""
        if not self.template_index:
            return None
        
        x1, y1, x2, y2 = bbox
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return None
        
        best_match = None
        best_score = 0.6  # Minimum threshold
        
        for filename, info in self.template_index.items():
            template = cv2.imread(info['path'])
            if template is None:
                continue
            
            # Resize template to match region size
            template_resized = cv2.resize(template, (region.shape[1], region.shape[0]))
            
            region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY) if len(template_resized.shape) == 3 else template_resized
            
            if region_gray.shape != template_gray.shape:
                continue
            
            # Compare using normalized cross-correlation
            result = cv2.matchTemplate(region_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            score = result.max()
            
            if score > best_score:
                best_score = score
                best_match = info['path']
                self.template_index[filename]['usage_count'] += 1
        
        return best_match


class WatermarkRemovalStep:
    DEFAULT_WEIGHTS_URL = "https://pub-1039b7ab1ee541c1a1f5ff68ddc309ce.r2.dev/best_watermark_model_mit_b5_best.pth"

    def __init__(self, device: str = None, weights_path: str = None):
        try:
            self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Initializing Watermark Engine on device: {self.device}")

            self.segmentor = smp.Unet(
                encoder_name="mit_b5",
                encoder_weights=None,
                in_channels=3,
                classes=1
            ).to(self.device)

            self.weights_path = weights_path or os.path.join(os.path.expanduser("~"), ".cache", "watermark_mit_b5.pth")
            self._ensure_weights_exist()

            if os.path.exists(self.weights_path):
                state_dict = torch.load(self.weights_path, map_location=self.device)
                new_state_dict = {(k[6:] if k.startswith("model.") else k): v for k, v in state_dict.items()}
                self.segmentor.load_state_dict(new_state_dict, strict=False)

            self.segmentor.eval()
            self.inpainter = SimpleLama()

            self.norm_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Initialize auto-template builder
            self.template_builder = AutoTemplateBuilder("auto_templates/")

        except Exception as e:
            logger.critical(f"Initialization error: {e}\n{traceback.format_exc()}")

    def _ensure_weights_exist(self):
        if not os.path.exists(self.weights_path):
            try:
                os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
                urllib.request.urlretrieve(self.DEFAULT_WEIGHTS_URL, self.weights_path)
            except Exception as e:
                logger.error(f"Failed to download weights: {e}")

    def _is_mockup_image(self, image_np: np.ndarray) -> bool:
        try:
            h, w, _ = image_np.shape
            top_border = image_np[0:int(h*0.05), :]
            bottom_border = image_np[int(h*0.95):, :]
            top_mean = np.mean(top_border)
            bottom_mean = np.mean(bottom_border)
            return top_mean > 210 or bottom_mean > 210
        except Exception:
            return False

    def _extract_translucent_overlay_mask(self, image_np: np.ndarray) -> np.ndarray:
        try:
            h, w, _ = image_np.shape
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
            tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
            
            _, binary = cv2.threshold(tophat, 22, 255, cv2.THRESH_BINARY)
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
            filtered_mask = np.zeros_like(binary)
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                w_box = stats[i, cv2.CC_STAT_WIDTH]
                h_box = stats[i, cv2.CC_STAT_HEIGHT]
                
                if 8 < area < 5000 and (w_box < 200 and h_box < 200):
                    filtered_mask[labels == i] = 255

            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            final_mask = cv2.dilate(filtered_mask, dilate_kernel, iterations=2)
            
            mask_coverage = (final_mask > 0).mean()
            if mask_coverage > 0.35:
                logger.warning(f"⚠️ TopHat mask explosion ({mask_coverage*100:.1f}%). Fallback to empty mask.")
                return np.zeros((h, w), dtype=np.uint8)

            return final_mask

        except Exception as e:
            logger.error(f"Error in overlay mask: {e}")
            return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

    def _extract_mockup_watermark_mask(self, image_np: np.ndarray, prob_map: np.ndarray) -> np.ndarray:
        try:
            h, w, _ = image_np.shape
            mask = np.zeros((h, w), dtype=np.uint8)

            protection_zone = np.zeros((h, w), dtype=np.uint8)
            margin_x, margin_y = int(w * 0.15), int(h * 0.15)
            protection_zone[margin_y:h - margin_y, margin_x:w - margin_x] = 255

            raw_mask = (prob_map > 0.40).astype(np.uint8) * 255
            
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            
            is_colorful_graphic = (saturation > 60) & (protection_zone == 255)
            raw_mask[is_colorful_graphic] = 0

            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.dilate(raw_mask, dilate_kernel, iterations=1)
            return mask

        except Exception as e:
            logger.error(f"Error in mockup mask: {e}")
            return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

    def _detect_corner_watermarks(self, image_np: np.ndarray) -> np.ndarray:
        """Detect small watermarks in corners."""
        try:
            h, w, _ = image_np.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            corner_size_x = int(w * 0.4)
            corner_size_y = int(h * 0.4)
            corners = [
                (w - corner_size_x, h - corner_size_y, w, h, "bottom-right"),
                (0, h - corner_size_y, corner_size_x, h, "bottom-left"),
                (w - corner_size_x, 0, w, corner_size_y, "top-right"),
                (0, 0, corner_size_x, corner_size_y, "top-left"),
            ]
            for cx1, cy1, cx2, cy2, corner_name in corners:
                if cx2 - cx1 < 50 or cy2 - cy1 < 50:
                    continue
                roi = image_np[cy1:cy2, cx1:cx2]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                adaptive = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
                tophat = cv2.morphologyEx(roi_gray, cv2.MORPH_TOPHAT, kernel)
                _, tophat_binary = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
                combined = cv2.bitwise_or(adaptive, tophat_binary)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined)
                corner_mask = np.zeros_like(combined)
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    w_box = stats[i, cv2.CC_STAT_WIDTH]
                    h_box = stats[i, cv2.CC_STAT_HEIGHT]
                    if 30 < area < 5000 and 8 < w_box < 300 and 8 < h_box < 300:
                        aspect = w_box / max(h_box, 1)
                        if 0.2 < aspect < 8.0:
                            corner_mask[labels == i] = 255
                dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                corner_mask = cv2.dilate(corner_mask, dilate_kernel, iterations=2)
                mask[cy1:cy2, cx1:cx2] = corner_mask
            return mask
        except Exception as e:
            logger.error(f"Error in corner detection: {e}")
            return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

    def auto_detect_mask(self, image: Image.Image) -> Image.Image:
        try:
            image_np = np.array(image.convert("RGB"))
            w, h = image.size

            # Pass 1: MIT-B5
            tensor = self.norm_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prob = torch.sigmoid(self.segmentor(tensor)).squeeze().cpu().numpy()
            prob_resized = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

            # Get primary mask
            if self._is_mockup_image(image_np):
                logger.info("Pass 1: Mockup pipeline")
                mask1 = self._extract_mockup_watermark_mask(image_np, prob_resized)
            else:
                logger.info("Pass 1: Photo pipeline")
                mask1 = self._extract_translucent_overlay_mask(image_np)

            # Pass 2: Corner detection
            logger.info("Pass 2: Corner detection")
            mask2 = self._detect_corner_watermarks(image_np)

            # Combine masks
            combined_mask = cv2.bitwise_or(mask1, mask2)
            
            # Safety check
            coverage = (combined_mask > 0).mean()
            logger.info(f"Combined coverage: {coverage*100:.2f}%")
            
            if coverage > 0.4:
                logger.warning(f"Mask too large, using only MIT-B5")
                combined_mask = mask1

            # AUTO-SAVE: Save detected regions as templates
            if combined_mask.any():
                self._auto_save_templates(image_np, combined_mask)

            return Image.fromarray(combined_mask)

        except Exception as e:
            logger.error(f"Error: {e}")
            return Image.new("L", image.size, 0)
    
    def _auto_save_templates(self, image_np, mask):
        """Automatically save detected watermark regions as templates."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            if area > 1000:
                self.template_builder.save_template(
                    image_np, 
                    (x, y, x+w, y+h),
                    watermark_type="auto"
                )

    def process(self, image_input, original_img=None, output_path=None) -> np.ndarray:
        loaded_img = None
        try:
            target_img = image_input if image_input is not None else original_img
            if isinstance(target_img, np.ndarray):
                loaded_img = Image.fromarray(target_img).convert("RGB")
            elif isinstance(target_img, Image.Image):
                loaded_img = target_img.convert("RGB")
            elif isinstance(target_img, str) and os.path.exists(target_img):
                loaded_img = Image.open(target_img).convert("RGB")

            predicted_mask = self.auto_detect_mask(loaded_img)
            cleaned_result = self.inpainter(loaded_img, predicted_mask)

            if output_path and isinstance(output_path, str):
                cleaned_result.save(output_path)

            return np.array(cleaned_result)

        except Exception as e:
            logger.error(f"Process error: {e}")
            if loaded_img is not None:
                return np.array(loaded_img)
            return np.array([])
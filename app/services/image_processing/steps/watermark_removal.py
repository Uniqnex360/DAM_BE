import cv2
import numpy as np
import torch
import logging
import traceback
from typing import Protocol, runtime_checkable, Tuple, Optional, Dict, Any
from enum import Enum, auto
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ContentType(Enum):
    UNIFORM = auto()
    TEXTURE = auto()
    TEXT = auto()
    MIXED = auto()
    PRODUCT_ON_WHITE = auto()


@dataclass
class DetectionResult:
    mask: np.ndarray
    confidence: float
    method: str


class ContentTypeClassifier:
    """Fixed classifier with better product detection."""
    
    def __init__(self):
        self.mser = cv2.MSER_create() if hasattr(cv2, 'MSER_create') else None
    
    def predict(self, image: np.ndarray) -> Tuple[ContentType, float]:
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate features
            h, w = gray.shape
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            color_uniformity = self._compute_color_uniformity(image)
            
            # CRITICAL FIX: Check product-on-white FIRST, before text detection
            # This prevents spiral watermarks from being mistaken for text
            is_product, product_conf = self._is_product_on_white_background(image, gray)
            if is_product:
                logger.info(f"Detected PRODUCT_ON_WHITE: conf={product_conf:.2f}")
                return ContentType.PRODUCT_ON_WHITE, product_conf
            
            # Only check text if NOT a product image
            text_density = self._estimate_text_density(gray)
            is_structured = self._is_structured_text_layout(gray) if SCIPY_AVAILABLE else False
            
            logger.debug(f"Features: lap_var={laplacian_var:.1f}, uniformity={color_uniformity:.2f}, text_density={text_density:.2f}")
            
            if color_uniformity > 0.92 and laplacian_var < 150:
                return ContentType.UNIFORM, 0.95
            
            # STRICT text detection: must have structured layout AND moderate texture
            elif text_density > 0.20 and is_structured and laplacian_var < 200:
                return ContentType.TEXT, 0.85
            
            elif laplacian_var > 400:
                return ContentType.TEXTURE, 0.85
            
            else:
                return ContentType.MIXED, 0.70
                
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ContentType.MIXED, 0.5
    
    def _is_product_on_white_background(self, image: np.ndarray, gray: np.ndarray) -> Tuple[bool, float]:
        """
        Detect e-commerce product photos.
        Key insight: Watermark spirals on white bg should NOT trigger text detection.
        """
        try:
            h, w = gray.shape
            
            # 1. Check for dominant white background
            white_thresh = 200
            white_mask = (gray > white_thresh)
            white_ratio = np.sum(white_mask) / white_mask.size
            
            # Must have significant white background (30-85%)
            if not (0.30 <= white_ratio <= 0.85):
                return False, 0.0
            
            # 2. Find the product region (non-white area)
            product_mask = ~white_mask
            
            if np.sum(product_mask) < 1000:  # Too small to be a product
                return False, 0.0
            
            # 3. Check if product is in center (typical e-commerce composition)
            y_indices, x_indices = np.where(product_mask)
            center_y, center_x = np.mean(y_indices), np.mean(x_indices)
            
            # Center should be roughly in image center (within 30% of center)
            center_dist = np.sqrt((center_x - w/2)**2 + (center_y - h/2)**2)
            max_dist = np.sqrt((w/2)**2 + (h/2)**2)
            center_score = 1.0 - min(center_dist / (max_dist * 0.5), 1.0)
            
            # 4. Check product has texture (not just noise)
            product_pixels = gray[product_mask]
            product_var = np.var(product_pixels)
            
            # 5. Check edges are mostly white (clean background)
            edge_size = min(h, w) // 10
            edges = np.concatenate([
                gray[:edge_size, :].flatten(),
                gray[-edge_size:, :].flatten(),
                gray[:, :edge_size].flatten(),
                gray[:, -edge_size:].flatten()
            ])
            edge_white_ratio = np.sum(edges > 200) / len(edges)
            
            # Scoring
            scores = {
                'white_bg': min(white_ratio * 1.5, 1.0),  # 0.3->0.45, 0.6->0.9
                'product_texture': min(product_var / 1000, 1.0) if product_var > 100 else 0,
                'centered': center_score,
                'clean_edges': edge_white_ratio
            }
            
            confidence = np.mean(list(scores.values()))
            
            logger.debug(f"Product check: white={white_ratio:.2f}, var={product_var:.0f}, "
                        f"center_score={center_score:.2f}, edge_white={edge_white_ratio:.2f}, "
                        f"conf={confidence:.2f}")
            
            # Threshold for product detection
            return confidence > 0.6, confidence
            
        except Exception as e:
            logger.error(f"Product detection error: {e}")
            return False, 0.0
    
    def _compute_color_uniformity(self, image: np.ndarray) -> float:
        try:
            if len(image.shape) == 3:
                std_color = np.std(image, axis=(0, 1))
                return max(0, 1.0 - (np.mean(std_color) / 128.0))
            return 0.5
        except:
            return 0.5
    
    def _estimate_text_density(self, gray: np.ndarray) -> float:
        """MSER-based, but ignore if image is mostly white (watermarks on white)."""
        if self.mser is None:
            return 0.0
        
        try:
            # If image is mostly white, MSER will find watermarks as text
            # Skip text detection on predominantly white images
            white_ratio = np.sum(gray > 200) / gray.size
            if white_ratio > 0.6:
                return 0.0  # Don't trust text detection on white backgrounds
            
            regions, _ = self.mser.detectRegions(gray)
            if not regions:
                return 0.0
            
            text_like = 0
            for region in regions:
                x, y, w, h = cv2.boundingRect(region)
                aspect = w / max(h, 1)
                area = w * h
                
                # Text characteristics
                if 0.2 < aspect < 12 and 50 < area < 30000:
                    text_like += 1
            
            return min(text_like / 25, 1.0)
        except:
            return 0.0
    
    def _is_structured_text_layout(self, gray: np.ndarray) -> bool:
        """Check for document-style text lines."""
        if not SCIPY_AVAILABLE:
            return False
        
        try:
            # Horizontal projection
            h_proj = np.sum(gray < 200, axis=1)
            peaks, _ = find_peaks(h_proj, height=gray.shape[1]*0.1, distance=20)
            return len(peaks) >= 3
        except:
            return False


class UniversalWatermarkDetector:
    """Multi-modal detector with coverage filtering."""
    
    def __init__(self):
        pass
    
    def detect(self, image: np.ndarray, content_type: ContentType) -> DetectionResult:
        try:
            h, w = image.shape[:2]
            results = []
            
            # Run detection methods
            try:
                freq_mask, freq_conf = self._frequency_detect(image)
                if freq_mask is not None:
                    results.append((freq_mask, freq_conf, "frequency"))
            except Exception as e:
                logger.debug(f"Freq detect failed: {e}")
            
            try:
                edge_mask, edge_conf = self._diagonal_detect(image)
                if edge_mask is not None:
                    results.append((edge_mask, edge_conf, "edge"))
            except Exception as e:
                logger.debug(f"Edge detect failed: {e}")
            
            try:
                opacity_mask, opacity_conf = self._opacity_detect(image, content_type)
                if opacity_mask is not None:
                    results.append((opacity_mask, opacity_conf, "opacity"))
            except Exception as e:
                logger.debug(f"Opacity detect failed: {e}")
            
            # Filter out full-image masks
            valid_results = []
            for mask, conf, name in results:
                coverage = np.sum(mask > 0) / mask.size
                if coverage > 0.75:
                    if content_type == ContentType.PRODUCT_ON_WHITE and name == "frequency":
                        # Erode the mask to remove full-image coverage
                        kernel = np.ones((21, 21), np.uint8)
                        mask = cv2.erode(mask, kernel, iterations=2)
                        new_coverage = np.sum(mask > 0) / mask.size
                        logger.info(f"Eroded frequency mask: {coverage:.1%} -> {new_coverage:.1%}")
                        if new_coverage < 0.75 and new_coverage > 0.05:
                            valid_results.append((mask, conf, name))
                    else:
                        logger.warning(f"Rejecting {name}: covers {coverage:.1%}")
                else:
                    valid_results.append((mask, conf, name))
            
            if not valid_results:
                return DetectionResult(np.zeros((h, w), dtype=np.uint8), 0.0, "none")
            
            # Combine masks
            combined = np.zeros((h, w), dtype=float)
            total_conf = sum([r[1] for r in valid_results])
            
            for mask, conf, name in valid_results:
                combined += mask.astype(float) * (conf / total_conf if total_conf > 0 else 1)
            
            # Threshold
            thresholds = {
                ContentType.UNIFORM: 0.15,
                ContentType.TEXT: 0.40,
                ContentType.TEXTURE: 0.25,
                ContentType.MIXED: 0.30,
                ContentType.PRODUCT_ON_WHITE: 0.20
            }
            
            thresh = thresholds.get(content_type, 0.25) * 255
            binary_mask = (combined > thresh).astype(np.uint8) * 255
            
            # Cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            
            final_conf = max([r[1] for r in valid_results])
            method_str = "+".join([r[2] for r in valid_results])
            
            return DetectionResult(binary_mask, final_conf, method_str)
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            h, w = image.shape[:2]
            return DetectionResult(np.zeros((h, w), dtype=np.uint8), 0.0, "error")
    
    def _frequency_detect(self, image: np.ndarray):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            h, w = gray.shape
            
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            mag_log = np.log(np.abs(fshift) + 1)
            
            # Exclude DC
            cy, cx = h // 2, w // 2
            mag_log[cy-5:cy+5, cx-5:cx+5] = 0
            
            peak_thresh = np.percentile(mag_log, 99.0)
            peaks = (mag_log > peak_thresh).astype(np.uint8) * 255
            
            if np.sum(peaks) < 50:
                return None, 0.0
            
            # Create mask from frequency peaks
            f_filtered = np.fft.fftshift(f).copy()
            f_filtered[mag_log < peak_thresh * 0.8] *= 0.3
            img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered)))
            
            diff = np.abs(gray.astype(float) - img_back)
            diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            
            mask = (diff_norm > 0.3).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            return mask, min(np.sum(peaks) / 1000, 1.0)
        except:
            return None, 0.0
    
    def _diagonal_detect(self, image: np.ndarray):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                                   minLineLength=40, maxLineGap=10)
            
            if lines is None:
                return None, 0.0
            
            mask = np.zeros_like(gray)
            count = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                
                if (30 < angle < 60) or (120 < angle < 150):
                    cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=10)
                    count += 1
            
            if count == 0:
                return None, 0.0
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.dilate(mask, kernel, iterations=1)
            return mask, min(count / 10, 1.0)
        except:
            return None, 0.0
    
    def _opacity_detect(self, image: np.ndarray, content_type: ContentType):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Local variance
            local_var = np.zeros_like(gray, dtype=float)
            window = 15
            
            for y in range(0, gray.shape[0], window):
                for x in range(0, gray.shape[1], window):
                    patch = gray[y:y+window, x:x+window]
                    if patch.size > 0:
                        local_var[y:y+window, x:x+window] = np.var(patch)
            
            local_var = (local_var - local_var.min()) / (local_var.max() - local_var.min() + 1e-8)
            
            # Different thresholds for different content
            if content_type == ContentType.UNIFORM:
                mask = ((local_var > 0.1) & (local_var < 0.6)).astype(np.uint8) * 255
            else:
                mask = ((local_var > 0.15) & (local_var < 0.7)).astype(np.uint8) * 255
            
            conf = min(np.sum(mask) / (mask.size * 255) * 8, 1.0)
            return mask, conf
        except:
            return None, 0.0


class RemovalStrategy:
    def remove(self, image: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class UniformBackgroundStrategy(RemovalStrategy):
    def remove(self, image, original, mask):
        try:
            if np.sum(mask) == 0:
                return image
            
            bg_mask = cv2.bitwise_not(mask)
            if np.sum(bg_mask) > 0:
                bg_color = np.median(original[bg_mask > 0].reshape(-1, 3), axis=0)
            else:
                bg_color = [255, 255, 255]
            
            result = image.copy()
            h, w = image.shape[:2]
            
            # Fill with noise
            noise = np.random.normal(0, 2, (h, w, 3))
            for c in range(3):
                channel = np.full((h, w), bg_color[c], dtype=np.float32) + noise[:, :, c]
                result[:, :, c] = np.where(mask > 0, 
                                          np.clip(channel, 0, 255).astype(np.uint8),
                                          result[:, :, c])
            
            # Blend
            if np.sum(mask) > 100:
                ys, xs = np.where(mask > 0)
                center = (int(np.mean(xs)), int(np.mean(ys)))
                temp = image.copy()
                temp[mask > 0] = result[mask > 0]
                result = cv2.seamlessClone(temp, image, mask, center, cv2.NORMAL_CLONE)
            
            return result
        except Exception as e:
            logger.error(f"Uniform strategy failed: {e}")
            return image
class ProductOnWhiteStrategy(RemovalStrategy):
    """Handles product photos with white backgrounds and watermark patterns."""
    
    def remove(self, image: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            if np.sum(mask) == 0:
                return image
            
            logger.info(f"ProductOnWhite: mask coverage {np.sum(mask>0)/mask.size:.2%}")
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            
            # CRITICAL FIX: If mask is small (<15%), watermarks weren't fully detected
            # Use frequency domain to catch the spiral patterns on white surfaces
            mask_coverage = np.sum(mask > 0) / mask.size
            
            if mask_coverage < 0.15:
                logger.info("Low mask coverage, adding frequency-based watermark removal")
                # Apply frequency filtering to entire image to suppress periodic watermarks
                result = self._remove_periodic_watermarks(image)
            else:
                result = image.copy()
            
            # Separate background (white) from product
            # Lower threshold for white detection (catch more background)
            bg_mask_bool = (gray > 180) & (mask > 0)  # Lowered from 200
            fg_mask_bool = (gray <= 180) & (mask > 0)
            
            # Fill background with pure white
            if np.sum(bg_mask_bool) > 0:
                result[bg_mask_bool] = [255, 255, 255]
                logger.info(f"Filled {np.sum(bg_mask_bool)} bg pixels")
            
            # Handle foreground (product) with texture-preserving inpainting
            if np.sum(fg_mask_bool) > 0:
                fg_mask = fg_mask_bool.astype(np.uint8) * 255
                # Use smaller radius for white products to avoid blurring
                fg_inpainted = cv2.inpaint(result, fg_mask, 2, cv2.INPAINT_TELEA)
                result[fg_mask_bool] = fg_inpainted[fg_mask_bool]
                logger.info(f"Inpainted {np.sum(fg_mask_bool)} fg pixels")
            
            # Handle blue banner at bottom if present
            result = self._remove_banner(result, original)
            
            return result
            
        except Exception as e:
            logger.error(f"ProductOnWhite failed: {e}")
            return image
    
    def _remove_periodic_watermarks(self, image: np.ndarray) -> np.ndarray:
        """Suppress periodic patterns (spirals) using frequency domain filtering."""
        try:
            result = image.copy().astype(np.float32)
            h, w = image.shape[:2]
            
            for c in range(3):
                channel = image[:, :, c].astype(np.float32)
                
                # FFT
                f = np.fft.fft2(channel)
                fshift = np.fft.fftshift(f)
                
                # Create notch filter to suppress high-frequency periodic patterns
                # (watermark spirals create specific frequency peaks)
                cy, cx = h // 2, w // 2
                
                # Suppress frequencies in rings (periodic patterns)
                y, x = np.ogrid[:h, :w]
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                
                # Attenuate high frequencies that are likely watermarks (not edges)
                # Keep low frequencies (overall shape) and very high (noise)
                # Target mid-high frequencies where watermarks live
                watermark_ring = (dist > 20) & (dist < 100)
                fshift[watermark_ring] *= 0.7  # Reduce by 30%
                
                # Inverse FFT
                f_ishift = np.fft.ifftshift(fshift)
                channel_back = np.abs(np.fft.ifft2(f_ishift))
                result[:, :, c] = channel_back
            
            return np.clip(result, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"Frequency removal failed: {e}")
            return image
    
    def _remove_banner(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Detect and remove colored banners (like dreamstime footer)."""
        try:
            h, w = image.shape[:2]
            # Check bottom 10% for colored banner
            bottom = original[int(h*0.9):, :]
            
            # Convert to HSV to detect non-white colors
            hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
            
            # If average saturation is high, it's a colored banner
            avg_sat = np.mean(hsv[:, :, 1])
            
            if avg_sat > 30:  # Colored banner detected
                logger.info("Removing colored banner at bottom")
                result = image.copy()
                # Fill with white or extend background
                result[int(h*0.9):, :] = [255, 255, 255]
                return result
            
            return image
        except Exception as e:
            logger.error(f"Banner removal failed: {e}")
            return image


class TexturePreservingStrategy(RemovalStrategy):
    def remove(self, image, original, mask):
        try:
            if np.sum(mask) == 0:
                return image
            return cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
        except Exception as e:
            logger.error(f"Texture strategy failed: {e}")
            return image


class TextAwareStrategy(RemovalStrategy):
    def __init__(self):
        self.ocr = None
        if EASYOCR_AVAILABLE:
            try:
                self.ocr = easyocr.Reader(['en'], gpu=False)
                logger.info("OCR initialized")
            except Exception as e:
                logger.warning(f"OCR init failed: {e}")
    
    def remove(self, image, original, mask):
        try:
            if np.sum(mask) == 0:
                return image
            
            if self.ocr is None:
                # Fallback
                return cv2.inpaint(image, mask, 2, cv2.INPAINT_TELEA)
            
            # Detect text regions
            try:
                results = self.ocr.readtext(original, paragraph=False, detail=1)
            except Exception as e:
                logger.warning(f"OCR read failed: {e}")
                return cv2.inpaint(image, mask, 2, cv2.INPAINT_TELEA)
            
            refined_mask = mask.copy()
            
            for (bbox, text, prob) in results:
                if prob < 0.4:
                    continue
                
                pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
                x, y, w, h = cv2.boundingRect(pts)
                
                # Check overlap
                if y+h > original.shape[0] or x+w > original.shape[1]:
                    continue
                    
                roi_mask = mask[y:y+h, x:x+w]
                overlap = np.sum(roi_mask > 0) / (w * h + 1e-8)
                
                if overlap > 0.2:
                    cv2.fillPoly(refined_mask, [pts], 0)
            
            if np.sum(refined_mask) > 0:
                return cv2.inpaint(image, refined_mask, 2, cv2.INPAINT_TELEA)
            
            return image
        except Exception as e:
            logger.error(f"Text strategy failed: {e}")
            return cv2.inpaint(image, mask, 2, cv2.INPAINT_TELEA)


class ConservativeStrategy(RemovalStrategy):
    def remove(self, image, original, mask):
        try:
            if np.sum(mask) == 0:
                return image
            return cv2.inpaint(image, mask, 2, cv2.INPAINT_TELEA)
        except Exception as e:
            logger.error(f"Conservative failed: {e}")
            return image


class QualityChecker:
    def check(self, original, result, mask):
        try:
            if np.sum(mask) == 0:
                return True, "no_mask"
            
            mask_ratio = np.sum(mask > 0) / mask.size
            
            # Large mask handling
            if mask_ratio > 0.80:
                return self._check_large_mask(original, result, mask)
            
            # Normal checks
            unmasked = cv2.bitwise_not(mask)
            
            if np.sum(unmasked) > 100:
                ssim = self._compute_ssim_masked(original, result, unmasked)
                if ssim < 0.85:
                    return False, f"ssim_low ({ssim:.2f})"
            
            # Texture check
            texture_score = self._check_texture_boundary(original, result, mask)
            if texture_score < 0.15:
                return False, f"texture_low ({texture_score:.2f})"
            
            return True, "passed"
        except Exception as e:
            logger.error(f"Quality check error: {e}")
            return False, "check_error"
    
    def _check_large_mask(self, original, result, mask):
        try:
            # Check not blurred
            if np.var(result) < 10:
                return False, "blurred"
            
            # Check gradient preservation
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            res_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            
            orig_grad = np.abs(cv2.Sobel(orig_gray, cv2.CV_64F, 1, 0))
            res_grad = np.abs(cv2.Sobel(res_gray, cv2.CV_64F, 1, 0))
            
            ratio = np.mean(res_grad) / (np.mean(orig_grad) + 1e-8)
            if ratio < 0.2:
                return False, f"grad_low ({ratio:.2f})"
            
            return True, "passed_large"
        except:
            return True, "large_fallback"
    
    def _compute_ssim_masked(self, img1, img2, mask):
        try:
            g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
            
            valid = mask > 0
            if np.sum(valid) < 100:
                return 1.0
            
            v1, v2 = g1[valid].astype(float), g2[valid].astype(float)
            if np.std(v1) == 0 or np.std(v2) == 0:
                return 1.0 if np.allclose(v1, v2) else 0.0
            
            return np.corrcoef(v1, v2)[0, 1]
        except:
            return 0.0
    
    def _check_texture_boundary(self, original, result, mask):
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(mask, kernel)
            boundary = dilated & ~mask
            
            if np.sum(boundary) < 10:
                return 1.0
            
            g1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            
            l1 = cv2.Laplacian(g1, cv2.CV_64F)
            l2 = cv2.Laplacian(g2, cv2.CV_64F)
            
            v1 = np.var(l1[boundary > 0])
            v2 = np.var(l2[boundary > 0])
            
            if v1 < 1:
                return 1.0
            return min(v2 / v1, 1.0)
        except:
            return 1.0


@runtime_checkable
class ProcessingStep(Protocol):
    def process(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        ...


class WatermarkRemovalStep(ProcessingStep):
    def __init__(self):
        try:
            self.classifier = ContentTypeClassifier()
            self.detector = UniversalWatermarkDetector()
            self.quality_checker = QualityChecker()
            
            self.strategies = {
                ContentType.UNIFORM: UniformBackgroundStrategy(),
                ContentType.TEXTURE: TexturePreservingStrategy(),
                ContentType.TEXT: TextAwareStrategy(),
                ContentType.MIXED: ConservativeStrategy(),
                ContentType.PRODUCT_ON_WHITE: ProductOnWhiteStrategy(),
            }
            
            logger.info("WatermarkRemovalStep initialized")
        except Exception as e:
            logger.error(f"Init failed: {e}")
            raise
    
    def process(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        try:
            if image is None or original is None:
                logger.error("None input")
                return image
            
            if image.shape != original.shape:
                logger.error(f"Shape mismatch: {image.shape} vs {original.shape}")
                return image
            
            logger.info(f"Processing: {image.shape}")
            
            # Classify
            try:
                content_type, conf = self.classifier.predict(original)
                logger.info(f"Content: {content_type.name} (conf: {conf:.2f})")
            except Exception as e:
                logger.error(f"Classify failed: {e}")
                content_type, conf = ContentType.MIXED, 0.5
            
            # Detect
            try:
                detection = self.detector.detect(original, content_type)
                logger.info(f"Detection: {detection.method}, conf: {detection.confidence:.2f}")
                
                coverage = np.sum(detection.mask > 0) / detection.mask.size
                logger.info(f"Mask coverage: {coverage:.2%}")
                
                if np.sum(detection.mask) == 0:
                    logger.info("No watermarks")
                    return image
                
                # Debug save
                try:
                    cv2.imwrite("/tmp/debug_mask.png", detection.mask)
                    overlay = image.copy()
                    overlay[detection.mask > 0] = [0, 0, 255]
                    cv2.imwrite("/tmp/debug_overlay.png", overlay)
                except Exception as e:
                    logger.debug(f"Debug save failed: {e}")
                    
            except Exception as e:
                logger.error(f"Detection failed: {e}")
                return image
            
            # Strategy
            try:
                strategy = self.strategies.get(content_type, ConservativeStrategy())
                logger.info(f"Strategy: {strategy.__class__.__name__}")
                result = strategy.remove(image, original, detection.mask)
            except Exception as e:
                logger.error(f"Strategy failed: {e}")
                result = ConservativeStrategy().remove(image, original, detection.mask)
            
            # Quality check
            try:
                passed, reason = self.quality_checker.check(original, result, detection.mask)
                logger.info(f"Quality: {reason}")
                
                if not passed:
                    logger.warning(f"Quality failed: {reason}, fallback...")
                    result = ConservativeStrategy().remove(image, original, detection.mask)
                    
                    passed2, reason2 = self.quality_checker.check(original, result, detection.mask)
                    if not passed2:
                        logger.error(f"Fallback failed: {reason2}, returning original")
                        return image
            except Exception as e:
                logger.error(f"Quality check failed: {e}")
            
            return result
            
        except Exception as e:
            logger.critical(f"Critical error: {e}\n{traceback.format_exc()}")
            return image
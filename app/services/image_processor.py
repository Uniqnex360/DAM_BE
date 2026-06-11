import cv2
import numpy as np
from PIL import Image
import io
import time
import logging
from typing import Tuple, Dict, List
from transparent_background import Remover

# from diffusers import StableDiffusionInpaintPipeline

import easyocr 
from rembg import remove, new_session
rembg_session = new_session("isnet-general-use")
# device = "cuda" if torch.cuda.is_available() else "cpu"

# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     "runwayml/stable-diffusion-inpainting",
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32
# )
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest, HDStrategy
from typing import Optional
from simple_lama_inpainting import SimpleLama

_lama = None
def get_lama():
    global _lama
    if _lama is None:
        _lama = SimpleLama()
    return _lama
# pipe = pipe.to(device)
from huggingface_hub import hf_hub_download
_iopaint = None
def get_iopaint():
    global _iopaint
    if _iopaint is None:
        _iopaint = ModelManager(
            name="sd2",          # generative fill
            device="cpu",
            # name="lama"        # fast erase/watermark
        )
    return _iopaint
logger = logging.getLogger(__name__)
TARGET_SIZE = (2000, 2000)
CONFIDENCE_THRESHOLD = 0.6
_remover = None
from ultralytics import YOLO

_wm_detector = None

def get_wm_detector():
    global _wm_detector
    if _wm_detector is None:
        try:
            # This returns the local path to the 'best.pt' file 
            # (It's already in the Docker image cache, so no download happens here)
            model_path = hf_hub_download(
                repo_id="qfisch/yolov8n-watermark-detection", 
                filename="best.pt"
            )
            # Load YOLO using the actual local path
            _wm_detector = YOLO(model_path)
            logger.info(f" Watermark detector loaded from: {model_path}")
        except Exception as e:
            logger.error(f" Failed to load watermark detector: {e}")
            _wm_detector = None
    return _wm_detector
def get_remover():
    global _remover
    if _remover is None:
        logger.info("Initializing background remover...")
        _remover = Remover(mode="fast")
        logger.info('Background remover ready!')
    return _remover

_ocr_reader = None
def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['en']) 
    return _ocr_reader
class ImageProcessor:
    def __init__(
        self,
        file_bytes: bytes,
        resize_dims: dict = None,
        operations: list = None,
        autoDetect: bool = False,
        skip_crop: bool = False,
        crop_mode: Optional[str] = None,         
        target_aspect_ratio: Optional[str] = None,
        target_dimensions: dict = None
    ):
        nparr = np.frombuffer(file_bytes, np.uint8)
        self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if self.img is None:
            raise ValueError("Could not decode image bytes")
        self.original_h, self.original_w = self.img.shape[:2]
        self.resize_dims = resize_dims
        self.operations = operations or []
        self.original_img = self.img.copy()
        self.auto_detect = autoDetect
        self.skip_crop = skip_crop
        self.crop_mode = crop_mode
        self.target_aspect_ratio = target_aspect_ratio
        self.last_ai_alpha = None 
        
        if target_dimensions:
            self.target_w = target_dimensions.get("width", self.original_w)
            self.target_h = target_dimensions.get("height", self.original_h)
        else:
            self.target_w = self.original_w
            self.target_h = self.original_h
        
        logger.info(
            f"ImageProcessor: input={self.original_w}x{self.original_h}, "
            f"target={self.target_w}x{self.target_h}, "
            f"skip_crop={self.skip_crop}, operations={self.operations}"
        )

    def crop_to_aspect_ratio(self, ratio_str: str):
        h, w = self.img.shape[:2]
        ratio_w, ratio_h = map(int, ratio_str.split(':'))
        target_ratio = ratio_w / ratio_h
        current_ratio = w / h

        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            offset = (w - new_w) // 2
            self.img = self.img[:, offset:offset + new_w]
        else:
            new_h = int(w / target_ratio)
            offset = (h - new_h) // 2
            self.img = self.img[offset:offset + new_h, :]
    def remove_text(self):
        
        try:
            logger.info("Starting clean-room text removal...")
            reader = get_ocr_reader()
            
            
            results = reader.readtext(self.img, text_threshold=0.3, link_threshold=0.2, low_text=0.2)
            
            if not results:
                logger.info("No text detected.")
                return

            h, w = self.img.shape[:2]
            
            for (bbox, text, prob) in results:
                
                points = np.array(bbox).astype(np.int32)
                x, y, bw, bh = cv2.boundingRect(points)
                
                
                x1, y1 = max(0, x-2), max(0, y-2)
                x2, y2 = min(w, x+bw+2), min(h, y+bh+2)
                
                
                
                roi = self.img[y1:y2, x1:x2]
                
                
                border_mask = np.ones(roi.shape[:2], dtype=np.uint8) * 255
                border_mask[2:-2, 2:-2] = 0 
                
                
                bg_color = np.median(roi[border_mask > 0], axis=0)

                
                
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                avg_bg_gray = np.mean(cv2.cvtColor(np.uint8([[bg_color]]), cv2.COLOR_BGR2GRAY))
                
                if avg_bg_gray > 127:
                    
                    _, text_mask = cv2.threshold(gray_roi, avg_bg_gray - 30, 255, cv2.THRESH_BINARY_INV)
                else:
                    
                    _, text_mask = cv2.threshold(gray_roi, avg_bg_gray + 30, 255, cv2.THRESH_BINARY)

                
                
                text_mask = cv2.dilate(text_mask, np.ones((3,3), np.uint8), iterations=1)
                
                
                roi[text_mask > 0] = bg_color
                
                
                roi_refined = cv2.GaussianBlur(roi, (3,3), 0)
                mask_feather = cv2.GaussianBlur(text_mask, (5,5), 0).astype(np.float32) / 255.0
                
                for c in range(3):
                    self.img[y1:y2, x1:x2, c] = (
                        roi_refined[:,:,c] * mask_feather + 
                        self.img[y1:y2, x1:x2, c] * (1 - mask_feather)
                    ).astype(np.uint8)

            logger.info(f"Surgically filled {len(results)} text areas.")
        except Exception as e:
            logger.error(f"Text removal failed: {e}")

    def _detect_corner_logos(self, mask):
        
        h, w = self.img.shape[:2]
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        
        corner_size = int(min(h, w) * 0.25)  
        
        corners = [
            gray[0:corner_size, 0:corner_size],                    
            gray[0:corner_size, w-corner_size:w],                   
            gray[h-corner_size:h, 0:corner_size],                    
            gray[h-corner_size:h, w-corner_size:w],                  
        ]
        
        corner_positions = [
            (0, 0, corner_size, corner_size),
            (0, w-corner_size, corner_size, w),
            (h-corner_size, 0, h, corner_size),
            (h-corner_size, w-corner_size, h, w),
        ]
        
        for corner_img, (y1, x1, y2, x2) in zip(corners, corner_positions):
            
            _, thresh = cv2.threshold(corner_img, 200, 255, cv2.THRESH_BINARY_INV)
            
            
            colored_ratio = np.sum(thresh > 0) / thresh.size
            if colored_ratio > 0.01 and colored_ratio < 0.3:
                
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for c in contours:
                    area = cv2.contourArea(c)
                    if area > (corner_img.shape[0] * corner_img.shape[1] * 0.005):
                        
                        offset_contour = c.copy()
                        offset_contour[:, :, 0] += x1
                        offset_contour[:, :, 1] += y1
                        
                        
                        x, y, cw, ch = cv2.boundingRect(offset_contour)
                        pad = 10
                        cv2.rectangle(mask, 
                            (max(0, x - pad), max(0, y - pad)), 
                            (min(mask.shape[1], x + cw + pad), min(mask.shape[0], y + ch + pad)), 
                            255, -1)
        
        logger.info("Corner logo detection complete")
    # def remove_watermark(self):
    #     try:
    #         logger.info("Starting Stable Diffusion watermark removal...")

    #         img = self.img.copy()
    #         h, w = img.shape[:2]

    #         # --- Detect watermark via brightness ---
    #         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #         lower_white = np.array([0, 0, 170])
    #         upper_white = np.array([180, 60, 255])

    #         mask = cv2.inRange(hsv, lower_white, upper_white)
    #         mask = cv2.GaussianBlur(mask, (7,7), 0)

    #         if np.sum(mask) == 0:
    #             logger.info("No watermark detected.")
    #             return

    #         # Convert to PIL
    #         image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #         mask_pil = Image.fromarray(mask).convert("L")

    #         prompt = "remove watermark, preserve fabric texture, preserve embroidery details"

    #         result = pipe(
    #             prompt=prompt,
    #             image=image_pil,
    #             mask_image=mask_pil,
    #             guidance_scale=7.5,
    #             num_inference_steps=30
    #         ).images[0]

    #         self.img = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    #         logger.info("Stable Diffusion watermark removal complete ✅")

    #     except Exception as e:
    #         logger.error(f"Watermark removal failed: {e}")
    def image_refill(self):
        
        try:
            logger.info("Auto-analyzing geometry for IOPaint refill...")
            h, w = self.img.shape[:2]

            # ── 1. Detect where the product touches the border ────────────────
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            # Find high-contrast edges (the product's 'stubs')
            edges = cv2.Canny(gray, 50, 150)
            
            # ── 2. Create Expanded Canvas ─────────────────────────────────────
            # Use BORDER_REPLICATE instead of a solid color.
            # This prevents the 'white smudge' by extending the actual background texture.
            pad = int(max(h, w) * 0.15)
            img_padded = cv2.copyMakeBorder(self.img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            mask = np.zeros(img_padded.shape[:2], dtype=np.uint8)

            # ── 3. Surgical Masking (Target only the cut-off areas) ────────────
            has_stubs = False
            
            # Check Top
            if np.any(edges[0, :]):
                idx = np.where(edges[0, :] > 0)[0]
                cv2.rectangle(mask, (idx[0]+pad-10, 0), (idx[-1]+pad+10, pad+20), 255, -1)
                has_stubs = True
            
            # Check Bottom
            if np.any(edges[-1, :]):
                idx = np.where(edges[-1, :] > 0)[0]
                cv2.rectangle(mask, (idx[0]+pad-10, h+pad-20), (idx[-1]+pad+10, h+2*pad), 255, -1)
                has_stubs = True

            # Check Left
            if np.any(edges[:, 0]):
                idx = np.where(edges[:, 0] > 0)[0]
                cv2.rectangle(mask, (0, idx[0]+pad-10), (pad+20, idx[-1]+pad+10), 255, -1)
                has_stubs = True

            # Check Right
            if np.any(edges[:, -1]):
                idx = np.where(edges[:, -1] > 0)[0]
                cv2.rectangle(mask, (w+pad-20, idx[0]+pad-10), (w+2*pad, idx[-1]+pad+10), 255, -1)
                has_stubs = True

            if not has_stubs:
                logger.info("No cut-off edges detected by AI analysis.")
                return False

            # ── 4. Call IOPaint ───────────────────────────────────────────────
            model = get_iopaint()
            img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
            
            # We use the method provided in your snippet
            # Note: If using LaMa, prompt is ignored, but helpful if you switch to SD
            result_rgb = model.inpaint(
                image=img_rgb,
                mask=mask,
                config=InpaintRequest(
                    hd_strategy=HDStrategy.ORIGINAL,
                    hd_strategy_crop_margin=128,
                    prompt="seamless product surface extension, high resolution metal texture",
                )
            )
            
            self.img = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            logger.info("IOPaint shape reconstruction complete ✅")
            return True

        except Exception as e:
            logger.error(f"IOPaint Refill failed: {e}")
            return False

    def remove_watermark(self):
        try:
            logger.info("Starting watermark removal...")
            h, w = self.img.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)

            # ── YOLOv8 watermark detection ────────────────────────────────────────
            detector = get_wm_detector()
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            results = detector(img_rgb, conf=0.25, verbose=False)

            detected = False
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    pad = 10
                    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
                    combined_mask[y1:y2, x1:x2] = 255
                    detected = True
                    logger.info(f"Watermark detected: box=({x1},{y1},{x2},{y2}) conf={box.conf[0]:.2f}")

            if not detected:
                logger.info("No watermark detected by YOLO, skipping.")
                return

            # ── Dilate mask slightly to catch edges ───────────────────────────────
            kernel = np.ones((15, 15), np.uint8)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

            # ── LaMa inpainting ───────────────────────────────────────────────────
            lama = get_lama()
            pil_img = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            pil_mask = Image.fromarray(combined_mask)
            result = lama(pil_img, pil_mask)
            self.img = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            logger.info("Watermark removal complete ✅")

        except Exception as e:
            logger.error(f"Watermark removal failed: {e}")
    def resize_ecom(self):
        h, w = self.img.shape[:2]
        if not self.resize_dims:
            return
        if isinstance(self.resize_dims, list):
            results = []
            for config in self.resize_dims:
                result = self._apply_single_resize(self.original_img, config)
                results.append({
                    "id": config.get("id"),
                    "width": config.get("width"),
                    "height": config.get("height"),
                    "image_bytes": result
                })
            return results
        else:
            return self._apply_single_resize(self.img, self.resize_dims)

    def _apply_single_resize(self, img, resize_config):
        target_w = resize_config.get("width")
        target_h = resize_config.get("height")
        if not target_w or not target_h:
            return None
        h, w = img.shape[:2]  
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  
        pil = pil.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
        canvas.paste(pil, offset)
        return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

    def _foreground_mask(self, img) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    def analyze(self) -> Dict[str, float]:
        h, w = self.img.shape[:2]
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        fg = self._foreground_mask(self.img)
        fg_ratio = np.sum(fg > 0) / (h * w)
        conf = {"bg_clean": 0.0, "shadow": 0.0, "crop": 0.0, "watermark": 0.0, "resize": 0.0}
        
        if self.resize_dims:
            conf["resize"] = 1.0
        if fg_ratio < 0.35:
            conf["crop"] = min(1.0, (0.5 - fg_ratio) * 3)
        
        corner_std = np.mean([np.std(gray[:80, :80]), np.std(gray[:80, -80:])])
        conf["bg_clean"] = np.clip((corner_std - 10) / 20, 0, 1)
        
        
        v = hsv[:, :, 2]
        mean_v = np.mean(v) if np.mean(v) > 0 else 1
        shadow_mask = (v < 0.35 * mean_v) & (fg > 0)
        shadow_ratio = np.sum(shadow_mask) / (h * w)
        conf["shadow"] = np.clip(shadow_ratio * 40, 0, 1)
        
        return conf

    def clean_background(self):
        try:
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            remover = get_remover()
            out = remover.process(pil_img)
            bg = Image.new("RGB", out.size, (255, 255, 255))
            if out.mode == 'RGBA':
                bg.paste(out, mask=out.split()[3])
            else:
                bg.paste(out)
            self.img = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"BG removal failed: {e}")

    

    def remove_shadow(self):
        try:
            logger.info("Starting ISNet shadow removal...")

            max_dim = 2000
            h, w = self.original_img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                self.original_img = cv2.resize(
                    self.original_img,
                    (int(w * scale), int(h * scale))
                )

            success, buffer = cv2.imencode(".png", self.original_img)
            if not success:
                raise ValueError("Failed to encode image")

            input_bytes = buffer.tobytes()

            output_bytes = remove(
                input_bytes,
                session=rembg_session
            )

            pil_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

            # Proper color-safe blending
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[3])

            self.img = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)

            logger.info("ISNet shadow removal complete ✅")

        except Exception as e:
            logger.error(f"ISNet shadow removal failed: {e}")
            self._traditional_shadow_removal()

    def _cleanup_internal_holes(self):
        hsv_orig = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2HSV)
        corners = [hsv_orig[5:15, 5:15], hsv_orig[5:15, -15:-5]]
        avg_h = np.median([np.median(c[:,:,0]) for c in corners])
        avg_s = np.median([np.median(c[:,:,1]) for c in corners])
        avg_v = np.median([np.median(c[:,:,2]) for c in corners])

        lower_bg = np.array([max(0, avg_h - 12), max(20, avg_s - 70), max(20, avg_v - 100)])
        upper_bg = np.array([min(179, avg_h + 12), 255, 255])
        color_mask = cv2.inRange(hsv_orig, lower_bg, upper_bg)

        if self.last_ai_alpha is not None:
            uncertain_area = np.zeros_like(self.last_ai_alpha, dtype=np.uint8)
            uncertain_area[(self.last_ai_alpha < 100) & (self.last_ai_alpha > 20)] = 255
            final_mask = cv2.bitwise_and(color_mask, uncertain_area)
        else:
            final_mask = color_mask
        edges = cv2.Canny(
        cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY),
        30, 100
        )
        edge_protection = cv2.dilate(edges, np.ones((7,7), np.uint8), iterations=2)
        final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(edge_protection))
        self.img[final_mask > 0] = [255, 255, 255]

    def _traditional_shadow_removal(self):
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        self.img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _check_alpha_quality(self, alpha: np.ndarray) -> float:
        edges = cv2.Canny(alpha, 50, 150)
        return 0.9 if (np.sum(edges > 0) / alpha.size) < 0.15 else 0.3

    def retouch_image(self, mode: str = "product"):
        try:
            if mode == "auto": mode = self._detect_image_type()
            return self._retouch_product() if mode == "product" else self._retouch_portrait()
        except Exception as e:
            logger.error(f"Retouch failed: {e}")
            return False

    def _detect_image_type(self) -> str:
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 30, 60])
        upper_skin = np.array([20, 150, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        return "product" if (np.sum(skin_mask > 0) / skin_mask.size) < 0.05 else "portrait"

    def _retouch_product(self):
        logger.info("Applying PRO-WEBSITE clarity retouch")
        
        self.img = cv2.edgePreservingFilter(self.img, flags=1, sigma_s=30, sigma_r=0.4)
        
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        l_float = l.astype(np.float32)
        blur = cv2.GaussianBlur(l_float, (0, 0), 3.0)
        high_pass = l_float - blur
        l = np.clip(l_float + (high_pass * 1.4), 0, 255).astype(np.uint8)
        
        gauss_fine = cv2.GaussianBlur(l, (0, 0), 0.8)
        l = cv2.addWeighted(l, 1.5, gauss_fine, -0.5, 0)
        self.img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]]) * 0.03
        self.img = cv2.addWeighted(self.img, 1.0, cv2.filter2D(self.img, -1, kernel), 1.0, 0)
        
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.10 
        self.img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        self.img = cv2.convertScaleAbs(self.img, alpha=1.04, beta=2)
        return True

    def _retouch_portrait(self):
        self.img = cv2.fastNlMeansDenoisingColored(self.img, None, 7, 7, 7, 21)
        return True

    def smart_crop(self):
        if self.skip_crop: return
        fg = self._foreground_mask(self.img)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return
        boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > (self.original_h * self.original_w * 0.01)]
        if not boxes: return
        x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x2, y2 = max(b[0]+b[2] for b in boxes), max(b[1]+b[3] for b in boxes)
        pad = int(0.08 * max(x2-x1, y2-y1))
        cropped = self.img[max(0, y1-pad):min(self.original_h, y2+pad), max(0, x1-pad):min(self.original_w, x2+pad)]
        self.img = self._upscale_to_size(cropped, self.original_w, self.original_h)

    def _upscale_to_size(self, img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((new_w, new_h), Image.LANCZOS)
        upscaled = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        canvas[(target_h-new_h)//2:(target_h-new_h)//2+new_h, (target_w-new_w)//2:(target_w-new_w)//2+new_w] = upscaled
        return canvas

    def process(self) -> Dict:  
        start_time = time.time()
        self.resize_results = None
        steps_applied = []
        confidence = self.analyze()
        if self.crop_mode == "preset" and self.target_aspect_ratio:
            self.crop_to_aspect_ratio(self.target_aspect_ratio)
        if self.auto_detect:
            if confidence["bg_clean"] > CONFIDENCE_THRESHOLD:
                self.clean_background()
                steps_applied.append("bg_removal")
        else:
            if "text-remove" in self.operations:
                self.remove_text()
                steps_applied.append("text_removal")
            if "image-refill" in self.operations:
                self.image_refill()
                steps_applied.append("geometry_reconstruction")
            if "watermark-remove" in self.operations:
                self.remove_watermark()
            if "retouch" in self.operations:
                self.retouch_image()
                steps_applied.append("retouch")
            if any(op in self.operations for op in ["shadow-remove", "shadow_fix"]):
                self.remove_shadow()
                steps_applied.append("shadow_fix")
            if "bg-remove" in self.operations:
                self.clean_background()
                steps_applied.append("bg_removal")
            if self.resize_dims:
                result = self.resize_ecom()
                if isinstance(result, list):
                    self.resize_results = result
                    steps_applied.append("resize_multiple")
                elif result is not None:
                    self.img = result
                    steps_applied.append("resize")

        
        if self.resize_results is None and "resize" not in steps_applied:
            if self.crop_mode != "preset":
                cur_h, cur_w = self.img.shape[:2]
                if (cur_w, cur_h) != (self.target_w, self.target_h):
                    self.img = self._upscale_to_size(self.img, self.target_w, self.target_h)

        success, encoded = cv2.imencode(".jpg", self.img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return {
            "image_bytes": encoded.tobytes(),
            "confidence": confidence,
            "steps_applied": steps_applied,
            "duration_ms": int((time.time() - start_time) * 1000),
            "resize_results": self.resize_results
        }
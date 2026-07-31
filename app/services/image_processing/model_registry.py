import logging
import threading
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO
from transparent_background import Remover
from simple_lama_inpainting import SimpleLama
from huggingface_hub import hf_hub_download
from rembg import remove, new_session

from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest, HDStrategy

logger = logging.getLogger(__name__)

_locks = {
    "lama": threading.Lock(),
    "iopaint": threading.Lock(),
    "remover": threading.Lock(),
    "ocr": threading.Lock(),
    "wm_detector": threading.Lock(),
    "rembg_session": threading.Lock(),
}

_lama: Optional[SimpleLama] = None
_iopaint: Optional[ModelManager] = None
_remover: Optional[Remover] = None
_ocr_reader: Optional[easyocr.Reader] = None
_wm_detector: Optional[YOLO] = None
_rembg_session = None


def get_lama() -> SimpleLama:
    global _lama
    if _lama is None:
        with _locks["lama"]:
            if _lama is None:
                _lama = SimpleLama()
    return _lama


def get_iopaint() -> ModelManager:
    global _iopaint
    if _iopaint is None:
        with _locks["iopaint"]:
            if _iopaint is None:
                _iopaint = ModelManager(name="sd2", device="cpu")
    return _iopaint

from transformers import AutoProcessor, AutoModelForCausalLM
import torch

_florence_model = None
_florence_processor = None
_locks["florence"] = threading.Lock()
def get_florence():
    global _florence_model, _florence_processor
    if _florence_model is None:
        with _locks["florence"]:
            if _florence_model is None:
                try:
                    logger.info("Loading Florence-2...")
                    _florence_processor = AutoProcessor.from_pretrained(
                        "microsoft/Florence-2-large",
                        trust_remote_code=True,
                    )
                    _florence_model = AutoModelForCausalLM.from_pretrained(
                        "microsoft/Florence-2-large",
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                    ).eval()
                    logger.info("Florence-2 ready.")
                except Exception as e:
                    logger.error(f"Failed to load Florence-2: {e}")
                    return None
    return _florence_model, _florence_processor
def get_remover() -> Remover:
    global _remover
    if _remover is None:
        with _locks["remover"]:
            if _remover is None:
                logger.info("Initializing background remover...")
                _remover = Remover(mode="fast")
                logger.info("Background remover ready!")
    return _remover


def get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        with _locks["ocr"]:
            if _ocr_reader is None:
                _ocr_reader = easyocr.Reader(["en"])
    return _ocr_reader

import segmentation_models_pytorch as smp
import torch
import threading

_segmenters_lock = threading.Lock()
_segmenters = {}

def get_segmenter(specialty: str = "logo"):
    """Load a UNet++ segmenter from christophernavas/watermark-remover.
    specialty options: centered_text, line_pattern, logo, overlay_text,
                      repeated_text, tiny_corner, universal"""
    if specialty not in _segmenters:
        with _segmenters_lock:
            if specialty not in _segmenters:
                filename = f"segmenter_{specialty}.pth"
                try:
                    model_path = hf_hub_download(
                        repo_id="christophernavas/watermark-remover",
                        filename=filename,
                    )
                    model = smp.UnetPlusPlus(
                        encoder_name="efficientnet-b4",
                        encoder_weights=None,
                        in_channels=3,
                        classes=1,
                    )
                    state_dict = torch.load(model_path, map_location="cpu")
                    model.load_state_dict(state_dict)
                    model.eval()
                    _segmenters[specialty] = model
                    logger.info(f"Loaded watermark segmenter: {specialty}")
                except Exception as e:
                    logger.error(f"Failed to load segmenter {specialty}: {e}")
                    return None
    return _segmenters[specialty]

def get_all_segmenters():
    """Load all 6+1 segmenters for parallel inference."""
    for specialty in ["logo", "centered_text", "overlay_text",
                      "repeated_text", "tiny_corner", "line_pattern",
                      "universal"]:
        get_segmenter(specialty)
    return _segmenters

# def get_wm_detector() -> Optional[YOLO]:
#     global _wm_detector
#     if _wm_detector is None:
#         with _locks["wm_detector"]:
#             if _wm_detector is None:
#                 try:
#                     model_path = hf_hub_download(
#                         repo_id="qfisch/yolov8n-watermark-detection",
#                         filename="best.pt",
#                     )
#                     _wm_detector = YOLO(model_path)
#                     logger.info(
#                         f"Watermark detector loaded from: {model_path}")
#                 except Exception as e:
#                     logger.error(f"Failed to load watermark detector: {e}")
#                     _wm_detector = None
#     return _wm_detector

def get_wm_detector():
    global _wm_detector
    if _wm_detector is None:
        with _locks["wm_detector"]:
            if _wm_detector is None:
                try:
                    model_path = hf_hub_download(
                        repo_id="qfisch/yolov8n-watermark-detection",
                        filename="best.pt",
                    )
                    _wm_detector = YOLO(model_path)
                    logger.info(f"Watermark detector loaded from: {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load watermark detector: {e}")
                    _wm_detector = None
    return _wm_detector
def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        with _locks["rembg_session"]:
            if _rembg_session is None:
                _rembg_session = new_session("isnet-general-use")
    return _rembg_session

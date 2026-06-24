from app.services.image_processing.model_registry import get_rembg_session
from pillow_heif import register_heif_opener
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import base64
import io
import os
import logging
import pillow_heif
from app.services.image_processing.mask_generator import MaskGeneratorService
from app.services.image_processing.wall_service import WallRecoloringService
from pathlib import Path

from PIL import Image as PILImage
from rembg import remove
logger=logging.getLogger(__name__)
from app.services.image_processing.steps.room_visualizer import RoomVisualizerStep, ROOM_REGISTRY
register_heif_opener()
router = APIRouter()


@router.get("/rooms")
async def get_available_rooms():
    return [
        {
            "id": rid,
            "label": info.get("label", rid.replace("_", " ").title()),
            "emoji": info.get("emoji", "🏠"),
            "floor_y": info.get("floor_y", 80),
            "scale_hint": info.get("scale_hint", 0.35)
        }
        for rid, info in ROOM_REGISTRY.items()
    ]


@router.post("/remove-bg")
async def remove_product_bg(product_image: UploadFile = File(...)):
    try:
        contents = await product_image.read()

        try:
            if pillow_heif.is_supported(contents):
                logger.info("AVIF/HEIC detected, using direct decoder...")
                heif_file = pillow_heif.read_heif(contents)
                pil_input = PILImage.frombytes(
                    heif_file.mode,
                    heif_file.size,
                    heif_file.data,
                    "raw",
                ).convert("RGB")
            else:
                pil_input = PILImage.open(io.BytesIO(contents)).convert("RGB")
        except Exception as decode_error:
            logger.error(f"Manual decode failed: {decode_error}")
            pil_input = PILImage.open(io.BytesIO(contents)).convert("RGB")

        session = get_rembg_session()
        cutout = remove(pil_input, session=session).convert("RGBA")

        buf = io.BytesIO()
        cutout.save(buf, format="PNG")
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return JSONResponse({
            "status": "success",
            "cutout": f"data:image/png;base64,{img_base64}"
        })

    except Exception as e:
        logger.exception("Background removal crashed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/room-image/{room_id}")
async def get_room_image(room_id: str):
    room_info = ROOM_REGISTRY.get(room_id)
    if not room_info:
        raise HTTPException(status_code=404, detail="Room not found")

    base_path = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    path = os.path.join(base_path, "static", "rooms", room_info["file"])

    if not os.path.exists(path):
        raise HTTPException(
            status_code=404, detail=f"Image file missing: {path}")

    return FileResponse(path)


@router.post("/composite")
async def visualize_product_full(
    product_image: UploadFile = File(...),
    room_id: str = Form("living_room"),
    scale: float = Form(0.38),
    x_percent: float = Form(50.0),
    y_percent: float = Form(78.0)
):
    contents = await product_image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    visualizer = RoomVisualizerStep(
        room_id=room_id,
        scale=scale,
        x_percent=x_percent,
        y_percent=y_percent
    )
    result_img = visualizer.process(img)

    _, buffer = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse({
        "image": f"data:image/jpeg;base64,{img_base64}",
        "room_id": room_id,
        "scale": scale,
        "x_percent": x_percent,
        "y_percent": y_percent
    })


def get_static_rooms_path() -> Path:
    
    package_root = Path(__file__).resolve().parent.parent.parent.parent

    path = package_root / "static" / "rooms"

    logger.info(f"DIRECTORY CHECK: Searching for rooms in {path.absolute()}")

    return path


def load_image_for_opencv(path: Path) -> np.ndarray:
    """
    Reads any image format (JPG, PNG, AVIF, HEIC) and returns a BGR NumPy array
    that OpenCV can work with.
    """
    if not path.exists():
        raise FileNotFoundError(f"No file at {path}")

    # Use Pillow to open (handles AVIF/HEIC/WebP thanks to our previous setup)
    with PILImage.open(path) as img:
        img = img.convert("RGB")
        # Convert RGB (Pillow) to BGR (OpenCV)
        numpy_img = np.array(img)
        return cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)


def safe_read_image(path: Path) -> np.ndarray:
    """
    Tries OpenCV first, then Pillow. This fixes 'UnidentifiedImageError' 
    for slightly corrupted or modern JPG headers.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # 1. Try OpenCV (Most robust for standard JPG/PNG)
    # This bypasses Pillow's strict header identification
    img = cv2.imread(str(path))
    if img is not None:
        return img

    # 2. Fallback to Pillow only if OpenCV fails
    try:
        from PIL import Image as PILImage
        import io
        with open(path, "rb") as f:
            content = f.read()

        # Enable AVIF/HEIC support
        if pillow_heif.is_supported(content):
            heif_file = pillow_heif.read_heif(content)
            pil_img = PILImage.frombytes(
                heif_file.mode, heif_file.size, heif_file.data, "raw"
            )
        else:
            pil_img = PILImage.open(io.BytesIO(content))

        pil_img = pil_img.convert("RGB")
        # Convert PIL (RGB) to OpenCV (BGR)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"FATAL: All decoders failed for {path.name}: {e}")
        raise ValueError(
            f"The file {path.name} is corrupted or not a valid image.")
@router.post("/recolor-room")
async def recolor_room(room_id: str = Form(...), hex_color: str = Form(...)):
    try:
        room_info = ROOM_REGISTRY.get(room_id)
        rooms_dir = get_static_rooms_path()
        room_path = rooms_dir / room_info["file"]
        mask_path = rooms_dir / (room_info["file"].split('.')[0] + "_mask.png")

        # 1. Load Room
        room_img = safe_read_image(room_path)

        # 2. Get/Generate High-Precision Mask
        if mask_path.exists():
            wall_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            logger.info("Generating Multi-Model Perfect Mask...")

            # Pass 1: Find Walls (AI)
            wall_only_mask = MaskGeneratorService.generate_wall_mask(room_img)

            # Pass 2: Find Objects to Protect (Plants/Furniture)
            # rembg is much better at fine details like leaves than SegFormer
            session = get_rembg_session()
            img_rgb = cv2.cvtColor(room_img, cv2.COLOR_BGR2RGB)
            object_mask = np.array(
                remove(img_rgb, session=session, only_mask=True))

            # Pass 3: Subtract Objects from Walls
            # (If it's an object, it cannot be a wall)
            wall_mask = cv2.bitwise_and(
                wall_only_mask, cv2.bitwise_not(object_mask))

            # Pass 4: Final smoothing
            wall_mask = cv2.GaussianBlur(wall_mask, (3, 3), 0)

            cv2.imwrite(str(mask_path), wall_mask)

        # 3. Apply Paint
        result_img = WallRecoloringService.apply_color(
            room_img, wall_mask, hex_color)

        # 4. Return Base64
        _, buffer = cv2.imencode('.jpg', result_img)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse({"status": "success", "image": f"data:image/jpeg;base64,{img_b64}"})

    except Exception as e:
        logger.exception("Recoloring failed")
        raise HTTPException(status_code=500, detail=str(e))

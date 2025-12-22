from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.api import deps
from app.db.session import get_db
from app.models.auth import User
from fastapi.concurrency import run_in_threadpool
from app.models.assets import Upload, Image
from app.services.media import upload_image_to_cloudinary
from app.schemas.asset import ImageResponse
import uuid
from sqlalchemy import select 
import requests
from app.services.image_processor import ImageProcessor
import os

router = APIRouter()

@router.post("/upload", response_model=ImageResponse)
async def upload_asset(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    upload_record = Upload(
        user_id=current_user.id,
        status="processing"
    )
    db.add(upload_record)
    await db.commit()
    await db.refresh(upload_record)

    try:
        # FIX: Keep filename extension so we can identify it locally
        unique_filename = f"{current_user.id}/{uuid.uuid4()}_{file.filename}"
        
        file_content = await file.read()
        
        result = upload_image_to_cloudinary(file_content, unique_filename)
        
        new_image = Image(
            upload_id=upload_record.id,
            user_id=current_user.id,
            url=result.get("secure_url"),
            thumbnail_url=result.get("secure_url"), 
            width=result.get("width"),
            height=result.get("height"),
            processing_status="ready",
            file_type=file.content_type,
            name=file.filename
        )
        
        db.add(new_image)
        await db.commit()
        await db.refresh(new_image)
        
        return new_image

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/{image_id}/process")
async def process_image_asset(
    image_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    # 1. Fetch Record
    result = await db.execute(select(Image).where(Image.id == image_id))
    img_record = result.scalars().first()
    
    if not img_record:
        raise HTTPException(status_code=404, detail="Image not found")
        
    # Update status immediately
    img_record.processing_status = "processing"
    await db.commit()

    try:
        print(f"DEBUG: Processing image {image_id}")
        print(f"DEBUG: URL is {img_record.url}")
        
        image_content = None # Variable to hold bytes

        # 2. Check source: Local vs Remote
        if "localhost" in img_record.url and "static/uploads" in img_record.url:
            print("DEBUG: Detected Local Storage. Reading from disk...") 
            # Parse filename from URL
            filename = img_record.url.split("/")[-1]
            local_path = f"static/uploads/{filename}"
            print(f"DEBUG: Path is {local_path}")
            
            if os.path.exists(local_path):
                with open(local_path, "rb") as f:
                    image_content = f.read()
            else:
                raise Exception(f"Local file not found: {local_path}")
        else:
            print("DEBUG: Downloading from Remote URL...")
            # Run download in threadpool to prevent blocking main loop
            def download():
                r = requests.get(img_record.url)
                if r.status_code != 200:
                    raise Exception("Failed to download source image")
                return r.content
            
            image_content = await run_in_threadpool(download)

        # 3. PROCESS
        print("DEBUG: Image content loaded. Starting Processor...")
        processor = ImageProcessor(image_content)
        
        print("DEBUG: Running processor.process()...")
        processed_bytes, conf, steps, duration = await run_in_threadpool(processor.process)
        print(f"DEBUG: Finished! Duration: {duration}ms")
        
        # 4. Upload Result (IO Bound)
        filename = f"processed/{img_record.user_id}/{image_id}.jpg"
        upload_res = upload_image_to_cloudinary(processed_bytes, filename)
        
        # 5. Save Telemetry
        img_record.processed_url = upload_res.get("secure_url")
        img_record.confidence_scores = conf
        img_record.applied_steps = steps
        img_record.processing_time_ms = duration
        img_record.processing_status = "completed"
        
        await db.commit()
        await db.refresh(img_record)
        
        return {
            "status": "completed",
            "telemetry": {
                "confidence": conf,
                "steps": steps,
                "time_ms": duration
            }
        }

    except Exception as e:
        img_record.processing_status = "failed"
        await db.commit()
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
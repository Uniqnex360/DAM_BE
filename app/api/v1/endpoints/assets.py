from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Body
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import uuid
import requests
import os
from app.api import deps
from app.db.session import get_db
from app.models.auth import User
from app.models.assets import Upload, Image
from app.services.media import upload_image_to_cloudinary
from app.services.image_processor import ImageProcessor
from app.services.quality_analyzer import analyze_image_quality
from app.schemas.asset import ImageResponse
from app.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from app.core.config import settings
router = APIRouter()
@router.post("/analyze")
async def analyze_endpoint(request: AnalyzeRequest):
    try:
        result = await run_in_threadpool(analyze_image_quality, request)
        return {"analysis": result}
    except Exception as e:
        print(f"Analysis Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
@router.post("/upload", response_model=ImageResponse)
async def upload_asset(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    upload_record = Upload(user_id=current_user.id, status="uploaded")
    db.add(upload_record)
    await db.commit()
    await db.refresh(upload_record)
    try:
        unique_filename = f"{current_user.id}/{uuid.uuid4()}_{file.filename}"
        file_content = await file.read()
        result = upload_image_to_cloudinary(file_content, unique_filename)
        new_image = Image(
            upload_id=upload_record.id,
            user_id=current_user.id,
            url=result.get("secure_url"),
            thumbnail_url=result.get("secure_url"),
            width=result.get("width", 0),
            height=result.get("height", 0),
            processing_status="pending",
            name=file.filename,
            file_type=file.content_type
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
    operations: list = Body(default=[], embed=True),
    options: dict = Body(default={}, embed=True), 
    autoDetect: bool = Body(default=False, embed=True),   
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    print(f"DEBUG: operations={operations}")
    print(f"DEBUG: options={options}")
    print(f"DEBUG: options type={type(options)}")
    result = await db.execute(select(Image).where(Image.id == image_id))
    img_record = result.scalars().first()
    if not img_record: raise HTTPException(status_code=404, detail="Image not found")
    img_record.processing_status = "processing"
    await db.commit()
    try:
        image_content = None
        if "localhost" in img_record.url and "static/uploads" in img_record.url:
            filename = img_record.url.split("/")[-1]
            local_path = f"static/uploads/{filename}"
            if os.path.exists(local_path):
                with open(local_path, "rb") as f: image_content = f.read()
        if not image_content:
            image_content = await run_in_threadpool(lambda: requests.get(img_record.url).content)
        resize_dims = options.get("resize") if options.get("resize") else None
        processor = ImageProcessor(image_content, resize_dims=resize_dims,operations=operations,autoDetect=autoDetect)
        processed_bytes, conf, steps, duration = await run_in_threadpool(processor.process)
        filename = f"processed/{img_record.user_id}/{image_id}.jpg"
        upload_res = upload_image_to_cloudinary(processed_bytes, filename)
        img_record.processed_url = upload_res.get("secure_url")
        img_record.confidence_scores = conf
        img_record.applied_steps = steps
        img_record.processing_time_ms = duration
        img_record.processing_status = "completed"
        await db.commit()
        await db.refresh(img_record)
        return {
            "status": "completed",
            "url": img_record.processed_url,
            "telemetry": {"confidence": conf, "steps": steps, "time_ms": duration}
        }
    except Exception as e:
        img_record.processing_status = "failed"
        await db.commit()
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/gallery")
async def get_gallery(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    query = select(Upload).where(Upload.user_id == current_user.id).order_by(Upload.created_at.desc()).limit(20)
    result = await db.execute(query)
    uploads = result.scalars().all()
    gallery = []
    for up in uploads:
        imgs = await db.execute(select(Image).where(Image.upload_id == up.id))
        images_list = imgs.scalars().all()
        gallery.append({
            "id": str(up.id),
            "status": up.status,
            "created_at": up.created_at,
            "metadata": {},
            "images": [
                {
                    "id": str(i.id),
                    "url": i.url,
                    "processed_url": i.processed_url,
                    "processing_status": i.processing_status,
                    "thumbnail_url": i.thumbnail_url or i.url,
                    "width": i.width,
                    "height": i.height,
                    "created_at": i.created_at
                } for i in images_list
            ]
        })
    return gallery
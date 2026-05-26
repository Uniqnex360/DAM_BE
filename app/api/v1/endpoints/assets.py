from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Body,Form
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import uuid
import requests
import asyncio
import os
import cv2
import numpy as np 
from app.api import deps
from app.db.session import get_db
from app.models.auth import User
from app.models.assets import Upload, Image
from app.services.media import upload_image_to_cloudinary
from app.services.image_processor import ImageProcessor
from app.services.quality_analyzer import analyze_image_quality
from app.schemas.asset import BatchUploadResponse, ImageResponse
from app.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from app.services.statistics import update_processing_stats
from app.models.project import Project
import traceback
import logging

logger = logging.getLogger("assets")
router = APIRouter()


@router.post("/analyze")
async def analyze_endpoint(request: AnalyzeRequest):
    try:
        result = await run_in_threadpool(analyze_image_quality, request)
        return {"analysis": result}
    except Exception as e:
        print(f"Analysis Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload", response_model=BatchUploadResponse)
async def upload_asset(
    files: list[UploadFile] = File(...),
    project_name:str=Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    project_id = None
    if project_name:
        existing = await db.execute(
            select(Project).where(Project.name == project_name, Project.user_id == current_user.id)
        )
        proj = existing.scalars().first()
        if not proj:
            proj = Project(user_id=current_user.id, name=project_name)
            db.add(proj)
            await db.commit()
            await db.refresh(proj)
        project_id = proj.id
    for file in files:
        if file.content_type not in ["image/jpeg", "image/png", "image/webp", "application/pdf"]:
            raise HTTPException(
                status_code=400, detail=f"Invalid file type: {file.filename}")
    upload_record = Upload(
    user_id=current_user.id, 
    status="uploaded", 
    project_id=project_id,
    metadata_obj={"project_name": project_name} if project_name else {}
)
    db.add(upload_record)
    await db.commit()
    await db.refresh(upload_record)
    successful_uploads=0
    results = []
    for file in files:
        try:
            file_ext = file.filename.rsplit(
                '.', 1)[-1] if '.' in file.filename else 'jpg'
            unique_filename = f"{current_user.id}/{uuid.uuid4()}.{file_ext}"
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
            results.append({
                "id": str(new_image.id),
                "name": new_image.name,
                "url": new_image.url,
                "width": new_image.width,
                "height": new_image.height
            })
            successful_uploads+=1
        except Exception as e:
            print(f"Failed to upload {file.filename}: {e}")
            continue
    if not results:
        await db.rollback()
        raise HTTPException(status_code=500, detail="All uploads failed")
    try:
        from app.db.session import AsyncSessionLocal
        async with AsyncSessionLocal() as stats_db:
            client_id = current_user.profile.client_id if current_user.profile else None
            
            for _ in range(successful_uploads):
                await update_processing_stats(stats_db, current_user.id, client_id, "upload", 0)
            
            await stats_db.commit()
            print(f"🔥 DEBUG: Recorded {successful_uploads} uploads")
    except Exception as e:
        print(f"❌ Upload stats failed (non-critical): {e}")
    return {"upload_id": str(upload_record.id), "images": results, "status": "uploaded"}

PROCESSING_SEMAPHORE = asyncio.Semaphore(
    int(os.getenv("MAX_CONCURRENT_PROCESSING", "2"))  # Never >4 on CPU
)

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
    async with PROCESSING_SEMAPHORE:
        result = await db.execute(select(Image).where(Image.id == image_id))
        img_record = result.scalars().first()
        if not img_record:
            raise HTTPException(status_code=404, detail="Image not found")
        client_id = current_user.profile.client_id if current_user.profile else None
        img_record.processing_status = "processing"
        await db.commit()
        try:
            image_content = None
            if "localhost" in img_record.url and "static/uploads" in img_record.url:
                filename = img_record.url.split("/")[-1]
                local_path = f"static/uploads/{filename}"
                if os.path.exists(local_path):
                    with open(local_path, "rb") as f:
                        image_content = f.read()
            if not image_content:
                image_content = await run_in_threadpool(lambda: requests.get(img_record.url).content)
            resize_dims = options.get("resize") if options.get("resize") else None
            processor = ImageProcessor(
                image_content, resize_dims=resize_dims, operations=operations, autoDetect=autoDetect)
            proc_result = await asyncio.to_thread(processor.process)
            processed_bytes = proc_result["image_bytes"]
            conf = proc_result["confidence"]
            steps = proc_result["steps_applied"]
            duration = proc_result["duration_ms"]
            resize_results = proc_result.get("resize_results")
            
            # Handle multiple marketplace outputs
            if resize_results and len(resize_results) > 0:
                outputs = []
                for res in resize_results:
                    img_data = res.get("image_bytes")
                    if isinstance(img_data, np.ndarray):
                        success, encoded = cv2.imencode(".jpg", img_data)
                        img_data = encoded.tobytes()
                    
                    marketplace_filename = f"processed/{current_user.id}/{image_id}_{res['id']}.jpg"
                    marketplace_upload = upload_image_to_cloudinary(img_data, marketplace_filename)
                    outputs.append({
                        "marketplace": res["id"],
                        "url": marketplace_upload.get("secure_url"),
                        "width": res["width"],
                        "height": res["height"]
                    })
                
                # Store primary processed URL as first marketplace image
                img_record.processed_url = outputs[0]["url"] if outputs else None
                img_record.confidence_scores = conf
                img_record.applied_steps = steps
                img_record.processing_time_ms = duration
                img_record.processing_status = "completed"
                await db.commit()
                await db.refresh(img_record)
                
                # Return multiple outputs
                return {
                    "status": "completed",
                    "outputs": outputs,
                    "original_image_id": image_id,
                    "telemetry": {"confidence": conf, "steps": steps, "time_ms": duration}
                }
            else:
                # Single output (original behavior)
                filename = f"processed/{img_record.user_id}/{image_id}.jpg"
                upload_res = upload_image_to_cloudinary(processed_bytes, filename)
                img_record.processed_url = upload_res.get("secure_url")
                img_record.confidence_scores = conf
                img_record.applied_steps = steps
                img_record.processing_time_ms = duration
                img_record.processing_status = "completed"
                await db.commit()
                await db.refresh(img_record)
                
                # Stats update (unchanged)
                try:
                    from app.db.session import AsyncSessionLocal
                    async with AsyncSessionLocal() as stats_db:
                        if steps:
                            for step in steps:
                                await update_processing_stats(stats_db, current_user.id, client_id, step, duration)
                                await stats_db.commit()
                        await stats_db.commit()
                except Exception as e:
                    print(f"❌ Stats update failed (non-critical): {e}")
                
                return {
                    "status": "completed",
                    "url": img_record.processed_url,
                    "name": img_record.name,
                    "telemetry": {"confidence": conf, "steps": steps, "time_ms": duration}
                }
                
        except Exception as e:
            logger.exception("❌ Image processing failed")

            traceback.print_exc()

            img_record.processing_status = "failed"
            await db.commit()

            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {str(e)}"
            )

@router.get("/gallery")
async def get_gallery(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    query = select(Upload).where(Upload.user_id == current_user.id).order_by(
        Upload.created_at.desc()).limit(20)
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
            "metadata": up.metadata_obj or {}, 
            "images": [
                {
                    "id": str(i.id),
                    "url": i.url,
                    "name": i.name,
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

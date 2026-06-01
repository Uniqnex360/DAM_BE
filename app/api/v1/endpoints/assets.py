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
from sqlalchemy import select, func

logger = logging.getLogger("assets")
logger.setLevel(logging.INFO)  
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
    original_dimensions: str = Form(None), 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    import json
    dimensions_map = {}
    if original_dimensions:
        try:
            dimensions_map = json.loads(original_dimensions)
            logger.info(f"Original dimensions received: {dimensions_map}")
        except json.JSONDecodeError:
            logger.warning("Failed to parse original_dimensions JSON")
    
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
            original_dims = dimensions_map.get(file.filename)
            
            # Build metadata with original dimensions if cropped
            image_metadata = {}
            if original_dims:
                image_metadata["original_dimensions"] = original_dims
                logger.info(f"Storing original dims for {file.filename}: {original_dims}")
            new_image = Image(
                upload_id=upload_record.id,
                user_id=current_user.id,
                url=result.get("secure_url"),
                thumbnail_url=result.get("secure_url"),
                width=result.get("width", 0),
                height=result.get("height", 0),
                processing_status="pending",
                name=file.filename,
                file_type=file.content_type,
                exif_data=image_metadata
            )
            db.add(new_image)
            await db.commit()
            await db.refresh(new_image)
            results.append({
                "id": str(new_image.id),
                "name": new_image.name,
                "url": new_image.url,
                "width": new_image.width,
                "height": new_image.height,
                "original_dimensions": original_dims  
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
        
        # 1. Update status to "processing" for both Image and Parent Upload
        img_record.processing_status = "processing"
        parent_res = await db.execute(select(Upload).where(Upload.id == img_record.upload_id))
        parent_upload = parent_res.scalars().first()
        if parent_upload and parent_upload.status != "processing":
            parent_upload.status = "processing"
            
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
            skip_crop = options.get("skip_crop", False)
            
            target_dimensions = None
            if img_record.exif_data and "original_dimensions" in img_record.exif_data:
                target_dimensions = img_record.exif_data["original_dimensions"]
                logger.info(f"Using original dimensions as target: {target_dimensions}")
                skip_crop = True
                
            processor = ImageProcessor(
                image_content, resize_dims=resize_dims, operations=operations, 
                autoDetect=autoDetect, skip_crop=skip_crop, target_dimensions=target_dimensions
            )
            proc_result = await asyncio.to_thread(processor.process)
            processed_bytes = proc_result["image_bytes"]
            conf = proc_result["confidence"]
            steps = proc_result["steps_applied"]
            duration = proc_result["duration_ms"]
            resize_results = proc_result.get("resize_results")
            
            # --- START SAVING RESULTS ---
            if resize_results and len(resize_results) > 0:
                outputs = []
                for res in resize_results:
                    img_data = res.get("image_bytes")
                    if isinstance(img_data, np.ndarray):
                        success, encoded = cv2.imencode(".jpg", img_data)
                        img_data = encoded.tobytes()
                    
                    m_filename = f"processed/{current_user.id}/{image_id}_{res['id']}.jpg"
                    m_upload = upload_image_to_cloudinary(img_data, m_filename)
                    outputs.append({
                        "marketplace": res["id"],
                        "url": m_upload.get("secure_url"),
                        "width": res["width"],
                        "height": res["height"]
                    })
                
                img_record.processed_url = outputs[0]["url"] if outputs else None
                img_record.processing_status = "completed"
                # Prep return data
                final_response = {
                    "status": "completed", "outputs": outputs, "original_image_id": image_id,
                    "telemetry": {"confidence": conf, "steps": steps, "time_ms": duration}
                }
            else:
                filename = f"processed/{img_record.user_id}/{image_id}.jpg"
                upload_res = upload_image_to_cloudinary(processed_bytes, filename)
                img_record.processed_url = upload_res.get("secure_url")
                img_record.processing_status = "completed"
                # Prep return data
                final_response = {
                    "status": "completed", "url": img_record.processed_url, "name": img_record.name,
                    "telemetry": {"confidence": conf, "steps": steps, "time_ms": duration}
                }

            # Common fields
            img_record.confidence_scores = conf
            img_record.applied_steps = steps
            img_record.processing_time_ms = duration
            
            await db.commit()
            
            # 🔥 NEW: UNIFIED BATCH COMPLETION CHECK
            # (Runs for both branches)
            upload_id = img_record.upload_id
            unfinished_query = await db.execute(
                select(func.count(Image.id))
                .where(Image.upload_id == upload_id)
                .where(Image.processing_status.in_(["pending", "processing"]))
            )
            if unfinished_query.scalar() == 0:
                parent_res = await db.execute(select(Upload).where(Upload.id == upload_id))
                parent_upload = parent_res.scalars().first()
                if parent_upload:
                    parent_upload.status = "completed"
                    await db.commit()
                    logger.info(f"✅ Batch {upload_id} fully completed.")

            # Stats update (Original functionality preserved)
            try:
                from app.db.session import AsyncSessionLocal
                async with AsyncSessionLocal() as stats_db:
                    if steps:
                        for step in steps:
                            await update_processing_stats(stats_db, current_user.id, client_id, step, duration)
                    await stats_db.commit()
            except Exception as e:
                print(f"❌ Stats update failed: {e}")

            return final_response
                
        except Exception as e:
            logger.exception("❌ Image processing failed")
            traceback.print_exc()
            img_record.processing_status = "failed"
            await db.commit()
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
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

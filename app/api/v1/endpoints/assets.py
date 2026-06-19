import json
import logging
import os
import traceback
import asyncio
import uuid

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Body, Form
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.api import deps
from app.db.session import get_db
from app.models.auth import User
from app.models.assets import Upload, Image
from app.models.project import Project
from app.services.media import upload_image_to_cloudinary
from app.services.quality_analyzer import analyze_image_quality
from app.services.statistics import update_processing_stats
from app.services.repositories import ImageRepository
from app.services.image_fetcher import ImageFetcher
from app.services.process_use_case import ProcessImageUseCase
from app.schemas.asset import BatchUploadResponse
from app.schemas.analysis import AnalyzeRequest
from app.api.utils.target_user_id import get_target_user_id

logger = logging.getLogger("assets")
logger.setLevel(logging.INFO)

router = APIRouter()

PROCESSING_SEMAPHORE = asyncio.Semaphore(
    int(os.getenv("MAX_CONCURRENT_PROCESSING", "2"))
)


@router.post("/analyze")
async def analyze_endpoint(request: AnalyzeRequest):
    try:
        result = await run_in_threadpool(analyze_image_quality, request)
        return {"analysis": result}
    except Exception as e:
        logger.error(f"Analysis Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload", response_model=BatchUploadResponse)
async def upload_asset(
    files: list[UploadFile] = File(...),
    project_name: str = Form(None),
    original_dimensions: str = Form(None),
    crop_settings: str = Form(None),
    user_id: str = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user),
):
    target_user_id = get_target_user_id(current_user, user_id)

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
            select(Project).where(
                Project.name == project_name,
                Project.user_id == target_user_id,
            )
        )
        proj = existing.scalars().first()
        if not proj:
            proj = Project(user_id=target_user_id, name=project_name)
            db.add(proj)
            await db.commit()
            await db.refresh(proj)
        project_id = proj.id

    crop_list = json.loads(crop_settings) if crop_settings else []
    crop_map = {item["filename"]: item for item in crop_list}

    upload_record = Upload(
        user_id=target_user_id,
        status="uploaded",
        project_id=project_id,
        metadata_obj={"project_name": project_name} if project_name else {},
    )
    db.add(upload_record)
    await db.commit()
    await db.refresh(upload_record)

    successful_uploads = 0
    results = []

    for file in files:
        if file.content_type not in [
            "image/jpeg",
            "image/png",
            "image/webp",
            "application/pdf",
            "image/avif",
        ]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}",
            )

        try:
            image_metadata = {}

            crop_info = crop_map.get(file.filename)
            if crop_info:
                image_metadata["crop_mode"] = crop_info.get("cropMode")
                image_metadata["target_aspect_ratio"] = crop_info.get(
                    "targetAspectRatio")

            file_ext = file.filename.rsplit(
                ".", 1)[-1] if "." in file.filename else "jpg"
            unique_filename = f"{target_user_id}/{uuid.uuid4()}.{file_ext}"
            file_content = await file.read()
            result = upload_image_to_cloudinary(file_content, unique_filename)

            original_dims = dimensions_map.get(file.filename)
            if original_dims:
                image_metadata["original_dimensions"] = original_dims
                logger.info(
                    f"Storing original dims for {file.filename}: {original_dims}")

            new_image = Image(
                upload_id=upload_record.id,
                user_id=target_user_id,
                url=result.get("secure_url"),
                thumbnail_url=result.get("secure_url"),
                width=result.get("width", 0),
                height=result.get("height", 0),
                processing_status="pending",
                name=file.filename,
                file_type=file.content_type,
                exif_data=image_metadata,
            )
            db.add(new_image)
            await db.commit()
            await db.refresh(new_image)

            results.append(
                {
                    "id": str(new_image.id),
                    "name": new_image.name,
                    "url": new_image.url,
                    "width": new_image.width,
                    "height": new_image.height,
                    "original_dimensions": original_dims,
                }
            )
            successful_uploads += 1

        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}")
            continue

    if not results:
        await db.rollback()
        raise HTTPException(status_code=500, detail="All uploads failed")

    try:
        from app.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as stats_db:
            for _ in range(successful_uploads):
                await update_processing_stats(stats_db, current_user.id, "upload", 0)
            await stats_db.commit()
            logger.debug(f"Recorded {successful_uploads} upload stats")
    except Exception as e:
        logger.warning(f"Upload stats failed (non-critical): {e}")

    return {
        "upload_id": str(upload_record.id),
        "images": results,
        "status": "uploaded",
    }


@router.post("/{image_id}/process")
async def process_image_asset(
    image_id: str,
    operations: list = Body(default=[], embed=True),
    options: dict = Body(default={}, embed=True),
    autoDetect: bool = Body(default=False, embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user),
):
    target_user_id = get_target_user_id(current_user, None)

    async with PROCESSING_SEMAPHORE:
        repo = ImageRepository(db)
        img_record = await repo.get_image(image_id)

        if not img_record:
            raise HTTPException(status_code=404, detail="Image not found")
        record_owner = str(img_record.user_id)
        requester_id = str(target_user_id)  
        if record_owner != requester_id:
            if getattr(current_user, "role", None) != "admin":
                raise HTTPException(status_code=403, detail="Not authorized")

        use_case = ProcessImageUseCase(repo, ImageFetcher())

        try:
            return await use_case.execute(
                image_id=image_id,
                target_user_id=target_user_id,
                operations=operations,
                options=options,
                autoDetect=autoDetect,
            )
        except Exception:
            logger.exception("Image processing failed")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Processing failed")


@router.get("/gallery")
async def get_gallery(
    user_id: str = None,
    all: bool = False,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user),
):
    try:
        is_admin = current_user.role == "admin"

        if all and is_admin:
            query = select(Upload).order_by(Upload.created_at.desc()).limit(50)
        else:
            target_user_id = get_target_user_id(current_user, user_id)
            query = (
                select(Upload)
                .where(Upload.user_id == target_user_id)
                .order_by(Upload.created_at.desc())
                .limit(20)
            )

        result = await db.execute(query)
        uploads = result.scalars().all()
        gallery = []

        for up in uploads:
            imgs = await db.execute(select(Image).where(Image.upload_id == up.id))
            images_list = imgs.scalars().all()

            gallery.append(
                {
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
                            "created_at": i.created_at,
                        }
                        for i in images_list
                    ],
                }
            )

        return gallery

    except Exception as e:
        logger.exception("Error fetching gallery")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch gallery: {str(e)}",
        )

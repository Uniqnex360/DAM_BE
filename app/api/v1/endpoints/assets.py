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
import tempfile
import zipfile
import shutil
from typing import List
import httpx
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from sqlalchemy import select
from app.models.assets import Upload, Image
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
    failed_uploads = []
    results = []
    for file in files:
        if file.content_type not in [
            "image/jpeg",
            "image/png",
            "image/webp",
            "application/pdf",
            "image/avif",
        ]:
            failed_uploads.append({
                'filename': file.filename,
                'error': f"Invalid file type:{file.content_type}"
            })
            continue

        try:
            image_metadata = {}
            crop_info = crop_map.get(file.filename)
            applied_steps_init = []
            if crop_info:
                image_metadata["crop_mode"] = crop_info.get("cropMode")
                image_metadata["target_aspect_ratio"] = crop_info.get(
                    "targetAspectRatio")
                applied_steps_init.append("smart_crop")
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
                applied_steps=applied_steps_init
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
            failed_uploads.append({
                'filename': file.filename,
                'error': str(e)
            })
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
        "failed_files": failed_uploads,
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


@router.get("/projects/{project_id}/download-zip")
async def download_project_zip(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user),
):

    result = await db.execute(
        select(Upload).where(Upload.id == project_id)
    )
    upload = result.scalars().first()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    if str(upload.user_id) != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized")
    images_result = await db.execute(
        select(Image).where(Image.upload_id == upload.id)
    )
    images: List[Image] = images_result.scalars().all()
    if not images:
        raise HTTPException(
            status_code=404, detail="No images for this project")
    temp_dir = tempfile.mkdtemp(prefix=f"project-{project_id}-")

    project_name: str | None = None

    if upload.project_id:
        project_result = await db.execute(
            select(Project).where(Project.id == upload.project_id)
        )
        project = project_result.scalars().first()
        if project and project.name:
            project_name = project.name

    if not project_name and isinstance(upload.metadata_obj, dict):
        project_name = upload.metadata_obj.get("project_name")

    display_name = project_name or f"session-{upload.id}"

    safe_name = "".join(
        c if c not in ('\\', '/', ':', '*', '?', '"', '<', '>', '|')
        else "_"
        for c in display_name
    )

    temp_dir = tempfile.mkdtemp(prefix=f"session-{upload.id}-")
    zip_path = os.path.join(temp_dir, f"{safe_name}.zip")
    async with httpx.AsyncClient(timeout=60.0) as client:
        seen_names: dict[str, int] = {}
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for img in images:
                source_url = img.processed_url or img.url
                if not source_url:
                    continue
                try:
                    resp = await client.get(source_url)
                    resp.raise_for_status()
                except httpx.HTTPError:
                    continue

                if img.name:
                    filename = img.name
                else:
                    filename = source_url.rstrip(
                        "/").split("/")[-1] or f"{img.id}.jpg"

                dot_idx = filename.rfind(".")
                if dot_idx > 0:
                    base, ext = filename[:dot_idx], filename[dot_idx:]
                else:
                    base, ext = filename, ""

                arcname = f"{base}_output{ext}"
                if arcname in seen_names:
                    seen_names[arcname] += 1
                    arcname = f"{base}_output_{seen_names[arcname]}{ext}"
                else:
                    seen_names[arcname] = 0

                zf.writestr(arcname, resp.content)
    if not os.path.exists(zip_path):
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500, detail="Failed to create zip file")
    return FileResponse(
        path=zip_path,
        filename=f"{safe_name}.zip",
        media_type="application/zip",
        background=BackgroundTask(
            lambda: shutil.rmtree(temp_dir, ignore_errors=True)),
    )


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
@router.delete("/{image_id}")
async def delete_image(
    image_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user),
):
    """
    Delete an image from the database and Cloudinary.
    Only the owner or an admin can delete the image.
    """
    try:
        
        result = await db.execute(
            select(Image).where(Image.id == image_id)
        )
        image = result.scalars().first()
        
        if not image:
            raise HTTPException(
                status_code=404, 
                detail=f"Image with ID {image_id} not found"
            )
        
        
        is_owner = str(image.user_id) == str(current_user.id)
        is_admin = getattr(current_user, "role", None) == "admin"
        
        if not is_owner and not is_admin:
            raise HTTPException(
                status_code=403, 
                detail="Not authorized to delete this image"
            )
        
        
        urls_to_delete = []
        if image.url:
            urls_to_delete.append(image.url)
        if image.processed_url and image.processed_url != image.url:
            urls_to_delete.append(image.processed_url)
        if image.thumbnail_url and image.thumbnail_url not in urls_to_delete:
            urls_to_delete.append(image.thumbnail_url)
        
        
        cloudinary_errors = []
        for url in urls_to_delete:
            try:
                
                public_id = extract_cloudinary_public_id(url)
                if public_id:
                    await delete_from_cloudinary(public_id)
                    logger.info(f"Deleted from Cloudinary: {public_id}")
            except Exception as e:
                logger.error(f"Failed to delete from Cloudinary: {url} - {str(e)}")
                cloudinary_errors.append(str(e))
        
        
        image_name = image.name
        await db.delete(image)
        await db.commit()
        
        
        logger.info(
            f"Image {image_id} ({image_name}) deleted by user {current_user.id}"
        )
        
        response_data = {
            "message": f"Image '{image_name}' deleted successfully",
            "image_id": image_id,
            "deleted_from_cloudinary": len(urls_to_delete) - len(cloudinary_errors),
            "cloudinary_errors": cloudinary_errors if cloudinary_errors else None
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting image {image_id}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete image: {str(e)}"
        )


@router.delete("/batch")
async def batch_delete_images(
    image_ids: List[str] = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user),
):
    """
    Delete multiple images in batch.
    Only owners or admins can delete images.
    """
    if not image_ids:
        raise HTTPException(status_code=400, detail="No image IDs provided")
    
    if len(image_ids) > 50:  
        raise HTTPException(
            status_code=400, 
            detail="Maximum 50 images can be deleted at once"
        )
    
    try:
        results = {
            "successful": [],
            "failed": [],
            "total_requested": len(image_ids)
        }
        
        for image_id in image_ids:
            try:
                
                result = await db.execute(
                    select(Image).where(Image.id == image_id)
                )
                image = result.scalars().first()
                
                if not image:
                    results["failed"].append({
                        "image_id": image_id,
                        "error": "Image not found"
                    })
                    continue
                
                
                is_owner = str(image.user_id) == str(current_user.id)
                is_admin = getattr(current_user, "role", None) == "admin"
                
                if not is_owner and not is_admin:
                    results["failed"].append({
                        "image_id": image_id,
                        "error": "Not authorized"
                    })
                    continue
                
                
                urls_to_delete = []
                if image.url:
                    urls_to_delete.append(image.url)
                if image.processed_url and image.processed_url != image.url:
                    urls_to_delete.append(image.processed_url)
                if image.thumbnail_url and image.thumbnail_url not in urls_to_delete:
                    urls_to_delete.append(image.thumbnail_url)
                
                for url in urls_to_delete:
                    try:
                        public_id = extract_cloudinary_public_id(url)
                        if public_id:
                            await delete_from_cloudinary(public_id)
                    except Exception as e:
                        logger.error(f"Cloudinary deletion failed for {url}: {e}")
                
                
                image_name = image.name
                await db.delete(image)
                
                results["successful"].append({
                    "image_id": image_id,
                    "name": image_name
                })
                
            except Exception as e:
                logger.error(f"Failed to delete image {image_id}: {e}")
                results["failed"].append({
                    "image_id": image_id,
                    "error": str(e)
                })
        
        
        await db.commit()
        
        logger.info(
            f"Batch deletion: {len(results['successful'])} succeeded, "
            f"{len(results['failed'])} failed out of {len(image_ids)} requested"
        )
        
        return results
        
    except Exception as e:
        logger.exception("Batch deletion failed")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Batch deletion failed: {str(e)}"
        )


@router.delete("/upload/{upload_id}")
async def delete_upload_and_images(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user),
):
    
    try:
        
        result = await db.execute(
            select(Upload).where(Upload.id == upload_id)
        )
        upload = result.scalars().first()
        
        if not upload:
            raise HTTPException(
                status_code=404,
                detail=f"Upload session {upload_id} not found"
            )
        
        
        is_owner = str(upload.user_id) == str(current_user.id)
        is_admin = getattr(current_user, "role", None) == "admin"
        
        if not is_owner and not is_admin:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to delete this upload session"
            )
        
        
        images_result = await db.execute(
            select(Image).where(Image.upload_id == upload.id)
        )
        images = images_result.scalars().all()
        
        
        deleted_count = 0
        failed_count = 0
        
        for image in images:
            try:
                
                urls_to_delete = []
                if image.url:
                    urls_to_delete.append(image.url)
                if image.processed_url and image.processed_url != image.url:
                    urls_to_delete.append(image.processed_url)
                if image.thumbnail_url and image.thumbnail_url not in urls_to_delete:
                    urls_to_delete.append(image.thumbnail_url)
                
                for url in urls_to_delete:
                    try:
                        public_id = extract_cloudinary_public_id(url)
                        if public_id:
                            await delete_from_cloudinary(public_id)
                    except Exception as e:
                        logger.error(f"Cloudinary deletion failed: {e}")
                
                
                await db.delete(image)
                deleted_count += 1
                
            except Exception as e:
                logger.error(f"Failed to delete image {image.id}: {e}")
                failed_count += 1
        
        
        await db.delete(upload)
        await db.commit()
        
        logger.info(
            f"Upload session {upload_id} deleted: "
            f"{deleted_count} images deleted, {failed_count} failed"
        )
        
        return {
            "message": f"Upload session deleted successfully",
            "upload_id": upload_id,
            "images_deleted": deleted_count,
            "images_failed": failed_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting upload session {upload_id}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete upload session: {str(e)}"
        )




def extract_cloudinary_public_id(url: str) -> str:
    
    try:
        if not url or "cloudinary" not in url:
            return None
        
        
        parts = url.split("upload/")
        if len(parts) < 2:
            return None
        
        
        after_upload = parts[1]
        version_parts = after_upload.split("/", 1)
        
        if len(version_parts) < 2:
            return None
        
        
        public_id_with_ext = version_parts[1]
        public_id = public_id_with_ext.rsplit(".", 1)[0]
        
        return public_id
        
    except Exception as e:
        logger.error(f"Failed to extract public_id from URL {url}: {e}")
        return None


async def delete_from_cloudinary(public_id: str) -> bool:
   
    import cloudinary
    import cloudinary.uploader
    
    try:
        result = cloudinary.uploader.destroy(public_id)
        
        if result.get("result") == "ok":
            logger.info(f"Successfully deleted from Cloudinary: {public_id}")
            return True
        else:
            logger.warning(
                f"Cloudinary deletion returned: {result.get('result')} "
                f"for {public_id}"
            )
            return False
            
    except Exception as e:
        logger.error(f"Cloudinary deletion error for {public_id}: {e}")
        raise
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from app.api import deps
from app.db.session import get_db
from app.models.assets import Image
from app.models.auth import User
import logging
from app.api.utils.target_user_id import get_target_user_id
logger=logging.getLogger('dashboard')
router = APIRouter()
@router.get("/batch/{upload_id}")
async def get_batch_analytics(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    summary_query = select(
        func.count(Image.id).label("total"),
        func.count(Image.id).filter(Image.processing_status == "completed").label("processed"),
        func.count(Image.id).filter(Image.processing_status == "failed").label("failed"),
        func.avg(Image.processing_time_ms).label("avg_time")
    ).where(Image.upload_id == upload_id)
    summary_res = await db.execute(summary_query)
    summary = summary_res.first()
    step_query = text("""
        SELECT step, COUNT(*) as count
        FROM images, jsonb_array_elements_text(applied_steps) as step
        WHERE upload_id = :uid
        GROUP BY step
    """)
    step_res = await db.execute(step_query, {"uid": upload_id})
    steps_dist = {row[0]: row[1] for row in step_res}
    conf_query = text("""
        SELECT 
            AVG((confidence_scores->>'bg_clean')::float) as bg_clean,
            AVG((confidence_scores->>'shadow')::float) as shadow,
            AVG((confidence_scores->>'crop')::float) as crop
        FROM images
        WHERE upload_id = :uid AND processing_status = 'completed'
    """)
    conf_res = await db.execute(conf_query, {"uid": upload_id})
    conf_avg = conf_res.mappings().first()
    return {
        "batch_id": upload_id,
        "summary": {
            "total": summary.total,
            "processed": summary.processed,
            "failed": summary.failed,
            "avg_processing_time_sec": round((summary.avg_time or 0) / 1000, 2)
        },
        "enhancements_distribution": steps_dist,
        "average_confidence": conf_avg
    }
# @router.get("/overview")
# async def get_user_overview(
#     user_id: str = None,
#     all: bool = False,
#     db: AsyncSession = Depends(get_db),
#     current_user: User = Depends(deps.get_current_user)
# ):
#     is_admin = current_user.role == "admin"

#     if all and is_admin:
#         target_user_id = None 
#         logger.info(f"Admin {current_user.email} viewing ALL users data")
#     else:
#         target_user_id = get_target_user_id(current_user, user_id)

#     summary_query = select(
#         func.count(Image.id).label("total"),
#         func.count(Image.id).filter(Image.processing_status == "completed").label("processed"),
#         func.count(Image.id).filter(Image.processing_status == "failed").label("failed"),
#         func.avg(Image.processing_time_ms).label("avg_time")
#     ).where(Image.user_id == target_user_id)
#     summary_res = await db.execute(summary_query)
#     summary = summary_res.first()
#     step_query = text("""
#         SELECT step, COUNT(*) as count
#         FROM images, jsonb_array_elements_text(applied_steps) as step
#         WHERE user_id = :uid
#         GROUP BY step
#     """)
#     step_res = await db.execute(step_query, {"uid": target_user_id})
#     steps_dist = {row[0]: row[1] for row in step_res}
#     recent_query = select(Image).where(Image.user_id == target_user_id).order_by(
#         Image.created_at.desc()).limit(10)
#     recent_res = await db.execute(recent_query)
#     recent_images = recent_res.scalars().all()
#     return {
#         "summary": {
#             "totalImagesUploaded": summary.total or 0,
#             "totalImagesProcessed": summary.processed or 0,
#             "failed": summary.failed or 0,
#             "avgProcessingTimeMs": float(summary.avg_time or 0)
#         },
#         "operationCounts": steps_dist,
#         "recentOperations": [
#             {
#                 "id": str(img.id),
#                 "operationType": " + ".join(img.applied_steps) if img.applied_steps else "Upload Only",
#                 "status": img.processing_status,
#                 "createdAt": img.created_at,
#                 "processingTimeMs": img.processing_time_ms,
#                 "fileName": img.name or "Untitled Image",
#                 "thumbnailUrl": img.processed_url if img.processed_url else img.url
#             } for img in recent_images
#         ]
#     }
    

@router.get("/overview")
async def get_user_overview(
    user_id: str = None,
    # ✅ ADD THIS (default False, so existing calls work as before)
    all: bool = False,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    is_admin = current_user.role == "admin"

    # ✅ Determine target user - existing logic unchanged when all=False
    if all and is_admin:
        target_user_id = None  # Special case: no user filter
        logger.info(f"Admin {current_user.email} viewing ALL users data")
    else:
        target_user_id = get_target_user_id(current_user, user_id)

    # ✅ Build query with conditional filter (existing queries unchanged)
    summary_query = select(
        func.count(Image.id).label("total"),
        func.count(Image.id).filter(Image.processing_status ==
                                    "completed").label("processed"),
        func.count(Image.id).filter(
            Image.processing_status == "failed").label("failed"),
        func.avg(Image.processing_time_ms).label("avg_time")
    )
    if target_user_id is not None:
        summary_query = summary_query.where(Image.user_id == target_user_id)
    summary_res = await db.execute(summary_query)
    summary = summary_res.first()

    # ✅ Step query - conditional filter
    if target_user_id is not None:
        step_query = text("""
            SELECT step, COUNT(*) as count
            FROM images, jsonb_array_elements_text(applied_steps) as step
            WHERE user_id = :uid
            GROUP BY step
        """)
        step_res = await db.execute(step_query, {"uid": target_user_id})
    else:
        step_query = text("""
            SELECT step, COUNT(*) as count
            FROM images, jsonb_array_elements_text(applied_steps) as step
            GROUP BY step
        """)
        step_res = await db.execute(step_query)
    steps_dist = {row[0]: row[1] for row in step_res}

    # ✅ Recent query - conditional filter
    recent_query = select(Image).order_by(Image.created_at.desc()).limit(10)
    if target_user_id is not None:
        recent_query = recent_query.where(Image.user_id == target_user_id)
    recent_res = await db.execute(recent_query)
    recent_images = recent_res.scalars().all()

    return {
        "summary": {
            "totalImagesUploaded": summary.total or 0,
            "totalImagesProcessed": summary.processed or 0,
            "failed": summary.failed or 0,
            "avgProcessingTimeMs": float(summary.avg_time or 0)
        },
        "operationCounts": steps_dist,
        "recentOperations": [
            {
                "id": str(img.id),
                "operationType": " + ".join(img.applied_steps) if img.applied_steps else "Upload Only",
                "status": img.processing_status,
                "createdAt": img.created_at,
                "processingTimeMs": img.processing_time_ms,
                "fileName": img.name or "Untitled Image",
                "thumbnailUrl": img.processed_url if img.processed_url else img.url
            } for img in recent_images
        ]
    }

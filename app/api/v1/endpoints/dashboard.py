from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from app.api import deps
from app.db.session import get_db
from app.models.assets import Image
from app.models.auth import User

router = APIRouter()

@router.get("/batch/{upload_id}")
async def get_batch_analytics(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
   
    
    # 1. Summary Stats
    summary_query = select(
        func.count(Image.id).label("total"),
        func.count(Image.id).filter(Image.processing_status == "completed").label("processed"),
        func.count(Image.id).filter(Image.processing_status == "failed").label("failed"),
        func.avg(Image.processing_time_ms).label("avg_time")
    ).where(Image.upload_id == upload_id)
    
    summary_res = await db.execute(summary_query)
    summary = summary_res.first()

    # 2. Enhancement Distribution (Postgres JSONB Aggregation)
    # This unpacks the ["crop", "bg"] array and counts occurrences
    step_query = text("""
        SELECT step, COUNT(*) as count
        FROM images, jsonb_array_elements_text(applied_steps) as step
        WHERE upload_id = :uid
        GROUP BY step
    """)
    step_res = await db.execute(step_query, {"uid": upload_id})
    steps_dist = {row[0]: row[1] for row in step_res}

    # 3. Average Confidence per Metric
    # Extracts values from the JSON object: confidence_scores->>'bg_clean'
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
@router.get("/overview")
async def get_user_overview(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """
    Aggregates stats AND recent activity with Image Details.
    """
    
    # 1. Summary Stats
    summary_query = select(
        func.count(Image.id).label("total"),
        func.count(Image.id).filter(Image.processing_status == "completed").label("processed"),
        func.count(Image.id).filter(Image.processing_status == "failed").label("failed"),
        func.avg(Image.processing_time_ms).label("avg_time")
    ).where(Image.user_id == current_user.id)
    
    summary_res = await db.execute(summary_query)
    summary = summary_res.first()

    # 2. Enhancement Distribution
    step_query = text("""
        SELECT step, COUNT(*) as count
        FROM images, jsonb_array_elements_text(applied_steps) as step
        WHERE user_id = :uid
        GROUP BY step
    """)
    step_res = await db.execute(step_query, {"uid": current_user.id})
    steps_dist = {row[0]: row[1] for row in step_res}

    # 3. Recent Operations (Last 10)
    recent_query = select(Image).where(Image.user_id == current_user.id)\
        .order_by(Image.created_at.desc()).limit(10)
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
                # NEW FIELDS:
                "fileName": img.name or "Untitled Image",
                # Prefer the processed version for thumbnail, fallback to original
                "thumbnailUrl": img.processed_url if img.processed_url else img.url
            } for img in recent_images
        ]
    }
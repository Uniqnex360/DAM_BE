from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import attributes
from app.models.processing import ProcessingStatistic
from datetime import date
import logging
from typing import Optional
import uuid
logger = logging.getLogger(__name__)
async def update_processing_stats(
    db: AsyncSession,
    user_id: str,
    operation: str,
    processing_time_ms: int
):
    try:
        today = date.today()
        result = await db.execute(
            select(ProcessingStatistic).where(
                ProcessingStatistic.user_id == user_id,
                func.date(ProcessingStatistic.date) == today
            )
        )
        stats = result.scalars().first()
        if not stats:
            print("🔥 DEBUG: Creating new stats record for today")
            stats = ProcessingStatistic(
                user_id=user_id,
                operation_counts={},
                daily_breakdown={},
                total_images_uploaded=0,
                total_images_processed=0,
                total_processing_time_ms=0,
                date=today  
            )
            db.add(stats)
        if operation == 'upload':
            stats.total_images_uploaded = (stats.total_images_uploaded or 0) + 1
        else:
            stats.total_images_processed = (stats.total_images_processed or 0) + 1
            stats.total_processing_time_ms += (processing_time_ms or 0)
        ops = dict(stats.operation_counts or {})
        ops[operation] = ops.get(operation, 0) + 1
        stats.operation_counts = ops
        daily = dict(stats.daily_breakdown or {})
        today_str = str(today)
        if today_str not in daily:
            daily[today_str] = {"total": 0}
        daily[today_str]["total"] = daily[today_str].get("total", 0) + 1
        daily[today_str][operation] = daily[today_str].get(operation, 0) + 1
        if operation != 'upload':
            daily[today_str]["time_ms"] = daily[today_str].get("time_ms", 0) + (processing_time_ms or 0)
        stats.daily_breakdown = daily
        attributes.flag_modified(stats, "operation_counts")
        attributes.flag_modified(stats, "daily_breakdown")
        logger.info(f"Stats updated: {operation}")
    except Exception as e:
        logger.error(f"Failed to update processing stats: {e}")
        raise
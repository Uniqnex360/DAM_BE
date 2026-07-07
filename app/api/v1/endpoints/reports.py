from fastapi import APIRouter, Depends,HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from app.api import deps
from app.db.session import get_db
from app.models.auth import User
from app.models.processing import ProcessingStatistic
from app.models.assets import Image
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
# @router.get("/processing")
# async def get_processing_report(
#     db: AsyncSession = Depends(get_db),
#     current_user: User = Depends(deps.get_current_user)
# ):
#     try:
#         result = await db.execute(
#             select(ProcessingStatistic)
#             .where(ProcessingStatistic.user_id == current_user.id)
#             .order_by(ProcessingStatistic.date.desc())
#             .limit(30)
#         )
#         stats = result.scalars().all()
        
#         # Aggregate all-time stats
#         total_processed = sum(s.total_images_processed or 0 for s in stats)
#         total_time = sum(s.total_processing_time_ms or 0 for s in stats)
#         total_uploaded = sum(s.total_images_uploaded or 0 for s in stats)
        
#         # Aggregate operation counts
#         all_ops = {}
#         for s in stats:
#             if s.operation_counts:
#                 for op, count in s.operation_counts.items():
#                     all_ops[op] = all_ops.get(op, 0) + count
        
#         return {
#             "total_images_uploaded": total_uploaded,
#             "total_images_processed": total_processed,
#             "total_processing_time_ms": total_time,
#             "avg_processing_time_ms": total_time // total_processed if total_processed > 0 else 0,
#             "operation_counts": all_ops,
#             "daily_breakdown": [{
#                 "date": str(s.date.date()) if s.date else None,
#                 "total_processed": s.total_images_processed or 0,
#                 "total_uploaded": s.total_images_uploaded or 0,
#                 "operations": s.operation_counts or {},
#                 "time_ms": s.total_processing_time_ms or 0
#             } for s in stats]
#         }
#     except Exception as e:
#         print(f"Error fetching processing report: {e}")
#         raise HTTPException(status_code=500, detail="Failed to fetch processing report")
@router.get("/processing")
async def get_processing_report(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """
    Generates a live, accurate processing report for the current user
    by querying the raw image data directly. This ensures all stats
    and operations are up-to-date.
    """
    try:
        # ✅ QUERY 1: Get the TRUE all-time summary from the Image table.
        # This part is correct and unchanged.
        all_time_summary_query = select(
            func.count(Image.id).label("total_uploaded"),
            func.count(Image.id).filter(Image.processing_status == "completed").label("total_processed"),
            func.sum(Image.processing_time_ms).label("total_time")
        ).where(Image.user_id == current_user.id)
        
        summary_result = await db.execute(all_time_summary_query)
        summary = summary_result.first()

        # ✅ QUERY 2: Get the TRUE all-time operation counts.
        # This part is also correct and unchanged.
        op_counts_query = text("""
            SELECT step, COUNT(*) as count
            FROM images, jsonb_array_elements_text(applied_steps) as step
            WHERE user_id = :uid
            GROUP BY step
        """)
        op_counts_result = await db.execute(op_counts_query, {"uid": current_user.id})
        operation_counts = {row[0]: row[1] for row in op_counts_result}

        # ✅ QUERY 3 (CORRECTED): Get the complete daily breakdown.
        # This new query uses CTEs to avoid the GroupingError.
        daily_breakdown_query = text("""
            WITH daily_summary AS (
                -- First, get the basic counts for each day
                SELECT
                    DATE_TRUNC('day', created_at AT TIME ZONE 'UTC')::date as date,
                    COUNT(id) as total_uploaded,
                    COUNT(id) FILTER (WHERE processing_status = 'completed') as total_processed,
                    SUM(processing_time_ms) as total_time_ms
                FROM images
                WHERE user_id = :uid
                GROUP BY date
            ), daily_operations AS (
                -- Second, correctly calculate the operations counts for each day
                SELECT
                    date,
                    jsonb_object_agg(step, step_count) as operations
                FROM (
                    SELECT
                        DATE_TRUNC('day', i.created_at AT TIME ZONE 'UTC')::date as date,
                        s.step,
                        COUNT(*) as step_count
                    FROM images i, jsonb_array_elements_text(i.applied_steps) s(step)
                    WHERE i.user_id = :uid
                    GROUP BY date, s.step
                ) as daily_step_counts
                GROUP BY date
            )
            -- Finally, join the two sets of results together
            SELECT
                ds.date,
                ds.total_uploaded,
                ds.total_processed,
                COALESCE(ds.total_time_ms, 0) as time_ms,
                COALESCE(do.operations, '{}'::jsonb) as operations
            FROM daily_summary ds
            LEFT JOIN daily_operations do ON ds.date = do.date
            ORDER BY ds.date DESC
        """)
        
        daily_result = await db.execute(daily_breakdown_query, {"uid": current_user.id})
        daily_stats_rows = daily_result.mappings().all()

        total_uploaded = summary.total_uploaded or 0
        total_processed = summary.total_processed or 0
        total_time = summary.total_time or 0

        return {
            "total_images_uploaded": total_uploaded,
            "total_images_processed": total_processed,
            "total_processing_time_ms": total_time,
            "avg_processing_time_ms": total_time // total_processed if total_processed > 0 else 0,
            "operation_counts": operation_counts,
            # The result is already perfectly formatted, just convert to a list of dicts
            "daily_breakdown": [dict(row) for row in daily_stats_rows]
        }
    except Exception as e:
        logger.error(f"Error fetching live processing report for user {current_user.id}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch processing report")
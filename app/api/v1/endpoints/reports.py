from fastapi import APIRouter, Depends,HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.api import deps
from app.db.session import get_db
from app.models.auth import User
from app.models.processing import ProcessingStatistic
router = APIRouter()
@router.get("/processing")
async def get_processing_report(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    try:
        result = await db.execute(
            select(ProcessingStatistic)
            .where(ProcessingStatistic.user_id == current_user.id)
            .order_by(ProcessingStatistic.date.desc())
            .limit(30)
        )
        stats = result.scalars().all()
        
        # Aggregate all-time stats
        total_processed = sum(s.total_images_processed or 0 for s in stats)
        total_time = sum(s.total_processing_time_ms or 0 for s in stats)
        total_uploaded = sum(s.total_images_uploaded or 0 for s in stats)
        
        # Aggregate operation counts
        all_ops = {}
        for s in stats:
            if s.operation_counts:
                for op, count in s.operation_counts.items():
                    all_ops[op] = all_ops.get(op, 0) + count
        
        return {
            "total_images_uploaded": total_uploaded,
            "total_images_processed": total_processed,
            "total_processing_time_ms": total_time,
            "avg_processing_time_ms": total_time // total_processed if total_processed > 0 else 0,
            "operation_counts": all_ops,
            "daily_breakdown": [{
                "date": str(s.date.date()) if s.date else None,
                "total_processed": s.total_images_processed or 0,
                "total_uploaded": s.total_images_uploaded or 0,
                "operations": s.operation_counts or {},
                "time_ms": s.total_processing_time_ms or 0
            } for s in stats]
        }
    except Exception as e:
        print(f"Error fetching processing report: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch processing report")
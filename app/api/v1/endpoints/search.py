from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_, func, text, desc
from sqlalchemy.orm import joinedload
from typing import Optional, List
from app.api import deps
from app.db.session import get_db
from app.models.assets import Image, Upload
from app.models.project import Project
from app.models.auth import User
from app.api.utils.target_user_id import get_target_user_id
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/images")
async def search_images(
    q: Optional[str] = Query(None, description="Search query for image name or project"),
    project_name: Optional[str] = Query(None, description="Filter by project name"),
    status: Optional[str] = Query(None, description="Filter by processing status"),
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    aspect_ratio: Optional[str] = Query(None, description="Filter by aspect ratio (1:1, 16:9, etc.)"),
    crop_mode: Optional[str] = Query(None, description="Filter by crop mode"),
    operations: Optional[str] = Query(None, description="Filter by operations (comma-separated)"),
    has_output: Optional[bool] = Query(None, description="Filter images with processed output"),
    date_from: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)"),
    sort_by: str = Query("newest", description="Sort by: newest, oldest, name, processing_time"),
    limit: int = Query(50, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    user_id: Optional[str] = Query(None, description="Admin: search specific user"),
    all_users: bool = Query(False, description="Admin: search all users"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """
    Production-ready image search with comprehensive filters
    """
    try:
        # Permission handling
        is_admin = current_user.role == "admin"
        target_user_id = None
        
        if all_users and is_admin:
            target_user_id = None  # Search all users
            logger.info(f"Admin {current_user.email} searching all users")
        elif user_id and is_admin:
            target_user_id = user_id
            logger.info(f"Admin {current_user.email} searching user {user_id}")
        else:
            target_user_id = current_user.id

        # Base query with joins for project data
        query = select(
            Image.id,
            Image.name,
            Image.url,
            Image.thumbnail_url,
            Image.processed_url,
            Image.width,
            Image.height,
            Image.processing_status,
            Image.file_type,
            Image.created_at,
            Image.processing_time_ms,
            Image.applied_steps,
            Image.exif_data,
            Upload.metadata_obj.label('upload_metadata'),
            Project.name.label('project_name')
        ).select_from(
            Image
        ).join(
            Upload, Image.upload_id == Upload.id
        ).outerjoin(
            Project, Upload.project_id == Project.id
        )

        # User filter
        if target_user_id is not None:
            query = query.where(Image.user_id == target_user_id)

        # Search query filter (name + project)
        if q:
            search_term = f"%{q.lower()}%"
            search_conditions = [
                func.lower(Image.name).like(search_term),
                func.lower(Project.name).like(search_term),
                func.lower(Upload.metadata_obj.op('->>')('project_name')).like(search_term)
            ]
            query = query.where(or_(*search_conditions))

        # Project name filter
        if project_name:
            query = query.where(
                or_(
                    Project.name.ilike(f"%{project_name}%"),
                    Upload.metadata_obj.op('->>')('project_name').ilike(f"%{project_name}%")
                )
            )

        # Processing status filter
        if status:
            query = query.where(Image.processing_status == status)

        # File type filter
        if file_type:
            query = query.where(Image.file_type.ilike(f"%{file_type}%"))

        # Aspect ratio filter
        if aspect_ratio:
            query = query.where(
                Image.exif_data.op('->>')('target_aspect_ratio') == aspect_ratio
            )

        # Crop mode filter
        if crop_mode:
            query = query.where(
                Image.exif_data.op('->>')('crop_mode') == crop_mode
            )

        # Operations filter
        if operations:
            op_list = [op.strip() for op in operations.split(',')]
            for operation in op_list:
                query = query.where(
                    Image.applied_steps.op('@>')([operation])
                )

        # Has processed output filter
        if has_output is not None:
            if has_output:
                query = query.where(Image.processed_url.isnot(None))
            else:
                query = query.where(Image.processed_url.is_(None))

        # Date range filters
        if date_from:
            try:
                from_date = datetime.strptime(date_from, "%Y-%m-%d")
                query = query.where(Image.created_at >= from_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format (use YYYY-MM-DD)")

        if date_to:
            try:
                to_date = datetime.strptime(date_to, "%Y-%m-%d")
                query = query.where(Image.created_at <= to_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format (use YYYY-MM-DD)")

        # Sorting
        if sort_by == "oldest":
            query = query.order_by(Image.created_at.asc())
        elif sort_by == "name":
            query = query.order_by(Image.name.asc())
        elif sort_by == "processing_time":
            query = query.order_by(Image.processing_time_ms.desc().nulls_last())
        else:  # newest (default)
            query = query.order_by(Image.created_at.desc())

        # Count total results (before pagination)
        count_query = select(func.count()).select_from(query.subquery())
        count_result = await db.execute(count_query)
        total = count_result.scalar()

        # Apply pagination
        query = query.offset(offset).limit(limit)

        # Execute main query
        result = await db.execute(query)
        images = result.mappings().all()

        # Format response
        formatted_results = []
        for img in images:
            # Extract project name from various sources
            project_name = (
                img.project_name or 
                (img.upload_metadata or {}).get('project_name') or 
                'Untitled Project'
            )
            
            # Extract original dimensions
            exif_data = img.exif_data or {}
            original_dims = exif_data.get('original_dimensions', {})
            dimensions = None
            if img.width and img.height:
                dimensions = f"{img.width}×{img.height}"
            elif original_dims and 'width' in original_dims and 'height' in original_dims:
                dimensions = f"{original_dims['width']}×{original_dims['height']}"

            formatted_results.append({
                "id": str(img.id),
                "name": img.name,
                "url": img.url,
                "thumbnail_url": img.thumbnail_url or img.url,
                "processed_url": img.processed_url,
                "dimensions": dimensions,
                "status": img.processing_status,
                "file_type": img.file_type,
                "created_at": img.created_at.isoformat() if img.created_at else None,
                "processing_time_ms": img.processing_time_ms,
                "operations": img.applied_steps or [],
                "project_name": project_name,
                "crop_mode": exif_data.get('crop_mode'),
                "aspect_ratio": exif_data.get('target_aspect_ratio'),
                "has_processed_output": bool(img.processed_url)
            })

        return {
            "results": formatted_results,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_next": offset + limit < total,
                "pages": (total + limit - 1) // limit,
                "current_page": (offset // limit) + 1
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed")

@router.get("/filters")
async def get_search_filters(
    user_id: Optional[str] = Query(None),
    all_users: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """
    Get available filter options based on actual data
    """
    try:
        is_admin = current_user.role == "admin"
        target_user_id = None
        
        if all_users and is_admin:
            target_user_id = None
        elif user_id and is_admin:
            target_user_id = user_id
        else:
            target_user_id = current_user.id

        # Base filter for user
        user_filter = []
        if target_user_id:
            user_filter.append(Image.user_id == target_user_id)

        # Get unique processing statuses
        status_query = select(
            Image.processing_status, 
            func.count(Image.id).label('count')
        ).group_by(Image.processing_status)
        
        if user_filter:
            status_query = status_query.where(and_(*user_filter))
        
        status_result = await db.execute(status_query)
        statuses = [
            {
                "value": row.processing_status, 
                "label": row.processing_status.replace('_', ' ').title(), 
                "count": row.count
            } 
            for row in status_result.all() if row.processing_status
        ]

        # Get unique file types
        filetype_query = select(
            Image.file_type,
            func.count(Image.id).label('count')
        ).group_by(Image.file_type)
        
        if user_filter:
            filetype_query = filetype_query.where(and_(*user_filter))
        
        filetype_result = await db.execute(filetype_query)
        file_types = [
            {
                "value": row.file_type,
                "label": row.file_type.replace('image/', '').upper() if row.file_type else 'Unknown',
                "count": row.count
            }
            for row in filetype_result.all() if row.file_type
        ]

        # Get unique aspect ratios
        aspect_query = text("""
            SELECT 
                exif_data->>'target_aspect_ratio' as aspect_ratio,
                COUNT(*) as count
            FROM images 
            WHERE exif_data->>'target_aspect_ratio' IS NOT NULL
            """ + (f"AND user_id = '{target_user_id}'" if target_user_id else "") + """
            GROUP BY exif_data->>'target_aspect_ratio'
            ORDER BY count DESC
        """)
        
        aspect_result = await db.execute(aspect_query)
        aspect_ratios = [
            {
                "value": row.aspect_ratio,
                "label": row.aspect_ratio,
                "count": row.count
            }
            for row in aspect_result.all()
        ]

        # Get unique operations from applied_steps
        ops_query = text("""
            SELECT 
                operation,
                COUNT(*) as count
            FROM images, jsonb_array_elements_text(applied_steps) as operation
            WHERE jsonb_array_length(applied_steps) > 0
            """ + (f"AND user_id = '{target_user_id}'" if target_user_id else "") + """
            GROUP BY operation
            ORDER BY count DESC
        """)
        
        ops_result = await db.execute(ops_query)
        operations = [
            {
                "value": row.operation,
                "label": row.operation.replace('_', ' ').title(),
                "count": row.count
            }
            for row in ops_result.all()
        ]

        # Get project names
        project_query = text("""
            SELECT DISTINCT 
                COALESCE(p.name, u.metadata->>'project_name') as project_name,
                COUNT(i.id) as count
            FROM images i
            JOIN uploads u ON i.upload_id = u.id
            LEFT JOIN projects p ON u.project_id = p.id
            WHERE COALESCE(p.name, u.metadata->>'project_name') IS NOT NULL
            """ + (f"AND i.user_id = '{target_user_id}'" if target_user_id else "") + """
            GROUP BY COALESCE(p.name, u.metadata->>'project_name')
            ORDER BY count DESC
            LIMIT 20
        """)
        
        project_result = await db.execute(project_query)
        projects = [
            {
                "value": row.project_name,
                "label": row.project_name,
                "count": row.count
            }
            for row in project_result.all()
        ]

        return {
            "statuses": statuses,
            "file_types": file_types,
            "aspect_ratios": aspect_ratios,
            "operations": operations,
            "projects": projects,
            "crop_modes": [
                {"value": "preset", "label": "Preset"},
                {"value": "manual", "label": "Manual"}
            ]
        }

    except Exception as e:
        logger.error(f"Failed to get search filters: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get search filters")
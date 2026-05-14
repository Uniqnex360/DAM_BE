from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from app.api import deps
from app.db.session import get_db
from app.models.auth import User
from app.models.project import Project
from app.models.assets import Upload, Image
from app.schemas.project import ProjectCreate, ProjectListResponse, ProjectResponse
router = APIRouter()


@router.post("/", response_model=ProjectResponse, status_code=201)
async def create_project(
    project: ProjectCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
) -> dict:
    if not project.name.strip():
        raise HTTPException(
            status_code=400, detail="Project name cannot be empty")
    try:
        new_project = Project(
            user_id=current_user.id,
            name=project.name.strip(),
            description=project.description,
            destination_count=project.destinations
        )
        db.add(new_project)
        await db.commit()
        await db.refresh(new_project)
        return {
            "id": str(new_project.id),
            "name": new_project.name,
            "description": new_project.description,
            "status": new_project.status,
            "destination_count": new_project.destination_count,
            "image_count": 0,
            "created_at": new_project.created_at
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to create project: {str(e)}")


@router.get("/", response_model=ProjectListResponse)
async def list_projects(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
) -> dict:
    try:
        base_query = select(Project).where(Project.user_id == current_user.id)
        if status:
            base_query = base_query.where(Project.status == status)
        count_query = select(func.count()).select_from(base_query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        offset = (page - 1) * limit
        result = await db.execute(
            base_query.order_by(Project.created_at.desc()
                                ).offset(offset).limit(limit)
        )
        projects = result.scalars().all()
        project_list = []
        for p in projects:
            img_count = await db.execute(
                select(func.count(Image.id))
                .join(Upload, Image.upload_id == Upload.id)
                .where(Upload.project_id == p.id)
            )
            count = img_count.scalar() or 0
            project_list.append({
                "id": str(p.id),
                "name": p.name,
                "description": p.description,
                "status": p.status,
                "destination_count": p.destination_count,
                "image_count": count,
                "created_at": p.created_at
            })
        return {
            "total": total,
            "page": page,
            "limit": limit,
            "projects": project_list
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch projects: {str(e)}")


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
) -> dict:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if str(project.user_id) != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized")
    img_count = await db.execute(
        select(func.count(Image.id))
        .join(Upload, Image.upload_id == Upload.id)
        .where(Upload.project_id == project.id)
    )
    count = img_count.scalar() or 0
    return {
        "id": str(project.id),
        "name": project.name,
        "description": project.description,
        "status": project.status,
        "destination_count": project.destination_count,
        "image_count": count,
        "created_at": project.created_at
    }


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
) -> dict:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if str(project.user_id) != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized")
    try:
        await db.execute(
            Upload.__table__.update()
            .where(Upload.project_id == project.id)
            .values(project_id=None)
        )
        await db.delete(project)
        await db.commit()
        return {"status": "deleted", "id": project_id}
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to delete project: {str(e)}")

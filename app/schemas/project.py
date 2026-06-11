
from pydantic import BaseModel
from uuid import UUID
from typing import Optional
from datetime import datetime

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    destinations: Optional[str] = None

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    status: str
    destination_count: Optional[str] = None
    image_count: int
    created_at: datetime

class ProjectListResponse(BaseModel):
    total: int
    page: int
    limit: int
    projects: list[ProjectResponse]
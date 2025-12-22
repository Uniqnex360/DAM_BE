from pydantic import BaseModel
from uuid import UUID
from typing import Optional
from datetime import datetime

class ImageResponse(BaseModel):
    id: UUID
    url: str
    thumbnail_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    processing_status: str
    created_at: datetime

    class Config:
        from_attributes = True
from pydantic import BaseModel
from uuid import UUID
from typing import Optional,List
from datetime import datetime

class ImageResponse(BaseModel):
    id: UUID
    url: str
    name: str
    thumbnail_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    processing_status: str
    created_at: datetime

class BatchImageItem(BaseModel):
    id: str
    name: str
    url: str
    width: Optional[int] = None
    height: Optional[int] = None
class FailedFile(BaseModel):
    filename: str
    error: str

class BatchUploadResponse(BaseModel):
    upload_id: str
    images: list[BatchImageItem]
    failed_uploads: List[FailedFile] = []
    status: str
from sqlalchemy import Column, String, ForeignKey, Numeric, Integer, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import text
from .base import Base
from sqlalchemy import Column, DateTime,func

class Job(Base):
    __tablename__ = "jobs"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    type = Column(String, nullable=False) # stain, segment, 3d
    status = Column(String, server_default="pending")
    
    input_data = Column(JSONB, nullable=False)
    output_data = Column(JSONB, server_default=text("'{}'::jsonb"))
    
    error_message = Column(Text)
    cost_estimate = Column(Numeric)
    cost_actual = Column(Numeric)
    webhook_url = Column(String)
    
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

class ProcessingStatistic(Base):
    __tablename__ = "processing_statistics"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"))
    date = Column(DateTime(timezone=True), server_default=func.now())
    total_images_uploaded = Column(Integer, default=0)
    total_images_processed = Column(Integer, default=0)
    total_processing_time_ms = Column(Integer, default=0)
    operation_counts = Column(JSONB, server_default=text("'{}'::jsonb"))
    # operation_counts example: {"bg_removal": 45, "smart_crop": 32, "resize": 28}
    daily_breakdown = Column(JSONB, server_default=text("'{}'::jsonb"))
    # daily_breakdown example: {"2026-05-12": {"bg_removal": 5, "total": 10}}
from sqlalchemy import Column, String, ForeignKey, Text,Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy import text
from .base import Base

class Project(Base):
    __tablename__ = "projects"
    
    name = Column(String, nullable=False, index=True)
    description = Column(Text)
    status = Column(String, server_default="active") 
    settings = Column(JSONB, server_default=text("'{}'::jsonb"))
    thumbnail_url = Column(String)
    destination_count = Column(String)  
    image_count = Column(Integer, default=0)
    
    # Relationships
    uploads = relationship("Upload", back_populates="project")
from typing import Any 
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.dialects.postgresql import UUID,JSONB
from sqlalchemy import Column,DateTime,func,text 
import uuid
class Base(DeclarativeBase):
    id = Column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4,nullable=False,)
    created_at=Column(DateTime(timezone=True),server_default=func.now())
    updated_at=Column(DateTime(timezone=True),onupdate=func.now(),server_default=func.now())
    type_annotation_map={
        dict:JSONB
    }
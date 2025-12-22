from sqlalchemy import Column, String, Boolean, Text,Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import text
from .base import Base

class StainLibrary(Base):
    __tablename__ = "stain_library"
    
    name = Column(String, nullable=False)
    color_hex = Column(String, nullable=False)
    color_lab = Column(JSONB)
    category = Column(String)
    preview_url = Column(String)
    texture_sample_url = Column(String)
    is_public = Column(Boolean, default=True)

class AIPrompt(Base):
    __tablename__ = "ai_prompts"
    
    name = Column(String, nullable=False)
    prompt_text = Column(Text, nullable=False)
    prompt_type = Column(String)
    parameters = Column(JSONB, server_default=text("'{}'::jsonb"))
    usage_count = Column(Integer, default=0)
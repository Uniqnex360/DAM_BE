from sqlalchemy import Column, String, ForeignKey, Integer,Boolean, Text, Numeric
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import text
from .base import Base
class Upload(Base):
    __tablename__='uploads'
    user_id=Column(UUID(as_uuid=True),ForeignKey('users.id'))
    product_id=Column(UUID(as_uuid=True),ForeignKey('products.id'))
    status=Column(String,server_default='uploaded')
    metadata_obj=Column('metadata',JSONB,server_default=text("'{}'::jsonb"))

class Image(Base):
    __tablename__='images'
    upload_id=Column(UUID(as_uuid=True),ForeignKey('uploads.id'))
    user_id=Column(UUID(as_uuid=True),ForeignKey('users.id'))
    confidence_scores = Column(JSONB, server_default=text("'{}'::jsonb")) 
    applied_steps = Column(JSONB, server_default=text("'[]'::jsonb"))
    processing_time_ms = Column(Integer)
    url=Column(String,nullable=False)
    thumbnail_url=Column(String)
    name = Column(String, nullable=True) 
    file_type = Column(String, nullable=True)
    width=Column(Integer)
    height=Column(Integer)
    masks=Column(JSONB, server_default=text("'{}'::jsonb"))
    exif_data=Column(JSONB,server_default=text("'{}'::jsonb"))
    embeddings_id=Column(String)
    category_id=Column(UUID(as_uuid=True),ForeignKey('categories.id'))
    brand_id=Column(UUID(as_uuid=True),ForeignKey('brands.id'))
    processed_url=Column(String)
    processing_status=Column(String,server_default='pending')

class Model3D(Base):
    __tablename__='models_3d'
    user_id=Column(UUID(as_uuid=True),ForeignKey('users.id'))
    product_id=Column(UUID(as_uuid=True),ForeignKey('products.id'))
    name=Column(String,nullable=False)
    glb_url = Column(String)
    usdz_url = Column(String)
    fbx_url = Column(String)
    gltf_url = Column(String)
    generation_method = Column(String)
    quality_score = Column(Numeric)
    polygon_count = Column(Integer)
    component_mapping = Column(JSONB, server_default=text("'{}'::jsonb"))
class ARAsset(Base):
    __tablename__ = "ar_assets"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    model_id = Column(UUID(as_uuid=True), ForeignKey("models_3d.id"))
    usdz_url = Column(String)
    glb_url = Column(String)
    ios_compatible = Column(Boolean, default=False)
    android_compatible = Column(Boolean, default=False)

class Texture(Base):
    __tablename__ = "textures"
    
    name = Column(String, nullable=False)
    albedo_url = Column(String, nullable=False)
    normal_url = Column(String)
    roughness_url = Column(String)
    ao_url = Column(String)
    preview_url = Column(String)
    
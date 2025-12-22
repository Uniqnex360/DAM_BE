from sqlalchemy import Column, String, Boolean, ForeignKey, Integer, Text, text, Numeric
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from .base import Base 
from .mixins import (
    DescribableMixin, 
    MetadataMixin, 
    ActiveMixin, 
    ClientOwnedMixin, 
    UserOwnedMixin,
    ProductRelatedMixin
)

class Category(Base, ClientOwnedMixin, DescribableMixin, MetadataMixin, ActiveMixin):
    __tablename__ = "categories"

    slug = Column(String, nullable=False, index=True)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("categories.id"))
    thumbnail_url = Column(String)
    display_order = Column(Integer, default=0)

class Brand(Base, ClientOwnedMixin, DescribableMixin, MetadataMixin, ActiveMixin):
    __tablename__ = 'brands'
    
    slug = Column(String, nullable=False, index=True)
    logo_url = Column(String)
    website_url = Column(String)
    
class Product(Base, UserOwnedMixin, DescribableMixin, MetadataMixin):
    __tablename__ = 'products'
    
    # FIX 1: Removed primary_key=True
    # FIX 2: Renamed to category_id for standard naming (optional but recommended)
    category_id = Column(UUID(as_uuid=True), ForeignKey('categories.id'), nullable=True)
    
    base_price = Column(Numeric(10,2))
    sku = Column(String, index=True)
    is_public = Column(Boolean, default=False)
    
class ProductComponent(Base, ProductRelatedMixin, DescribableMixin, MetadataMixin):
    __tablename__ = 'product_components'
    
    type = Column(String, nullable=False)
    material = Column(String)
    is_default = Column(Boolean, default=False)
    price_modifier = Column(Numeric(10,2), default=0)
    mesh_url = Column(String)
    thumbnail_url = Column(String)
    
class Configuration(Base, UserOwnedMixin, ProductRelatedMixin):
    __tablename__ = 'configurations'
    
    name = Column(String)
    components = Column(JSONB, server_default=text("'{}'::jsonb"))
    custom_options = Column(JSONB, server_default=text("'{}'::jsonb"))
    total_price = Column(Numeric(10,2))
    preview_url = Column(String)
    is_saved = Column(Boolean, default=False)
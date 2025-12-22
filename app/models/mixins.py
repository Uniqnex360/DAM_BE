from sqlalchemy import Column, String, Boolean, ForeignKey, Text, Integer
from sqlalchemy.orm import declared_attr
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import text

class DescribableMixin:
    name = Column(String, nullable=False)
    description = Column(Text)

class MetadataMixin:
    metadata_obj = Column("metadata", JSONB, server_default=text("'{}'::jsonb"))

class ActiveMixin:
    is_active = Column(Boolean, default=True)

class ClientOwnedMixin:
    @declared_attr
    def client_id(cls):
        return Column(UUID(as_uuid=True), ForeignKey("clients.id"), index=True)

class UserOwnedMixin:
    @declared_attr
    def user_id(cls):
        return Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)

class ProductRelatedMixin:
    @declared_attr
    def product_id(cls):
        return Column(UUID(as_uuid=True), ForeignKey("products.id"), index=True)
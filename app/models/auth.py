from sqlalchemy import Column, String, Boolean, text
from sqlalchemy.dialects.postgresql import UUID
from .base import Base


class User(Base):
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True,
                server_default=text("gen_random_uuid()"))
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)

    # role can be: 'admin' or 'user'
    role = Column(String, nullable=False, default='user', index=True)

from sqlalchemy import Column,String,Boolean,ForeignKey,Integer,text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID,JSONB
from .base import Base 
class User(Base):
    __tablename__='users'
    email=Column(String,unique=True,index=True,nullable=False)
    hashed_password=Column(String,nullable=False)
    is_active=Column(Boolean,default=True)
    full_name = Column(String, nullable=True) 
    profile=relationship("Profile",back_populates='user',uselist=False)

class Profile(Base):
    __tablename__='profiles'
    id=Column(UUID(as_uuid=True),ForeignKey('users.id'),primary_key=True)
    email=Column(String,nullable=False)
    role=Column(String,nullable=False,default='client')
    company=Column(String)
    avatar_url=Column(String)
    quota_limit=Column(Integer,default=100)
    quota_used=Column(Integer,default=0)
    client_id=Column(UUID(as_uuid=True),ForeignKey('clients.id'))
    settings = Column(JSONB, server_default=text("'{}'::jsonb"))
    user=relationship("User",back_populates='profile')
    client=relationship("Client",back_populates='profiles')
    
class Client(Base):
    __tablename__='clients'
    name=Column(String,nullable=False)
    company_code=Column(String,nullable=False)
    cloudinary_folder=Column(String)
    logo_url=Column(String)
    primary_color=Column(String,server_default='#3B82F6')
    secondary_color=Column(String,server_default='#1E40AF')
    business_rules = Column(JSONB, server_default=text("'{}'::jsonb"))
    api_credentials=Column(JSONB, server_default=text("'{}'::jsonb"))
    settings=Column(JSONB, server_default=text("'{}'::jsonb"))
    is_active=Column(Boolean,default=True)
    profiles=relationship('Profile',back_populates='client')
    
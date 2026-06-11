from typing import Optional
from pydantic import BaseModel, EmailStr, Field
from uuid import UUID
class UserBase(BaseModel):
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = True
    full_name: Optional[str] = None
    role: str = "user"  
class UserCreate(UserBase):
    email: EmailStr
    password: str = Field(..., min_length=8)
class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)
    role: Optional[str] = None
    is_active: Optional[bool] = None
class User(UserBase):
    id: UUID
    class Config:
        from_attributes = True

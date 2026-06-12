from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator
from uuid import UUID


class EmailOptionalModel(BaseModel):
    @field_validator('email', mode='before', check_fields=False)
    @classmethod
    def empty_str_to_none(cls, v):
        return None if v == "" else v


class UserBase(EmailOptionalModel):
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = True
    full_name: Optional[str] = None
    role: str = "user"  
class UserCreate(UserBase):
    password: str = Field(..., min_length=1)


class UserUpdate(EmailOptionalModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=1)
    role: Optional[str] = None
    is_active: Optional[bool] = None
class User(UserBase):
    id: UUID
    class Config:
        from_attributes = True

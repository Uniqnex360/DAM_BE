from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.api import deps
import logging
from app.core import security
from app.core.config import settings
from app.db.session import get_db
from app.models.auth import User
from app.schemas import user as user_schema
from app.schemas import token as token_schema
from uuid import UUID
logger=logging.getLogger("auth.py")

router = APIRouter()

@router.post("/login", response_model=token_schema.Token)
async def login_access_token(
    db: AsyncSession = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalars().first()

    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
        
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return {
        "access_token": security.create_access_token(
            user.id, expires_delta=access_token_expires
        ),
        "token_type": "bearer",
    }


@router.post("/register", response_model=user_schema.User)
async def register_new_user(
    *,
    db: AsyncSession = Depends(get_db),
    user_in: user_schema.UserCreate,
) -> Any:
    try:
        
        email = user_in.email if user_in.email and user_in.email.strip() else None

        
        if email:
            result = await db.execute(select(User).where(User.email == email))
            if result.scalars().first():
                raise HTTPException(
                    status_code=400,
                    detail="The user with this email already exists in the system",
                )

        user = User(
            email=email,  
            hashed_password=security.get_password_hash(user_in.password),
            full_name=user_in.full_name,
            is_active=True
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

        logger.info(
            f"User registered successfully: {user_in.full_name} (email: {email or 'No email'})")
        return user

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            f"Registration failed for user {user_in.full_name}: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to register user. Please try again later."
        )
@router.get('/verify',response_model=dict)
async def verify_token(current_user: User = Depends(deps.get_current_user),) -> Any:
    return {
        "valid": True,
        "user": current_user.email,
        "id": current_user.id,
        "role": current_user.role
    }

@router.post('/impersonate/{user_id}')
async def impersonate_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(deps.PermissionChecker(["admin"]))
) -> token_schema.Token:
    try:
        result=await db.execute(select(User).where(User.id==user_id))
        target_user=result.scalars().first()
        if not target_user:
            raise HTTPException(status_code=404,detail='User not found')
        if not target_user.is_active:
            raise HTTPException(status_code=400,detail='Cannot impersonate inactive user')
        logger.info(f"Admin {current_admin.email} impersonating {target_user.email}")
        access_token_expires=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token=security.create_access_token(target_user.id,expires_delta=access_token_expires)
        return {
            'access_token':access_token,
            'token_type':'bearer',
            'impersonated_user':{
                "id":str(target_user.id),
                "email":target_user.email,
                'full_name':target_user.full_name,
                'role':target_user.role
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            f"Impersonation failed for user_id={user_id}: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to impersonate user"
        )


@router.post("/stop-impersonation")
async def stop_impersonation(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    try:
        return {
            "message": "Impersonation stopped. Please log in again."
        }

    except Exception as e:
        logger.exception(f"Stop impersonation failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Failed to stop impersonation"
        )

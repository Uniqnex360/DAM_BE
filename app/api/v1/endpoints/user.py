from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.exc import SQLAlchemyError
from app.db.session import get_db
from app.models.auth import User as UserModel
from app.schemas import user as user_schema
from app.core import security
from app.api import deps
from uuid import UUID
from typing import List
import logging
logger = logging.getLogger('user')
router = APIRouter()
@router.get("/", response_model=List[user_schema.User])
async def read_users(
    db: AsyncSession = Depends(get_db),
    current_admin: UserModel = Depends(deps.PermissionChecker(["admin"]))
):
    try:
        result = await db.execute(select(UserModel).order_by(UserModel.email))
        return result.scalars().all()
    except Exception as e:
        logger.error(f"Error fetching users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving users from database"
        )
@router.post("/", response_model=user_schema.User)
async def create_user(
    user_in: user_schema.UserCreate,
    db: AsyncSession = Depends(get_db),
    current_admin: UserModel = Depends(deps.PermissionChecker(["admin"]))
):
    try:
        query = await db.execute(select(UserModel).where(UserModel.email == user_in.email))
        if query.scalars().first():
            raise HTTPException(
                status_code=400, detail="User with this email already exists")
        new_user = UserModel(
            email=user_in.email,
            hashed_password=security.get_password_hash(user_in.password),
            full_name=user_in.full_name,
            role=user_in.role,
            is_active=user_in.is_active
        )
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        return new_user
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Database error during user creation: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to create user due to database error")
    except Exception as e:
        await db.rollback()
        logger.error(f"Unexpected error during user creation: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred")
@router.patch("/{user_id}", response_model=user_schema.User)
async def update_user(
    user_id: UUID,
    user_in: user_schema.UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_admin: UserModel = Depends(deps.PermissionChecker(["admin"]))
):
    try:
        result = await db.execute(select(UserModel).where(UserModel.id == user_id))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user_id == current_admin.id:
            if user_in.is_active is False or (user_in.role and user_in.role != "admin"):
                raise HTTPException(
                    status_code=400,
                    detail="Security risk: Cannot deactivate or demote your own admin account"
                )
        update_data = user_in.model_dump(exclude_unset=True)
        if "password" in update_data:
            update_data["hashed_password"] = security.get_password_hash(
                update_data.pop("password"))
        for field, value in update_data.items():
            setattr(user, field, value)
        await db.commit()
        await db.refresh(user)
        return user
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Database error during user update {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update user")
    except Exception as e:
        await db.rollback()
        logger.error(
            f"Unexpected error during user update {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred")
@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_admin: UserModel = Depends(deps.PermissionChecker(["admin"]))
):
    try:
        if user_id == current_admin.id:
            raise HTTPException(
                status_code=400, detail="Cannot delete your own admin account")
        result = await db.execute(select(UserModel).where(UserModel.id == user_id))
        if not result.scalars().first():
            raise HTTPException(status_code=404, detail="User not found")
        await db.execute(delete(UserModel).where(UserModel.id == user_id))
        await db.commit()
        return None
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(
            f"Database error during user deletion {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete user")
    except Exception as e:
        await db.rollback()
        logger.error(
            f"Unexpected error during user deletion {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred")

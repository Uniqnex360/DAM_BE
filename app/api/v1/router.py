from fastapi import APIRouter
from app.api.v1.endpoints import auth
from app.api.v1.endpoints import assets
from app.api.v1.endpoints import dashboard
api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(assets.router, prefix="/assets", tags=["assets"]) 
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["analytics"])


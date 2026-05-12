from fastapi import APIRouter
from app.api.v1.endpoints import auth
from app.api.v1.endpoints import assets
from app.api.v1.endpoints import dashboard
from app.api.v1.endpoints import reports

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(assets.router, prefix="/assets", tags=["assets"]) 
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])  
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["analytics"])


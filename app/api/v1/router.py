from fastapi import APIRouter
from app.api.v1.endpoints import auth
from app.api.v1.endpoints import assets
from app.api.v1.endpoints import dashboard
from app.api.v1.endpoints import reports
from app.api.v1.endpoints import user
from app.api.v1.endpoints import room_visualizer
from app.api.v1.endpoints import search






api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(assets.router, prefix="/assets", tags=["assets"]) 
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])  
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(user.router, prefix="/users", tags=["users"])
api_router.include_router(
    room_visualizer.router,
    prefix="/room-visualizer",
    tags=["room-visualizer"]
)
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["analytics"])


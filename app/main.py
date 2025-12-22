from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.router import api_router 
from fastapi.staticfiles import StaticFiles
import os
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)
if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")
for dir_name in ["static/uploads", "static/processed"]:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(api_router, prefix=settings.API_V1_STR) 

@app.get("/health")
def health_check():
    return {"status": "ok"}
# import os
# import uvicorn
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from app.core.config import settings

# import huggingface_hub
# if not hasattr(huggingface_hub, "cached_download"):
#     huggingface_hub.cached_download = huggingface_hub.hf_hub_download

# from app.api.v1.router import api_router
# from fastapi.staticfiles import StaticFiles
# app = FastAPI(
#     title=settings.PROJECT_NAME,
#     openapi_url=f"{settings.API_V1_STR}/openapi.json"
# )

# if not os.path.exists("static/uploads"):
#     os.makedirs("static/uploads")
# for dir_name in ["static/uploads", "static/processed"]:
#     if not os.path.exists(dir_name):
#         os.makedirs(dir_name)

# if settings.BACKEND_CORS_ORIGINS:
#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
#         allow_credentials=True,
#         allow_methods=["*"],
#         allow_headers=["*"],
#         expose_headers=["Content-Disposition"],
#     )
# app.mount("/static", StaticFiles(directory="app/static"), name="static")
# @app.get("/")
# @app.head("/")
# def root():
#     return {
#         "status": "ok",
#         "message": "Digital Assets Management API",
#         "docs": "/docs",
#         "health": "/health"
#     }

# app.include_router(api_router, prefix=settings.API_V1_STR) 

# @app.get("/health")
# def health_check():
#     return {"status": "ok"}

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
import os
import logging
import uvicorn
import asyncio
from contextlib import asynccontextmanager

import huggingface_hub
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core.config import settings

# ── Logging config: fixes the per-module WARNING bug ───────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services.image_processing.model_registry import (
        get_wm_detector, get_lama, get_all_segmenters, get_florence,
    )
    logger.info("Preloading heavy models...")
    await asyncio.gather(
        asyncio.to_thread(get_wm_detector),
        asyncio.to_thread(get_lama),
        asyncio.to_thread(get_all_segmenters),
        asyncio.to_thread(get_florence),
    )
    logger.info("Models ready.")
    yield
    
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
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
        expose_headers=["Content-Disposition"],
    )

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
@app.head("/")
def root():
    return {
        "status": "ok",
        "message": "Digital Assets Management API",
        "docs": "/docs",
        "health": "/health",
    }


app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
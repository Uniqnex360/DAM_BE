# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from app.core.config import settings
# from app.api.v1.router import api_router 
# from fastapi.staticfiles import StaticFiles
# import os
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
#     )
# app.mount("/static", StaticFiles(directory="static"), name="static")
# app.include_router(api_router, prefix=settings.API_V1_STR) 

# @app.get("/health")
# def health_check():
#     return {"status": "ok"}
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.router import api_router 
from fastapi.staticfiles import StaticFiles

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

# Add this at the end
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
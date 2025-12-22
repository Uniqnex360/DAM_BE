import cloudinary
import cloudinary.uploader
import os
import shutil
from pathlib import Path
from app.core.config import settings

# Configure Cloudinary globally
if settings.CLOUDINARY_CLOUD_NAME:
    cloudinary.config(
        cloud_name=settings.CLOUDINARY_CLOUD_NAME,
        api_key=settings.CLOUDINARY_API_KEY,
        api_secret=settings.CLOUDINARY_API_SECRET
    )

def upload_image_to_cloudinary(file_bytes: bytes, filename: str) -> dict:
    """
    Uploads file to Cloudinary OR Local Disk based on settings.
    filename example: "processed/user_id/image_id.jpg"
    """
    
    # ==========================
    # OPTION A: LOCAL STORAGE
    # ==========================
    if settings.STORAGE_PROVIDER == "local":
        # 1. Clean up filename to prevent directory traversal attacks
        # But allow 1 level of subfolder (e.g. processed/image.jpg)
        
        # Determine target folder
        if filename.startswith("processed/"):
            target_dir = "static/processed"
            # Extract just the filename part
            safe_name = filename.replace("/", "_")
        else:
            target_dir = "static/uploads"
            safe_name = filename.replace("/", "_")

        # Create directory if missing
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        file_path = f"{target_dir}/{safe_name}"
        
        # 2. Write to Disk
        with open(file_path, "wb") as f:
            f.write(file_bytes)
            
        # 3. Return URL
        return {
            "secure_url": f"http://localhost:8000/{file_path}",
            "url": f"http://localhost:8000/{file_path}",
            "width": 0,
            "height": 0,
            "public_id": safe_name
        }

    # ==========================
    # OPTION B: CLOUDINARY
    # ==========================
    try:
        upload_result = cloudinary.uploader.upload(
            file_bytes,
            public_id=filename,
            upload_preset=settings.CLOUDINARY_UPLOAD_PRESET, 
            resource_type="auto"
        )
        return upload_result
    except Exception as e:
        print(f"Cloudinary Error: {e}")
        raise e
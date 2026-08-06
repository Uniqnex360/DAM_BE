import logging
import cloudinary
import cloudinary.uploader

logger = logging.getLogger(__name__)


def extract_cloudinary_public_id(url: str) -> str:
    try:
        if not url or "cloudinary" not in url:
            return None
        parts = url.split("upload/")
        if len(parts) < 2:
            return None
        after_upload = parts[1]
        version_parts = after_upload.split("/", 1)
        if len(version_parts) < 2:
            return None
        public_id_with_ext = version_parts[1]
        public_id = public_id_with_ext.rsplit(".", 1)[0]
        return public_id
    except Exception as e:
        logger.error(f"Failed to extract public_id from URL {url}: {e}")
        return None


async def delete_from_cloudinary(public_id: str) -> bool:
    try:
        result = cloudinary.uploader.destroy(public_id)
        if result.get("result") == "ok":
            logger.info(f"Successfully deleted from Cloudinary: {public_id}")
            return True
        else:
            logger.warning(f"Cloudinary deletion returned: {result.get('result')} for {public_id}")
            return False
    except Exception as e:
        logger.error(f"Cloudinary deletion error for {public_id}: {e}")
        raise
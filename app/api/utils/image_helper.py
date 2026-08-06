import logging
from app.services.cloudinary_service import extract_cloudinary_public_id, delete_from_cloudinary

logger = logging.getLogger(__name__)


def collect_image_urls(image) -> list[str]:
    urls = []
    if image.url:
        urls.append(image.url)
    if image.processed_url and image.processed_url != image.url:
        urls.append(image.processed_url)
    if image.thumbnail_url and image.thumbnail_url not in urls:
        urls.append(image.thumbnail_url)
    return urls


async def delete_urls_from_cloudinary(urls: list[str]) -> list[str]:
    errors = []
    for url in urls:
        try:
            public_id = extract_cloudinary_public_id(url)
            if public_id:
                await delete_from_cloudinary(public_id)
                logger.info(f"Deleted from Cloudinary: {public_id}")
        except Exception as e:
            logger.error(f"Cloudinary deletion failed: {url} - {e}")
            errors.append(str(e))
    return errors
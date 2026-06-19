from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models.assets import Image, Upload


class ImageRepository:
    def __init__(self, db: AsyncSession):
        self._db = db

    async def get_image(self, image_id: str) -> Image | None:
        result = await self._db.execute(select(Image).where(Image.id == image_id))
        return result.scalars().first()

    async def get_upload(self, upload_id: str) -> Upload | None:
        result = await self._db.execute(select(Upload).where(Upload.id == upload_id))
        return result.scalars().first()

    async def start_processing(self, image: Image, upload: Upload | None):
        image.processing_status = "processing"
        if upload and upload.status != "processing":
            upload.status = "processing"
        await self._db.commit()

    async def complete_image(
        self,
        image: Image,
        processed_url: str | None,
        confidence: dict,
        steps: list,
        duration: int,
    ):
        image.processed_url = processed_url
        image.processing_status = "completed"
        image.confidence_scores = confidence
        image.applied_steps = steps
        image.processing_time_ms = duration
        await self._db.commit()

    async def fail_image(self, image: Image):
        image.processing_status = "failed"
        await self._db.commit()

    async def unfinished_count(self, upload_id: str) -> int:
        result = await self._db.execute(
            select(func.count(Image.id))
            .where(Image.upload_id == upload_id)
            .where(Image.processing_status.in_(["pending", "processing"]))
        )
        return result.scalar()

    async def complete_upload(self, upload: Upload):
        upload.status = "completed"
        await self._db.commit()

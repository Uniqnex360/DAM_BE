import asyncio
import logging
from typing import Optional

import cv2
import numpy as np

from app.db.session import AsyncSessionLocal
from app.services.image_fetcher import ImageFetcher
from app.services.image_processing import ImageProcessor
from app.services.media import upload_image_to_cloudinary
from app.services.repositories import ImageRepository
from app.services.statistics import update_processing_stats

logger = logging.getLogger("assets")


class ProcessImageUseCase:
    def __init__(self, repo: ImageRepository, fetcher: ImageFetcher):
        self._repo = repo
        self._fetcher = fetcher

    async def execute(
        self,
        image_id: str,
        target_user_id: str,
        operations: list,
        options: dict,
        autoDetect: bool,
    ) -> dict:
        img_record = await self._repo.get_image(image_id)
        if not img_record:
            raise ValueError("Image not found")

        upload = await self._repo.get_upload(img_record.upload_id)
        await self._repo.start_processing(img_record, upload)

        try:
            image_content = await self._fetcher.fetch(img_record.url)

            resize_dims = options.get("resize") or None
            background_color = options.get("background_color", "#FFFFFF")
            skip_crop = options.get("skip_crop", False)

            target_dimensions = crop_mode = target_aspect_ratio = None
            if img_record.exif_data:
                target_dimensions = img_record.exif_data.get(
                    "original_dimensions")
                crop_mode = img_record.exif_data.get("crop_mode")
                target_aspect_ratio = img_record.exif_data.get(
                    "target_aspect_ratio")

            processor = ImageProcessor(
                image_content,
                resize_dims=resize_dims,
                operations=operations,
                autoDetect=autoDetect,
                skip_crop=skip_crop,
                target_dimensions=target_dimensions,
                crop_mode=crop_mode,
                target_aspect_ratio=target_aspect_ratio,
                background_color=background_color,
            )

            proc_result = await asyncio.to_thread(processor.process)

            
            resize_results = proc_result.get("resize_results")
            if resize_results:
                outputs = self._build_multi_outputs(
                    resize_results, img_record.user_id, image_id
                )
                processed_url = outputs[0]["url"] if outputs else None
                response = {
                    "status": "completed",
                    "outputs": outputs,
                    "original_image_id": image_id,
                    "telemetry": {
                        "confidence": proc_result["confidence"],
                        "steps": proc_result["steps_applied"],
                        "time_ms": proc_result["duration_ms"],
                    },
                }
            else:
                filename = f"processed/{img_record.user_id}/{image_id}.jpg"
                upload_res = upload_image_to_cloudinary(
                    proc_result["image_bytes"], filename)
                processed_url = upload_res.get("secure_url")
                response = {
                    "status": "completed",
                    "url": processed_url,
                    "name": img_record.name,
                    "telemetry": {
                        "confidence": proc_result["confidence"],
                        "steps": proc_result["steps_applied"],
                        "time_ms": proc_result["duration_ms"],
                    },
                }

            await self._repo.complete_image(
                img_record,
                processed_url,
                proc_result["confidence"],
                proc_result["steps_applied"],
                proc_result["duration_ms"],
            )

            
            unfinished = await self._repo.unfinished_count(img_record.upload_id)
            if unfinished == 0 and upload:
                await self._repo.complete_upload(upload)

            
            await self._record_stats(
                target_user_id, proc_result["steps_applied"], proc_result["duration_ms"]
            )

            return response

        except Exception:
            await self._repo.fail_image(img_record)
            raise

    def _build_multi_outputs(self, resize_results, user_id, image_id):
        outputs = []
        for res in resize_results:
            img_data = res.get("image_bytes")
            if isinstance(img_data, np.ndarray):
                success, encoded = cv2.imencode(".jpg", img_data)
                img_data = encoded.tobytes()
            filename = f"processed/{user_id}/{image_id}_{res['id']}.jpg"
            upload_res = upload_image_to_cloudinary(img_data, filename)
            outputs.append({
                "marketplace": res["id"],
                "url": upload_res.get("secure_url"),
                "width": res["width"],
                "height": res["height"],
            })
        return outputs

    async def _record_stats(self, user_id: str, steps: list, duration: int):
        try:
            async with AsyncSessionLocal() as stats_db:
                if steps:
                    for step in steps:
                        await update_processing_stats(stats_db, user_id, step, duration)
                await stats_db.commit()
        except Exception as e:
            logger.error(f"Stats update failed: {e}")

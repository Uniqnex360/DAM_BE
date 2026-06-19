import os
import logging
from typing import Optional
from urllib.parse import urlparse
import httpx
from fastapi.concurrency import run_in_threadpool
logger = logging.getLogger(__name__)
_http_client: Optional[httpx.AsyncClient] = None
def _get_shared_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={"User-Agent": "DAM-Backend/1.0"},
            follow_redirects=True,
            max_redirects=5,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
            ),
        )
    return _http_client
class ImageFetchError(Exception):
    pass
class ImageTooLargeError(ImageFetchError):
    pass
class ImageFetcher:
    def __init__(
        self,
        local_base_path: str = "static/uploads",
        max_size_bytes: int = 50 * 1024 * 1024,  
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self._local_base = os.path.abspath(local_base_path)
        self._max_size = max_size_bytes
        self._client = http_client
    async def fetch(self, url: str) -> bytes:
        if "localhost" in url and "static/uploads" in url:
            try:
                return await self._fetch_local(url)
            except ImageFetchError:
                logger.info(f"Local fallback failed for {url}, trying remote")
        return await self._fetch_remote(url)
    async def _fetch_local(self, url: str) -> bytes:
        filename = url.split("/")[-1]
        target = os.path.abspath(os.path.join(self._local_base, filename))
        if not target.startswith(self._local_base + os.sep) and target != self._local_base:
            logger.warning(f"Path traversal blocked: {filename}")
            raise ImageFetchError("Invalid local path")
        if not os.path.exists(target):
            raise ImageFetchError(f"Local file not found: {target}")
        size = os.path.getsize(target)
        if size > self._max_size:
            raise ImageTooLargeError(
                f"Local file {filename} ({size} B) exceeds limit {self._max_size} B"
            )
        return await run_in_threadpool(self._read_file_sync, target)
    @staticmethod
    def _read_file_sync(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()
    async def _fetch_remote(self, url: str) -> bytes:
        client = self._client if self._client is not None else _get_shared_client()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ImageFetchError(f"Unsupported URL scheme: {parsed.scheme}")
        try:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self._max_size:
                    raise ImageTooLargeError(
                        f"Content-Length {content_length} exceeds max {self._max_size}"
                    )
                chunks = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    chunks.extend(chunk)
                    if len(chunks) > self._max_size:
                        raise ImageTooLargeError(
                            "Download exceeded maximum size limit during streaming"
                        )
                return bytes(chunks)
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"HTTP {exc.response.status_code} while fetching {url}")
            raise ImageFetchError(
                f"Upstream returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error(f"Timeout while fetching {url}")
            raise ImageFetchError("Image fetch timed out") from exc
        except httpx.RequestError as exc:
            logger.error(f"Network error fetching {url}: {exc}")
            raise ImageFetchError("Network error during image fetch") from exc

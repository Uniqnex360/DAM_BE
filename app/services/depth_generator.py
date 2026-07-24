# import io
# import os
# import logging
# import tempfile
# from gradio_client import Client
# from PIL import Image
# from app.core.config import settings

# logger = logging.getLogger(__name__)

# class ThreeDGenerator:
#     """Uses InstantMesh (hosted on Hugging Face GPUs) for high-quality 3D generation"""
    
#     def generate_3d_mesh(self, image_bytes: bytes) -> bytes:
#         temp_input = None
#         try:
#             # 1. Save image to a temp file (gradio_client needs a file path)
#             temp_input = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            
#             # Ensure the image is clean RGB
#             image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#             image.save(temp_input.name, format="PNG")
#             temp_input.close()

#             logger.info("Connecting to InstantMesh on Hugging Face...")
#             client = Client("TencentARC/InstantMesh", hf_token=settings.HF_TOKEN)


#             # 2. Pre-process (Background removal + centering)
#             logger.info("Step 1/3: Pre-processing image...")
#             processed = client.predict(
#     temp_input.name,
#     True,
#     api_name="/preprocess"
# )

#             # 3. Generate 6 views (Front, Back, Left, Right, Top, Bottom)
#             logger.info("Step 2/3: Generating multi-view images...")
#             mv_result = client.predict(
#     processed,
#     75,
#     42,
#     api_name="/generate_mvs"
# )

#             # 4. Reconstruct the 3D mesh from all 6 views
#             logger.info("Step 3/3: Reconstructing 3D mesh...")
#             mesh_path = client.predict(
#                 api_name="/make3d"
#             )

#             # 5. Read the generated .obj or .glb file
#             # InstantMesh returns a file path to the generated model
#             if isinstance(mesh_path, tuple):
#                 mesh_path = mesh_path[0]  # Sometimes returns (path, path)
            
#             with open(mesh_path, "rb") as f:
#                 mesh_data = f.read()

#             logger.info(f"3D mesh generated successfully! Size: {len(mesh_data)} bytes")
#             return mesh_data

#         except Exception as e:
#             logger.error(f"3D Mesh generation failed: {e}")
#             raise

#         finally:
#             # Clean up temp file
#             if temp_input and os.path.exists(temp_input.name):
#                 os.remove(temp_input.name)

# # Backward compatibility alias
# DepthGenerator = ThreeDGenerator
# import io
# import os
# import logging
# import tempfile
# from gradio_client import Client
# from PIL import Image
# from rembg import remove
# from app.core.config import settings

# logger = logging.getLogger(__name__)

# class ThreeDGenerator:
#     def generate_3d_mesh(self, image_bytes: bytes) -> bytes:
#         temp_input = None
#         try:
#             # 1. Local Pre-processing
#             logger.info("Cleaning up image background...")
#             image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#             image = remove(image) # Use rembg to clean the image locally
            
#             # Save to temp file
#             temp_input = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
#             image.save(temp_input.name, format="PNG")
#             temp_input.close()

#             # 2. Connect to Stability AI
#             logger.info("Connecting to Stability AI 3D Engine...")
#             client = Client("stabilityai/stable-fast-3d", hf_token=settings.HF_TOKEN)

#             # 3. Call using api_name instead of fn_index
#             logger.info("Reconstructing high-quality 3D mesh...")
#             try:
#                 result = client.predict(
#                     temp_input.name,     # Argument 1: Image path
#                     0.85,                # Argument 2: Foreground Ratio
#                     api_name="/predict"  # Identifies the correct function safely
#                 )
#             except Exception as api_err:
#                 # If it still fails, print out the remote API structure to your container logs 
#                 # so you can see exactly what they changed.
#                 logger.error("API Call failed. Printing current API Schema:")
#                 client.view_api()
#                 raise api_err

#             # 4. Handle result
#             mesh_path = result
#             if isinstance(result, (list, tuple)):
#                 mesh_path = result[0]
            
#             with open(mesh_path, "rb") as f:
#                 mesh_data = f.read()

#             logger.info(f"3D mesh generated! Size: {len(mesh_data)} bytes")
#             return mesh_data

#         except Exception as e:
#             logger.error(f"3D Generation failed: {e}")
#             raise
#         finally:
#             if temp_input and os.path.exists(temp_input.name):
#                 os.remove(temp_input.name)

# DepthGenerator = ThreeDGenerator

# import torch
# import io
# import logging
# from PIL import Image
# from tsr.system import TSR
# from tsr.utils import remove_background

# logger = logging.getLogger(__name__)

# class ThreeDGenerator:
#     _instance = None
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#             cls._instance._initialized = False
#         return cls._instance
    
#     def __init__(self):
#         if self._initialized: return
#         self._initialized = True
#         logger.info("Loading TripoSR local model...")
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # This loads weights automatically to /root/.cache/huggingface
#         self.model = TSR.from_pretrained(
#             "stabilityai/TripoSR",
#             config_name="config.yaml",
#             weight_name="model.ckpt"
#         )
#         self.model.to(self.device)
    
#     def generate_3d_mesh(self, image_bytes: bytes) -> bytes:
#         try:
#             image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#             # 1. Clean background
#             image = remove_background(image)
#             if image.mode == 'RGBA':
#                 # Create a white background canvas
#                 white_bg = Image.new("RGB", image.size, (255, 255, 255))
#                 # Paste the image using its own transparency as a mask
#                 white_bg.paste(image, mask=image.split()[3]) 
#                 image = white_bg
#             # 2. AI Mesh Generation
#             with torch.no_grad():
#                 scene_codes = self.model([image], device=self.device)
            
#             # 3. Extract (Use 256 for CPU speed, 384 or 512 for better AWS results)
#             meshes = self.model.extract_mesh(scene_codes, True, resolution=256)
            
#             # 4. Export to GLB
#             output = io.BytesIO()
#             meshes[0].export(output, file_type='glb')
#             return output.getvalue()
#         except Exception as e:
#             logger.error(f"Local 3D Error: {e}"); raise

# DepthGenerator = ThreeDGenerator

# import io
# import os
# import time
# import logging
# import tempfile
# from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
# from gradio_client import Client, handle_file
# from PIL import Image
# from rembg import remove
# from app.core.config import settings

# logger = logging.getLogger(__name__)

# class ThreeDGenerator:
#     _client = None  # reuse connection, don't reconnect every call

#     def _get_client(self) -> Client:
#         if self._client is None:
#             logger.info("Connecting to Stability AI 3D Engine...")
#             self._client = Client("stabilityai/stable-fast-3d", hf_token=settings.HF_TOKEN)
#         return self._client

#     def _call_with_timeout(self, client, image_path, timeout=60):
#         # USE THIS ENDPOINT: /requires_bg_remove (Reliable API endpoint)
#         def _predict():
#             return client.predict(
#                 image=handle_file(image_path),
#                 fr=0.85,
#                 api_name="/requires_bg_remove",
#             )
#         with ThreadPoolExecutor(max_workers=1) as executor:
#             future = executor.submit(_predict)
#             return future.result(timeout=timeout)

#     # def generate_3d_mesh(self, image_bytes: bytes, max_retries: int = 3, timeout: int = 60) -> bytes:
#     #     temp_input = None
#     #     try:
#     #         image = Image.open(io.BytesIO(image_bytes))
#     #         image.verify()
#     #         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#     #         logger.info("Cleaning up image background...")
#     #         image = remove(image)

#     #         # if image.mode == "RGBA":
#     #         #     white_bg = Image.new("RGB", image.size, (255, 255, 255))
#     #         #     white_bg.paste(image, mask=image.split()[3])
#     #         #     image = white_bg
#     #         if image.mode != "RGBA":
#     #             image = image.convert("RGBA")

#     #         temp_input = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
#     #         image.save(temp_input.name, format="PNG")
#     #         temp_input.close()
#     #         image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
#     #         canvas = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
#     #         # Center image
#     #         offset = ((1024 - image.width) // 2, (1024 - image.height) // 2)
#     #         canvas.paste(image, offset)

#     #         canvas.save(temp_input.name, format="PNG")
#     #         temp_input.close()
#     #         client = self._get_client()

#     #         last_err = None
#     #         mesh_path = None
#     #         for attempt in range(1, max_retries + 1):
#     #             try:
#     #                 logger.info(f"Reconstructing mesh, attempt {attempt}/{max_retries}...")
#     #                 result = self._call_with_timeout(client, temp_input.name, timeout=timeout)
                    
#     #                 # Extract the 3D model object (always the last returned item)
#     #                 mesh_obj = result[-1] if isinstance(result, (tuple, list)) else result
                    
#     #                 # Resolve string path from dict/object/string
#     #                 if isinstance(mesh_obj, dict):
#     #                     mesh_path = mesh_obj.get("path") or mesh_obj.get("name")
#     #                 elif hasattr(mesh_obj, "path"):
#     #                     mesh_path = mesh_obj.path
#     #                 else:
#     #                     mesh_path = mesh_obj

#     #                 if not mesh_path:
#     #                     raise ValueError(f"Could not extract file path from result: {result}")

#     #                 break
#     #             except FutureTimeoutError as e:
#     #                 last_err = f"Timeout after {timeout}s"
#     #                 logger.warning(f"Attempt {attempt} timed out after {timeout}s")
#     #             except Exception as e:
#     #                 last_err = f"{type(e).__name__}: {e!r}"
#     #                 logger.warning(f"Attempt {attempt} failed: {type(e).__name__}: {e!r}")
#     #                 logger.exception("Full traceback:")  # <-- actually shows what broke

#     #             if attempt < max_retries:
#     #                 time.sleep(2 ** attempt)
#     #         else:
#     #             logger.error("All retries exhausted. Dumping API schema for diagnosis:")
#     #             try:
#     #                 client.view_api()
#     #             except Exception:
#     #                 pass
#     #             raise RuntimeError(f"3D generation failed after {max_retries} attempts: {last_err}")

#     #         # mesh_path already correct from loop — no reassignment here
#     #         if isinstance(mesh_path, dict):
#     #             mesh_path = mesh_path.get("path") or mesh_path.get("name")

#     #         with open(mesh_path, "rb") as f:
#     #             mesh_data = f.read()

#     #         logger.info(f"3D mesh generated! Size: {len(mesh_data)} bytes")
#     #         return mesh_data

#     #     except Exception as e:
#     #         logger.error(f"3D Generation failed: {type(e).__name__}: {e!r}")
#     #         raise
#     #     finally:
#     #         if temp_input and os.path.exists(temp_input.name):
#     #             os.remove(temp_input.name)
#     def generate_3d_mesh(self, image_bytes: bytes, max_retries: int = 3, timeout: int = 60) -> bytes:
#         temp_input = None
#         try:
#             image = Image.open(io.BytesIO(image_bytes))
#             image.verify()
#             image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#             logger.info("Cleaning up image background...")
#             image = remove(image)

#             # 1. Crop tight to the subject
#             bbox = image.getbbox()
#             if bbox:
#                 image = image.crop(bbox)

#             # 2. Resize maintaining aspect ratio within 512x512 (Native SF3D resolution)
#             image.thumbnail((512, 512), Image.Resampling.LANCZOS)

#             # 3. Composite onto a clean white RGB background (Avoid alpha channel bugs on HF)
#             white_bg = Image.new("RGB", (512, 512), (255, 255, 255))
            
#             # Center the image
#             offset = ((512 - image.width) // 2, (512 - image.height) // 2)
#             if image.mode == "RGBA":
#                 white_bg.paste(image, offset, mask=image.split()[3])
#             else:
#                 white_bg.paste(image, offset)

#             # 4. Save to temporary disk location
#             temp_input = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
#             white_bg.save(temp_input.name, format="JPEG", quality=95)
#             temp_input.close()

#             client = self._get_client()

#             last_err = None
#             mesh_path = None

#             for attempt in range(1, max_retries + 1):
#                 try:
#                     logger.info(f"Reconstructing mesh, attempt {attempt}/{max_retries}...")
#                     result = self._call_with_timeout(client, temp_input.name, timeout=timeout)
                    
#                     mesh_obj = result[-1] if isinstance(result, (tuple, list)) else result
                    
#                     if isinstance(mesh_obj, dict):
#                         if mesh_obj.get("visible") is False:
#                             raise ValueError("Remote SF3D model rejected image (returned visible=False)")
#                         mesh_path = mesh_obj.get("path") or mesh_obj.get("value") or mesh_obj.get("name")
#                     elif hasattr(mesh_obj, "path"):
#                         mesh_path = mesh_obj.path
#                     else:
#                         mesh_path = str(mesh_obj)

#                     if not mesh_path or not os.path.exists(mesh_path):
#                         raise ValueError(f"Could not extract valid mesh path from result: {result}")

#                     break

#                 except FutureTimeoutError:
#                     last_err = f"Timeout after {timeout}s"
#                     logger.warning(f"Attempt {attempt} timed out after {timeout}s")
#                 except Exception as e:
#                     last_err = f"{type(e).__name__}: {e!r}"
#                     logger.warning(f"Attempt {attempt} failed: {type(e).__name__}: {e!r}")

#                 if attempt < max_retries:
#                     time.sleep(2 ** attempt)
#             else:
#                 raise RuntimeError(f"3D generation failed after {max_retries} attempts: {last_err}")

#             with open(mesh_path, "rb") as f:
#                 mesh_data = f.read()

#             logger.info(f"3D mesh generated! Size: {len(mesh_data)} bytes")
#             return mesh_data

#         except Exception as e:
#             logger.error(f"3D Generation failed: {type(e).__name__}: {e!r}")
#             raise
#         finally:
#             if temp_input and os.path.exists(temp_input.name):
#                 try:
#                     os.remove(temp_input.name)
#                 except OSError:
#                     pass

# DepthGenerator = ThreeDGenerator
import io
import os
import time
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from gradio_client import Client, handle_file
from PIL import Image
from app.core.config import settings

logger = logging.getLogger(__name__)

class ThreeDGenerator:
    _client = None

    def _get_client(self) -> Client:
        if self._client is None:
            logger.info("Connecting to InstantMesh Space...")
            self._client = Client("TencentARC/InstantMesh", hf_token=settings.HF_TOKEN)
        return self._client

    def _call_with_timeout(self, client, image_path, timeout=120):
        def _run():
            # Step 1: preprocess (bg removal + centering)
            processed = client.predict(
                handle_file(image_path),
                True,
                api_name="/preprocess"
            )
            # Step 2: generate 6 multi-view images
            mv_result = client.predict(
                processed,
                75,   # sample steps
                42,   # seed
                api_name="/generate_mvs"
            )
            # Step 3: reconstruct mesh
            mesh_path = client.predict(api_name="/make3d")
            return mesh_path

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            return future.result(timeout=timeout)

    def generate_3d_mesh(self, image_bytes: bytes, max_retries: int = 3, timeout: int = 120) -> bytes:
        temp_input = None
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            temp_input = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            image.save(temp_input.name, format="PNG")
            temp_input.close()

            client = self._get_client()
            last_err = None
            mesh_path = None

            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"InstantMesh reconstruction, attempt {attempt}/{max_retries}...")
                    mesh_path = self._call_with_timeout(client, temp_input.name, timeout=timeout)
                    if isinstance(mesh_path, (tuple, list)):
                        mesh_path = mesh_path[0]
                    break
                except FutureTimeoutError:
                    last_err = f"Timeout after {timeout}s"
                    logger.warning(f"Attempt {attempt} timed out")
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e!r}"
                    logger.warning(f"Attempt {attempt} failed: {last_err}")

                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, 20))
            else:
                raise RuntimeError(f"InstantMesh generation failed after {max_retries} attempts: {last_err}")

            with open(mesh_path, "rb") as f:
                mesh_data = f.read()

            logger.info(f"Mesh generated! Size: {len(mesh_data)} bytes")
            return mesh_data

        except Exception as e:
            logger.error(f"InstantMesh generation failed: {type(e).__name__}: {e!r}")
            raise
        finally:
            if temp_input and os.path.exists(temp_input.name):
                os.remove(temp_input.name)

DepthGenerator = ThreeDGenerator
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
import io
import os
import logging
import tempfile
from gradio_client import Client
from PIL import Image
from rembg import remove
from app.core.config import settings

logger = logging.getLogger(__name__)

class ThreeDGenerator:
    def generate_3d_mesh(self, image_bytes: bytes) -> bytes:
        temp_input = None
        try:
            # 1. Local Pre-processing
            logger.info("Cleaning up image background...")
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image = remove(image) # Use rembg to clean the image locally
            
            # Save to temp file
            temp_input = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            image.save(temp_input.name, format="PNG")
            temp_input.close()

            # 2. Connect to Stability AI
            logger.info("Connecting to Stability AI 3D Engine...")
            client = Client("stabilityai/stable-fast-3d", hf_token=settings.HF_TOKEN)

            # 3. Call using api_name instead of fn_index
            logger.info("Reconstructing high-quality 3D mesh...")
            try:
                result = client.predict(
                    temp_input.name,     # Argument 1: Image path
                    0.85,                # Argument 2: Foreground Ratio
                    api_name="/predict"  # Identifies the correct function safely
                )
            except Exception as api_err:
                # If it still fails, print out the remote API structure to your container logs 
                # so you can see exactly what they changed.
                logger.error("API Call failed. Printing current API Schema:")
                client.view_api()
                raise api_err

            # 4. Handle result
            mesh_path = result
            if isinstance(result, (list, tuple)):
                mesh_path = result[0]
            
            with open(mesh_path, "rb") as f:
                mesh_data = f.read()

            logger.info(f"3D mesh generated! Size: {len(mesh_data)} bytes")
            return mesh_data

        except Exception as e:
            logger.error(f"3D Generation failed: {e}")
            raise
        finally:
            if temp_input and os.path.exists(temp_input.name):
                os.remove(temp_input.name)

DepthGenerator = ThreeDGenerator

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
import os
import requests
import json
import time

# ==========================================
# CONFIGURATION
# ==========================================
API_URL = "http://localhost:8000/api/v1"
INPUT_DIR = "test_images"  # Folder containing images to test
EMAIL = "wewejit442@naqulu.com" # CHANGE THIS to your registered user
PASSWORD = "123123"   # CHANGE THIS to your password

# ==========================================
# UTILS
# ==========================================
def login():
    print(f"🔑 Logging in as {EMAIL}...")
    try:
        response = requests.post(
            f"{API_URL}/auth/login",
            data={"username": EMAIL, "password": PASSWORD}
        )
        if response.status_code != 200:
            print(f"❌ Login failed: {response.text}")
            exit(1)
        token = response.json()["access_token"]
        print("✅ Login successful!\n")
        return {"Authorization": f"Bearer {token}"}
    except Exception as e:
        print(f"❌ Connection refused. Is the server running? {e}")
        exit(1)

def run_batch_test():
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"❌ Folder '{INPUT_DIR}' created. Please put images inside it and restart.")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    if not files:
        print(f"❌ No images found in '{INPUT_DIR}'")
        return

    headers = login()
    
    print(f"🚀 Starting Batch Test on {len(files)} images...")
    print("=" * 60)

    for filename in files:
        file_path = os.path.join(INPUT_DIR, filename)
        print(f"📄 Processing: {filename}")

        # 1. UPLOAD
        print("   ⬆️  Uploading...", end=" ", flush=True)
        try:
            with open(file_path, "rb") as f:
                files_payload = {'file': (filename, f, 'image/jpeg')}
                up_res = requests.post(f"{API_URL}/assets/upload", headers=headers, files=files_payload)
            
            if up_res.status_code != 200:
                print(f"❌ Upload Failed: {up_res.text}")
                continue
                
            asset_data = up_res.json()
            asset_id = asset_data['id']
            print("✅ Uploaded")
            
            # Check if using local storage
            if "localhost" in asset_data['url']:
                print(f"      (Saved locally at: {asset_data['url']})")

        except Exception as e:
            print(f"❌ Error during upload: {e}")
            continue

        # 2. PROCESS
        print("   🧠 Processing...", end=" ", flush=True)
        try:
            # Note: This might take time depending on image size
            proc_res = requests.post(f"{API_URL}/assets/{asset_id}/process", headers=headers)
            
            if proc_res.status_code != 200:
                print(f"❌ Processing Failed: {proc_res.text}")
                continue

            result = proc_res.json()
            telemetry = result['telemetry']
            
            print(f"✅ Done ({telemetry['time_ms']}ms)")
            
            # Print Stats
            steps = telemetry['steps']
            conf = telemetry['confidence']
            
            print(f"      🛠️  Steps: {steps if steps else 'None'}")
            print(f"      📊 Confidence: BG Clean: {conf['bg_clean']:.2f} | Crop: {conf['crop']:.2f}")

        except Exception as e:
            print(f"❌ Error during processing: {e}")

        print("-" * 60)

if __name__ == "__main__":
    run_batch_test()
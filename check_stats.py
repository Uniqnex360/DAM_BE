import requests
import json

API_URL = "http://localhost:8000/api/v1"
EMAIL = "wewejit442@naqulu.com"  # Your email
PASSWORD = "123123"    # Your password

def get_stats():
    # 1. Login
    try:
        auth_res = requests.post(f"{API_URL}/auth/login", data={"username": EMAIL, "password": PASSWORD})
        token = auth_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
    except:
        print("Login failed. Is server running?")
        return

    # 2. Get Overview
    print("📊 Fetching Dashboard Analytics...\n")
    res = requests.get(f"{API_URL}/dashboard/overview", headers=headers)
    
    if res.status_code == 200:
        data = res.json()
        print(json.dumps(data, indent=2))
    else:
        print(f"Error: {res.text}")

if __name__ == "__main__":
    get_stats()
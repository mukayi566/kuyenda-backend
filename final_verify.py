import requests
import json
import time

# Configuration
BASE_URL = "http://192.168.1.60:8000"
TEST_ID = int(time.time())

print("="*60)
print(f"🚀 FINAL SYSTEM VERIFICATION (Test ID: {TEST_ID})")
print(f"Target Server: {BASE_URL}")
print("="*60)

# 1. Test Server Connectivity
print("\n[1/3] CHECKING SERVER HEALTH...")
try:
    resp = requests.get(f"{BASE_URL}/", timeout=5)
    if resp.status_code == 200:
        print("✅ Server is reachable and healthy.")
    else:
        print(f"❌ Server returned unexpected status: {resp.status_code}")
        exit(1)
except Exception as e:
    print(f"❌ Could not connect to server: {e}")
    print("   -> Check if uvicorn is running")
    print("   -> Check if Firewall is blocking port 8000")
    exit(1)

# 2. Test Login (Connectivity Check)
print("\n[2/3] VERIFYING LOGIN ENDPOINT...")
# We use a known fake user just to check if endpoint responds, not to actually login
try:
    login_data = {"email": "check_conn@test.com", "password": "password123"}
    resp = requests.post(f"{BASE_URL}/auth/login", json=login_data, timeout=10)
    
    # We expect 401 (Invalid creds) or 200 (Success). 
    # If we get connection error, that's a fail.
    if resp.status_code in [200, 401]:
        print(f"✅ Login endpoint is connected (Status: {resp.status_code})")
    else:
        print(f"⚠️  Login endpoint returned unusual status: {resp.status_code}")
        print(f"   Response: {resp.text}")

except Exception as e:
    print(f"❌ Login endpoint connection failed: {e}")

# 3. Test Signup (Final Check)
print("\n[3/3] VERIFYING SIGNUP ENDPOINT...")
try:
    signup_data = {
        "email": f"verify_{TEST_ID}@test.com",
        "password": "Verify123456!",
        "full_name": "Verification User",
        "user_type": "rider"
    }
    
    print(f"   Attempting signup for {signup_data['email']}...")
    resp = requests.post(f"{BASE_URL}/auth/signup", json=signup_data, timeout=30)
    
    if resp.status_code == 200:
        print("✅ Signup successful! User created.")
        print(f"   Token received: {resp.json().get('token')[:15]}...")
    elif "timed out" in resp.text:
        print("❌ Signup timed out.")
        print("   -> ACTION REQUIRED: Disable Email Confirmation in Supabase.")
    else:
        print(f"⚠️  Signup returned: {resp.status_code}")
        print(f"   Response: {resp.text}")

except requests.exceptions.Timeout:
    print("❌ Signup Request Timed Out (30s)")
    print("   -> CRITICAL: Supabase is waiting for email confirmation.")
    print("   -> GO TO SUPABASE DASHBOARD AND DISABLE EMAIL CONFIRMATIONS.")
except Exception as e:
    print(f"❌ Signup endpoint connection failed: {e}")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)

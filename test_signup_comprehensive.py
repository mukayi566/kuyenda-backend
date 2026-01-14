import requests
import json
import time

BASE_URL = "http://192.168.1.60:8000"

with open("test_results2.txt", "w", encoding='utf-8') as f:
    f.write("=== Testing Signup with New User ===\n")
    signup_data = {
        "email": f"testuser{int(time.time())}@example.com",
        "password": "Test123456",
        "full_name": "Brand New User",
        "user_type": "rider"
    }

    f.write(f"Attempting signup with email: {signup_data['email']}\n")
    
    try:
        response = requests.post(f"{BASE_URL}/auth/signup", json=signup_data, timeout=30)
        f.write(f"Status Code: {response.status_code}\n")
        f.write(f"Response: {json.dumps(response.json(), indent=2)}\n")
        
        if response.status_code == 200:
            f.write("\nSIGNUP SUCCESSFUL!\n")
            token = response.json().get('token')
            user_id = response.json().get('user_id')
            
            # Test if we can fetch profile
            f.write("\n=== Testing Profile Fetch ===\n")
            headers = {"Authorization": f"Bearer {token}"}
            profile_response = requests.get(f"{BASE_URL}/profiles/me", headers=headers, timeout=10)
            f.write(f"Profile Status Code: {profile_response.status_code}\n")
            f.write(f"Profile Response: {json.dumps(profile_response.json(), indent=2)}\n")
        else:
            f.write("\nSIGNUP FAILED\n")
            
    except requests.exceptions.Timeout:
        f.write("Request timed out after 30 seconds\n")
    except requests.exceptions.RequestException as e:
        f.write(f"Signup Error: {e}\n")
        if hasattr(e, 'response') and e.response:
            f.write(f"Response status: {e.response.status_code}\n")
            f.write(f"Response text: {e.response.text}\n")

print("Test complete. Check test_results2.txt for output.")

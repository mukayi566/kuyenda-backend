import requests
import json

BASE_URL = "http://192.168.1.60:8000"

with open("test_results.txt", "w") as f:
    f.write("=== Testing Signup ===\n")
    signup_data = {
        "email": "newuser123@example.com",
        "password": "Test123456",
        "full_name": "New Test User",
        "user_type": "rider"
    }

    try:
        response = requests.post(f"{BASE_URL}/auth/signup", json=signup_data, timeout=10)
        f.write(f"Status Code: {response.status_code}\n")
        f.write(f"Response: {json.dumps(response.json(), indent=2)}\n")
    except requests.exceptions.RequestException as e:
        f.write(f"Signup Error: {e}\n")
        if hasattr(e, 'response') and e.response:
            f.write(f"Response status: {e.response.status_code}\n")
            f.write(f"Response text: {e.response.text}\n")

    f.write("\n=== Testing Login with existing user ===\n")
    login_data = {
        "email": "newuser123@example.com",
        "password": "Test123456"
    }

    try:
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data, timeout=10)
        f.write(f"Status Code: {response.status_code}\n")
        f.write(f"Response: {json.dumps(response.json(), indent=2)}\n")
    except requests.exceptions.RequestException as e:
        f.write(f"Login Error: {e}\n")
        if hasattr(e, 'response') and e.response:
            f.write(f"Response status: {e.response.status_code}\n")
            f.write(f"Response text: {e.response.text}\n")

print("Test complete. Check test_results.txt for output.")

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

print(f"Supabase URL: {url}")
print(f"Supabase Key: {key[:20]}..." if key else "No key")

supabase = create_client(url, key)

# Test signup
try:
    print("\n=== Testing Signup ===")
    test_email = "test@example.com"
    test_password = "Test123456"
    
    auth_response = supabase.auth.sign_up({
        "email": test_email,
        "password": test_password,
        "options": {
            "data": {
                "full_name": "Test User",
                "user_type": "rider"
            }
        }
    })
    
    print(f"Signup Response: {auth_response}")
    if auth_response.user:
        print(f"User created: {auth_response.user.id}")
    else:
        print("No user returned from signup")
        
except Exception as e:
    print(f"Signup Error: {e}")
    import traceback
    traceback.print_exc()

# Test login
try:
    print("\n=== Testing Login ===")
    auth_response = supabase.auth.sign_in_with_password({
        "email": test_email,
        "password": test_password
    })
    
    print(f"Login Response: {auth_response}")
    if auth_response.user:
        print(f"User logged in: {auth_response.user.id}")
    else:
        print("No user returned from login")
        
except Exception as e:
    print(f"Login Error: {e}")
    import traceback
    traceback.print_exc()

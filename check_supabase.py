
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

print(f"URL: {url}")
print(f"Key Found: {'Yes' if key else 'No'}")

if not url or "your-project" in url:
    print("ERROR: Invalid Supabase URL in .env")
    exit(1)

try:
    supabase: Client = create_client(url, key)
    print("Client created.")
    
    # Try a simple read or auth check
    # We can't easily sign up a dummy user without polluting DB, 
    # but we can try to sign in with a non-existent user and expect a specific error.
    try:
        res = supabase.auth.sign_in_with_password({"email": "test@kuyenda.com", "password": "wrongpassword"})
    except Exception as e:
        print(f"Auth Request Result: {e}")
        # If we get "Invalid login credentials", connection is GOOD.
        # If we get "Request timed out" or "Name or service not known", connection is BAD.
        
except Exception as e:
    print(f"CRITICAL ERROR: {e}")

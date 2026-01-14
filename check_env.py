from dotenv import load_dotenv
import os
from supabase import create_client

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

print(f"URL: {url}")
if key:
    print(f"KEY: {key[:5]}...{key[-5:]}")
else:
    print("KEY: None")

if not url or "your-anon-key" in key:
    print("INVALID CONFIG FOUND")
else:
    try:
        supabase = create_client(url, key)
        # Try a simple select to verify connection
        print("Attempting connection...")
        response = supabase.table("profiles").select("count", count="exact").execute()
        print("Connection Successful!")
    except Exception as e:
        print(f"Connection Failed: {e}")

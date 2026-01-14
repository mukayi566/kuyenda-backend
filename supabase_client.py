import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

if not url or not key:
    # We'll use dummy values for now to avoid crashing, 
    # but the user needs to provide real ones.
    url = "https://azplaptlxdxsgfjhgdhg.supabase.co"
    key = "your-anon-key"
    print("WARNING: SUPABASE_URL and SUPABASE_KEY not found in environment variables.")

# Create Supabase client with simple configuration
supabase: Client = create_client(url, key)

def get_supabase():
    return supabase

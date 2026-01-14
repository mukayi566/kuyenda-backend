"""
Diagnostic script to check Supabase configuration and identify signup issues.
"""
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("SUPABASE CONFIGURATION DIAGNOSTIC")
print("=" * 60)

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

print(f"\n1. Environment Variables:")
print(f"   SUPABASE_URL: {url}")
print(f"   SUPABASE_KEY: {'*' * 20}{key[-10:] if key else 'NOT SET'}")

# Check if it's service_role or anon key
if key:
    import jwt
    try:
        decoded = jwt.decode(key, options={"verify_signature": False})
        role = decoded.get('role', 'unknown')
        print(f"   Key Role: {role}")
        if role == 'service_role':
            print("   ✓ Using service_role key (Good for backend)")
        elif role == 'anon':
            print("   ⚠ Using anon key (May have restrictions)")
    except:
        print("   Could not decode key")

print(f"\n2. Testing Supabase Connection:")
try:
    from supabase_client import get_supabase
    supabase = get_supabase()
    print("   ✓ Supabase client created successfully")
    
    # Try to query profiles table
    try:
        result = supabase.table("profiles").select("id").limit(1).execute()
        print(f"   ✓ Can query profiles table ({len(result.data)} rows)")
    except Exception as e:
        print(f"   ✗ Cannot query profiles table: {e}")
        
except Exception as e:
    print(f"   ✗ Failed to create Supabase client: {e}")

print(f"\n3. Common Issues & Solutions:")
print("   Issue: 'The read operation timed out'")
print("   Solution: Disable email confirmation in Supabase Dashboard")
print("   Path: Authentication → Settings → Email Auth → Disable 'Confirm email'")
print()
print("   Issue: 'Profile creation failed'")
print("   Solution: Ensure profiles table has proper trigger or RLS policies")
print()
print("   Issue: 'Already registered'")
print("   Solution: User already exists, try logging in instead")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

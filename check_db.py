from supabase_client import get_supabase
supabase = get_supabase()

try:
    res = supabase.table("profiles").select("*").limit(1).execute()
    if res.data:
        print("Columns in profiles:", res.data[0].keys())
    else:
        # If no data, try to insert dummy data and see if it fails
        print("No data in profiles table.")
        # Try to inspect columns via RPC or just check if 'email' exists
        try:
            supabase.table("profiles").select("email").limit(1).execute()
            print("Column 'email' exists.")
        except Exception as e:
            print("Column 'email' probably does NOT exist:", e)
except Exception as e:
    print("Error checking profiles table:", e)

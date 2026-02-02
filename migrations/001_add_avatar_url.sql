-- Supabase SQL Migration: Add/Update avatar_url column in profiles table
-- Run this in Supabase Dashboard > SQL Editor

-- 1. Add avatar_url column if it doesn't exist
ALTER TABLE profiles
ADD COLUMN IF NOT EXISTS avatar_url TEXT;

-- 2. Add a helpful comment
COMMENT ON COLUMN profiles.avatar_url IS 'Public URL of user avatar stored in Supabase Storage (bucket: avatars)';

-- 3. Optional: Create an index for faster lookups (if you query by avatar_url)
-- CREATE INDEX IF NOT EXISTS idx_profiles_avatar_url ON profiles(avatar_url);

-- 4. Clean up any local file:// paths that may have been stored incorrectly
-- This sets them to NULL so the app shows initials instead
UPDATE profiles
SET avatar_url = NULL
WHERE avatar_url LIKE 'file://%'
   OR avatar_url LIKE 'content://%'
   OR avatar_url = '';

-- 5. Verify the column exists and show sample data
SELECT 
    id,
    full_name,
    email,
    avatar_url,
    CASE 
        WHEN avatar_url IS NULL THEN 'No avatar (will show initials)'
        WHEN avatar_url LIKE 'https://%' THEN 'Supabase Storage URL ✓'
        WHEN avatar_url LIKE 'http://%' THEN 'Legacy HTTP URL'
        ELSE 'Other/Unknown'
    END AS avatar_status
FROM profiles
LIMIT 10;

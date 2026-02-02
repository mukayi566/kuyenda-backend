# Supabase Storage Setup for Avatar Images

This guide explains how to set up Supabase Storage to persist user avatar images permanently.

## Why Was Cache Clearing Causing Avatar Loss?

**Before (The Problem):**
1. User picks image → Expo saves to device cache (`file:///data/user/0/.../cache/...`)
2. App stores this local file path in AsyncStorage or profiles table
3. User clears app cache → Cache directory deleted → Avatar gone!

**After (The Solution):**
1. User picks image → App uploads to Supabase Storage (cloud)
2. Supabase returns permanent public URL (`https://xxx.supabase.co/storage/v1/...`)
3. App stores this URL in `profiles.avatar_url`
4. User clears cache → URL still works → Avatar persists!

---

## Step 1: Create Storage Bucket in Supabase

1. Go to your **Supabase Dashboard** → **Storage**
2. Click **"New bucket"**
3. Configure:
   - **Name**: `avatars`
   - **Public**: ✅ **Yes** (recommended for profile images)
   - Click **Create bucket**

### Why Public vs Private?

| Type | Pros | Cons | Use Case |
|------|------|------|----------|
| **Public** | Simple URL, no expiry, fast | Anyone with URL can view | Profile avatars, public content |
| **Private** | Secure, signed URLs | URLs expire, more complexity | Private documents, sensitive files |

**Recommendation**: Use **public** for avatars. Profile pictures are typically visible to others anyway.

---

## Step 2: Set Storage Policies (RLS)

Go to **Storage** → **avatars** bucket → **Policies** tab.

### Policy 1: Allow authenticated users to upload to their own folder

```sql
-- INSERT Policy (Upload)
CREATE POLICY "Users can upload their own avatar"
ON storage.objects
FOR INSERT
TO authenticated
WITH CHECK (
    bucket_id = 'avatars' AND
    (storage.foldername(name))[1] = auth.uid()::text
);
```

### Policy 2: Allow users to update/delete their own avatars

```sql
-- UPDATE Policy
CREATE POLICY "Users can update their own avatar"
ON storage.objects
FOR UPDATE
TO authenticated
USING (
    bucket_id = 'avatars' AND
    (storage.foldername(name))[1] = auth.uid()::text
);

-- DELETE Policy
CREATE POLICY "Users can delete their own avatar"
ON storage.objects
FOR DELETE
TO authenticated
USING (
    bucket_id = 'avatars' AND
    (storage.foldername(name))[1] = auth.uid()::text
);
```

### Policy 3: Allow public read access (for public bucket)

```sql
-- SELECT Policy (Read - for public avatars)
CREATE POLICY "Public can read all avatars"
ON storage.objects
FOR SELECT
TO public
USING (bucket_id = 'avatars');
```

---

## Step 3: Ensure profiles Table Has avatar_url Column

Run this in **SQL Editor**:

```sql
-- Add avatar_url column if it doesn't exist
ALTER TABLE profiles
ADD COLUMN IF NOT EXISTS avatar_url TEXT;

-- Optional: Add a comment for documentation
COMMENT ON COLUMN profiles.avatar_url IS 'Public URL of user avatar stored in Supabase Storage';
```

---

## Step 4: Test the Setup

1. **Upload Test**: Use the app to pick and upload a profile picture
2. **Verify in Supabase**: Check Storage → avatars bucket → user folder
3. **Clear Cache Test**: Clear app cache on Android → Reopen app → Avatar should still appear

---

## File Structure in Storage

```
avatars/
  ├── {user_id_1}/
  │   └── avatar_1706889600000.jpg
  ├── {user_id_2}/
  │   └── avatar_1706889612345.png
  └── ...
```

Each user has their own folder, and old avatars are automatically cleaned up when a new one is uploaded.

---

## Public URL Format

After upload, the URL will look like:
```
https://[project-ref].supabase.co/storage/v1/object/public/avatars/[user-id]/avatar_[timestamp].[ext]
```

This URL:
- ✅ Never expires
- ✅ Works even after cache clear
- ✅ Works after app reinstall
- ✅ Works on any device

---

## Troubleshooting

### "new row violates row-level security policy"
- Ensure the storage policies are correctly set up (see Step 2)
- Verify the user is authenticated

### "Bucket 'avatars' not found"
- Create the bucket in Supabase Dashboard (Step 1)

### Image not showing after upload
- Check if the bucket is public
- Verify the URL is being saved to profiles.avatar_url
- Check browser network tab for CORS issues

---

## Environment Variables Required

Ensure these are set in your backend `.env`:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here
```

The service role key is NOT needed for storage uploads when using RLS policies correctly.

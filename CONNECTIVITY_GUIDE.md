# 🌐 Fix Network Error - "Failed to Login"

The "Network Error" on your mobile device happens because **Windows Firewall** blocks incoming connections from your phone.

## 🛠️ Step 1: Fix Firewall (Run as Admin)

1.  **Open a NEW PowerShell window as Administrator**:
    *   Press `Windows Key`
    *   Type `PowerShell`
    *   Most Important: **Right-click** > **Run as Administrator**

2.  **Run this command**:
    ```powershell
    cd c:\Users\mukie\Desktop\kuyenda\backend-mobile
    powershell -ExecutionPolicy Bypass -File .\fix_firewall.ps1
    ```

3.  **Success Message**:
    *   You should see: `✅ Firewall rule created successfully!` or `Port 8000 is now open`

## 📱 Step 2: Verify WiFi

1.  ensure your **Phone** and **Computer** are on the **Same WiFi Network**.
2.  Your Computer IP is confirmed as: `192.168.1.60`
    *   If your phone is on 4G/5G or a different WiFi (Guest network), it **will not work**.

## 🔄 Step 3: Restart App

1.  Close the app completely on your phone.
2.  Re-open it.
3.  Try to **Login** again.

---

### Still having issues?

If you fixed the firewall and still get "Network Error":
1.  **Disable Antivirus Firewall**: If you have McAfee, Norton, or Avast, their firewall might override Windows Firewall. Temporarily disable "Smart Firewall".
2.  **Test in Browser**: Open Chrome on your phone and go to:
    `http://192.168.1.60:8000`
    *   If you see `"message": "Kuyenda Mapbox Proxy Running"`, the connection is GOOD.
    *   If it keeps loading -> Firewall is still blocking.

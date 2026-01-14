Write-Host "Checking firewall rules for Kuyenda Backend (Port 8000)..." -ForegroundColor Cyan

$port = 8000
$ruleName = "Kuyenda Backend Python"

# Check if rule exists
$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue

if ($existing) {
    Write-Host "✅ Firewall rule '$ruleName' already exists." -ForegroundColor Green
    Write-Host "Ensuring it is enabled and allows traffic..."
    Set-NetFirewallRule -DisplayName $ruleName -Enabled True -Action Allow -Direction Inbound
    Write-Host "Rule updated." -ForegroundColor Green
} else {
    Write-Host "Creating new inbound firewall rule for port $port..." -ForegroundColor Yellow
    New-NetFirewallRule -DisplayName $ruleName `
                        -Direction Inbound `
                        -LocalPort $port `
                        -Protocol TCP `
                        -Action Allow `
                        -Profile Any `
                        -Enabled True
    Write-Host "✅ Firewall rule created successfully!" -ForegroundColor Green
}

Write-Host "`nVerifying rule..."
Get-NetFirewallRule -DisplayName $ruleName | Select-Object DisplayName, Action, Direction, Enabled, Profile

Write-Host "`n✅ Port 8000 is now open. Your mobile device should be able to connect." -ForegroundColor Green
Write-Host "Make sure your phone is connected to the SAME WiFi network as this computer: 192.168.1.x" -ForegroundColor Yellow

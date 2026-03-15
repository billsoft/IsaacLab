$zip  = "D:\Downloads\isaac_assets_tmp\isaac-sim-assets-complete-5.1.0.zip"
$dest = "D:\code\IsaacLab\rs"

Write-Host "Free space before: $([math]::Round((Get-PSDrive D).Free/1GB,1)) GB"
Write-Host "Extracting $zip -> $dest"
Write-Host "This may take 10-30 minutes..."

& "C:\Windows\System32\tar.exe" -xf $zip -C $dest

if ($LASTEXITCODE -eq 0) {
    Write-Host "Done! Contents:"
    Get-ChildItem $dest | Select-Object Name, LastWriteTime
    Write-Host "Free space after: $([math]::Round((Get-PSDrive D).Free/1GB,1)) GB"
} else {
    Write-Host "tar.exe failed with exit code $LASTEXITCODE"
}

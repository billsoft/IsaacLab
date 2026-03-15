$files = @(
    "D:\code\IsaacLab\apps\isaaclab.python.headless.kit",
    "D:\code\IsaacLab\apps\isaaclab.python.headless.rendering.kit",
    "D:\code\IsaacLab\apps\isaaclab.python.rendering.kit",
    "D:\code\IsaacLab\apps\isaaclab.python.xr.openxr.kit",
    "D:\code\IsaacLab\apps\isaaclab.python.xr.openxr.headless.kit",
    "D:\code\IsaacLab\apps\isaacsim_4_5\isaaclab.python.headless.kit",
    "D:\code\IsaacLab\apps\isaacsim_4_5\isaaclab.python.headless.rendering.kit",
    "D:\code\IsaacLab\apps\isaacsim_4_5\isaaclab.python.rendering.kit",
    "D:\code\IsaacLab\apps\isaacsim_4_5\isaaclab.python.xr.openxr.kit"
)

$localRoot = "D:/code/IsaacLab/Assets/Isaac/5.1"
$s3_51  = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
$s3_45  = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"

foreach ($f in $files) {
    if (-not (Test-Path $f)) {
        Write-Host "SKIP (not found): $f"
        continue
    }

    $content = Get-Content $f -Raw -Encoding UTF8

    # Check if already patched
    if ($content -match 'asset_root\.local') {
        Write-Host "ALREADY PATCHED: $f"
        continue
    }

    # Determine which S3 cloud URL this file uses
    $cloudUrl = if ($content -match [regex]::Escape($s3_45)) { $s3_45 } else { $s3_51 }

    # Replace default and nvidia to local; keep cloud as S3; add local key after nvidia line
    $content = $content `
        -replace "persistent\.isaac\.asset_root\.default = `"[^`"]+`"", "persistent.isaac.asset_root.default = `"$localRoot`"" `
        -replace "persistent\.isaac\.asset_root\.nvidia = `"[^`"]+`"",  "persistent.isaac.asset_root.nvidia = `"$localRoot`"`npersistent.isaac.asset_root.local = `"$localRoot`""

    Set-Content $f $content -Encoding UTF8 -NoNewline
    Write-Host "PATCHED: $f"
}

Write-Host "Done."

param(
    [string]$DownloadDir  = "D:\Downloads\isaac_assets_tmp",
    [string]$ExtractDir   = "D:\code\IsaacLab\rs",
    [switch]$SkipDownload,
    [switch]$SkipCombine,
    [switch]$CleanupAfter
)

$ErrorActionPreference = "Stop"

$BaseUrl     = "https://download.isaacsim.omniverse.nvidia.com"
$Parts       = @(
    "isaac-sim-assets-complete-5.1.0.zip.001",
    "isaac-sim-assets-complete-5.1.0.zip.002",
    "isaac-sim-assets-complete-5.1.0.zip.003"
)
$CombinedZip = "$DownloadDir\isaac-sim-assets-complete-5.1.0.zip"

function Log-Step($m) { Write-Host "`n==[ $m ]==" -ForegroundColor Cyan }
function Log-OK($m)   { Write-Host "  [OK] $m"   -ForegroundColor Green }
function Log-Info($m) { Write-Host "  [..] $m"   -ForegroundColor Yellow }
function Log-Err($m)  { Write-Host "  [!!] $m"   -ForegroundColor Red }

# ── Step 0: check disk space ──────────────────────────────────────────────────
Log-Step "Checking environment"

foreach ($dir in @($DownloadDir, $ExtractDir) | Sort-Object -Unique) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

$drives = @(
    (Split-Path -Qualifier $DownloadDir),
    (Split-Path -Qualifier $ExtractDir)
) | Sort-Object -Unique

foreach ($drv in $drives) {
    $letter = $drv -replace ':',''
    $freeGB = [math]::Round((Get-PSDrive $letter).Free / 1GB, 1)
    Log-Info "${drv} free space: ${freeGB} GB"
    if ($freeGB -lt 5) {
        Log-Err "Less than 5 GB free on ${drv}. Please free up space and retry."
        exit 1
    }
}
Log-OK "Directories ready"

# ── Step 1: download parts ────────────────────────────────────────────────────
if (-not $SkipDownload) {
    Log-Step "Downloading asset parts (approx 30 GB, resumable)"

    # prefer aria2c for speed; fall back to BITS (built-in)
    $aria2 = $null
    $cmd = Get-Command aria2c -ErrorAction SilentlyContinue
    if ($cmd) {
        $aria2 = $cmd.Source
    } else {
        $found = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" `
                     -Filter "aria2c.exe" -Recurse -ErrorAction SilentlyContinue |
                 Select-Object -First 1
        if ($found) { $aria2 = $found.FullName }
    }

    if ($aria2) {
        Log-OK "Using aria2c (multi-thread): $aria2"
    } else {
        Log-Info "aria2c not found; using Windows BITS (slower)."
        Log-Info "Tip: run  winget install --id=aria2.aria2 -e  then re-run for faster downloads."
    }

    foreach ($part in $Parts) {
        $dest = "$DownloadDir\$part"
        $url  = "$BaseUrl/$part"

        if (Test-Path $dest) {
            $mb = [math]::Round((Get-Item $dest).Length / 1MB, 0)
            Log-Info "$part already exists (${mb} MB) -- skipping"
            continue
        }

        Log-Info "Downloading $part ..."

        if ($aria2) {
            & $aria2 --dir="$DownloadDir" --out="$part" `
                --max-connection-per-server=16 --split=16 `
                --continue=true --file-allocation=none `
                "$url"
        } else {
            Start-BitsTransfer -Source $url -Destination $dest -DisplayName $part
        }

        if (-not (Test-Path $dest)) {
            Log-Err "$part download failed. Check your network and retry."
            exit 1
        }
        Log-OK "$part done"
    }
} else {
    Log-Info "-SkipDownload set, skipping download step"
}

# ── Step 2: combine parts ─────────────────────────────────────────────────────
if (-not $SkipCombine) {
    Log-Step "Combining parts -> $CombinedZip"

    if (Test-Path $CombinedZip) {
        Log-Info "Combined zip already exists, skipping combine"
    } else {
        foreach ($part in $Parts) {
            if (-not (Test-Path "$DownloadDir\$part")) {
                Log-Err "Missing part: $part"
                exit 1
            }
        }

        Log-Info "Concatenating (binary)..."
        $out = [System.IO.File]::Open($CombinedZip, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
        try {
            foreach ($part in $Parts) {
                Log-Info "  appending $part ..."
                $in = [System.IO.File]::OpenRead("$DownloadDir\$part")
                $in.CopyTo($out)
                $in.Close()
            }
        } finally {
            $out.Close()
        }

        $gb = [math]::Round((Get-Item $CombinedZip).Length / 1GB, 2)
        Log-OK "Combined: ${gb} GB"
    }
} else {
    Log-Info "-SkipCombine set, skipping combine step"
}

# ── Step 3: extract ───────────────────────────────────────────────────────────
Log-Step "Extracting to $ExtractDir"

$checkPath = "$ExtractDir\Assets\Isaac\5.1"
if (Test-Path $checkPath) {
    Log-Info "Assets already present at $checkPath -- skipping extract"
    Log-Info "(Delete $checkPath manually if you want to re-extract)"
} else {
    if (-not (Test-Path $CombinedZip)) {
        Log-Err "Combined zip not found: $CombinedZip"
        exit 1
    }

    Log-Info "Extracting (may take several minutes)..."
    tar -xf "$CombinedZip" -C "$ExtractDir"

    if ($LASTEXITCODE -ne 0) {
        Log-Info "tar failed, trying Expand-Archive..."
        Expand-Archive -Path $CombinedZip -DestinationPath $ExtractDir -Force
    }

    Log-OK "Extraction complete"
}

# ── Step 4: cleanup ───────────────────────────────────────────────────────────
if ($CleanupAfter) {
    Log-Step "Cleaning up temp files"
    Remove-Item -Recurse -Force $DownloadDir
    Log-OK "Removed $DownloadDir"
}

# ── Done ─────────────────────────────────────────────────────────────────────
$localRoot = "$ExtractDir\Assets\Isaac\5.1"
$localRootFwd = $localRoot -replace '\\','/'

Log-Step "All done!"
Write-Host ""
Write-Host "Assets at: $localRoot" -ForegroundColor Green
Write-Host ""
Write-Host "Configure Isaac Sim to use local assets (choose one):" -ForegroundColor White
Write-Host ""
Write-Host "  [Option A] Launch flag:" -ForegroundColor Yellow
Write-Host "  isaaclab.bat -s --/persistent/isaac/asset_root/default=`"$localRootFwd`"" -ForegroundColor Gray
Write-Host ""
Write-Host "  [Option B] Edit source (permanent):" -ForegroundColor Yellow
Write-Host "  File: source\isaaclab\isaaclab\utils\assets.py" -ForegroundColor Gray
Write-Host "  Set:  NUCLEUS_ASSET_ROOT_DIR = `"$localRootFwd`"" -ForegroundColor Gray
Write-Host ""
Write-Host "  [Verify] In Isaac Sim: Edit -> Preferences -> Isaac Sim" -ForegroundColor Yellow
Write-Host "           -> click 'Check Default Assets Root Path'" -ForegroundColor Gray
Write-Host ""

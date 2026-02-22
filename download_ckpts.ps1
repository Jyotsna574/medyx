# ==============================================================================
# MedSAM-2 Checkpoint Download Script (PowerShell)
# ==============================================================================
# Downloads required model weights for MedSAM-2 inference
#
# Usage:
#   .\download_ckpts.ps1
#
# Checkpoints downloaded:
#   1. SAM-2 base model (sam2.1_hiera_large.pt) - Meta's foundation model
#   2. MedSAM-2 fine-tuned weights - Medical imaging adaptation
# ==============================================================================

$ErrorActionPreference = "Stop"

# Configuration
$CHECKPOINTS_DIR = ".\checkpoints"
$SAM2_MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
$SAM2_MODEL_NAME = "sam2.1_hiera_large.pt"

# MedSAM-2 weights from Hugging Face
$MEDSAM2_REPO = "wanglab/MedSAM-2"
$MEDSAM2_WEIGHTS = "medsam2_checkpoint.pt"

Write-Host "=============================================="
Write-Host "MedSAM-2 Checkpoint Download Script"
Write-Host "=============================================="

# Create checkpoints directory
if (-not (Test-Path $CHECKPOINTS_DIR)) {
    New-Item -ItemType Directory -Path $CHECKPOINTS_DIR -Force | Out-Null
    Write-Host "Created checkpoints directory: $CHECKPOINTS_DIR"
}

# Function to download with progress
function Download-FileWithProgress {
    param (
        [string]$Url,
        [string]$OutputPath
    )
    
    Write-Host "Downloading: $(Split-Path $OutputPath -Leaf)"
    Write-Host "      URL: $Url"
    Write-Host "      This may take several minutes..."
    
    try {
        # Use Invoke-WebRequest with progress for large files
        $ProgressPreference = 'SilentlyContinue'  # Faster download
        Invoke-WebRequest -Uri $Url -OutFile $OutputPath -UseBasicParsing
        $ProgressPreference = 'Continue'
        
        $fileSize = (Get-Item $OutputPath).Length / 1MB
        Write-Host "      Download complete: $(Split-Path $OutputPath -Leaf) ($([math]::Round($fileSize, 1)) MB)" -ForegroundColor Green
    }
    catch {
        Write-Host "      Error downloading: $_" -ForegroundColor Red
        throw
    }
}

# Download SAM-2 base model
Write-Host ""
Write-Host "[1/2] Downloading SAM-2.1 Base Model (Hiera Large)..." -ForegroundColor Cyan
Write-Host "      This is Meta's Segment Anything 2.1 foundation model"
Write-Host "      Size: ~900MB"
Write-Host ""

$SAM2_PATH = Join-Path $CHECKPOINTS_DIR $SAM2_MODEL_NAME

if (Test-Path $SAM2_PATH) {
    Write-Host "      SAM-2 checkpoint already exists, skipping..." -ForegroundColor Yellow
} else {
    Download-FileWithProgress -Url $SAM2_MODEL_URL -OutputPath $SAM2_PATH
}

# Download MedSAM-2 weights via huggingface-hub
Write-Host ""
Write-Host "[2/2] Downloading MedSAM-2 Fine-tuned Weights..." -ForegroundColor Cyan
Write-Host "      Medical imaging adaptation layer"
Write-Host ""

$pythonScript = @"
import sys
try:
    from huggingface_hub import hf_hub_download
    import os
    
    checkpoint_dir = r'$CHECKPOINTS_DIR'
    medsam2_path = os.path.join(checkpoint_dir, '$MEDSAM2_WEIGHTS')
    
    if os.path.exists(medsam2_path):
        print('      MedSAM-2 checkpoint already exists, skipping...')
    else:
        try:
            downloaded_path = hf_hub_download(
                repo_id='$MEDSAM2_REPO',
                filename='$MEDSAM2_WEIGHTS',
                local_dir=checkpoint_dir,
                local_dir_use_symlinks=False
            )
            print(f'      Downloaded to: {downloaded_path}')
        except Exception as e:
            print(f'      Note: MedSAM-2 weights not available from HF ({e})')
            print('      Using SAM-2 base model - fine-tuning weights can be added later')
except ImportError:
    print('      Note: huggingface-hub not installed')
    print('      The SAM-2 base model will be used for inference')
"@

try {
    python -c $pythonScript
}
catch {
    Write-Host "      Note: Python/huggingface-hub not available, skipping MedSAM-2 specific weights" -ForegroundColor Yellow
    Write-Host "      The SAM-2 base model will be used for inference"
}

Write-Host ""
Write-Host "=============================================="
Write-Host "Checkpoint Download Complete!" -ForegroundColor Green
Write-Host "=============================================="
Write-Host ""
Write-Host "Checkpoints location: $CHECKPOINTS_DIR\"

if (Test-Path $CHECKPOINTS_DIR) {
    Get-ChildItem $CHECKPOINTS_DIR | Format-Table Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB, 2)}}
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Activate environment: conda activate medsam2"
Write-Host "  2. Run inference test: python -m tests.test_vision_engine"
Write-Host ""

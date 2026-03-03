# ==============================================================================
# MedSAM Checkpoint Download Script (PowerShell)
# ==============================================================================
# Downloads required model weights for MedSAM inference (bowang-lab/MedSAM)
#
# Usage:
#   .\download_ckpts.ps1
#
# Checkpoints downloaded:
#   medsam_vit_b.pth - MedSAM ViT-B fine-tuned on medical images (~380MB)
#
# Reference:
#   Ma et al., "Segment Anything in Medical Images", Nature Communications 2024
#   https://github.com/bowang-lab/MedSAM
# ==============================================================================

# Configuration
$CHECKPOINTS_DIR = if ($env:MEDSAM_CHECKPOINT_DIR) { $env:MEDSAM_CHECKPOINT_DIR } elseif ($env:MEDSAM2_CHECKPOINT_DIR) { $env:MEDSAM2_CHECKPOINT_DIR } else { ".\checkpoints" }
$MEDSAM_CHECKPOINT_URL = "https://huggingface.co/wanglab/medsam-vit-b/resolve/main/medsam_vit_b.pth"
$MEDSAM_CHECKPOINT_NAME = "medsam_vit_b.pth"

Write-Host "=============================================="
Write-Host "MedSAM Checkpoint Download Script"
Write-Host "=============================================="
Write-Host "Model: MedSAM (bowang-lab/MedSAM)"
Write-Host "Architecture: ViT-B"
Write-Host "Target: $CHECKPOINTS_DIR"
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
    
    try {
        $ProgressPreference = 'SilentlyContinue'  # Faster download
        Invoke-WebRequest -Uri $Url -OutFile $OutputPath -UseBasicParsing
        
        $fileSize = (Get-Item $OutputPath).Length / 1MB
        Write-Host "      Download complete: $(Split-Path $OutputPath -Leaf) ($([math]::Round($fileSize, 1)) MB)" -ForegroundColor Green
    }
    catch {
        Write-Host "      Error downloading: $_" -ForegroundColor Red
        throw
    }
}

# Download MedSAM checkpoint
Write-Host ""
Write-Host "[1/1] Downloading MedSAM ViT-B Checkpoint..." -ForegroundColor Cyan
Write-Host "      Fine-tuned on 1.5M+ medical image-mask pairs"
Write-Host "      Size: ~380MB"
Write-Host ""

$MEDSAM_PATH = Join-Path $CHECKPOINTS_DIR $MEDSAM_CHECKPOINT_NAME

if (Test-Path $MEDSAM_PATH) {
    Write-Host "      MedSAM checkpoint already exists, skipping download..." -ForegroundColor Yellow
    Write-Host "      Location: $MEDSAM_PATH"
} else {
    Download-FileWithProgress -Url $MEDSAM_CHECKPOINT_URL -OutputPath $MEDSAM_PATH
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
Write-Host "Installation (if not done):"
Write-Host "  git clone https://github.com/bowang-lab/MedSAM"
Write-Host "  cd MedSAM"
Write-Host "  pip install -e ."
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Activate environment: conda activate medsam"
Write-Host "  2. Run inference: python run_mas_diagnosis.py"
Write-Host ""

#!/usr/bin/env python
"""
MAS Diagnostic Pipeline - Multi-Agent Medical Diagnosis System.

Usage:
    python run_mas_diagnosis.py

Environment:
    ACTIVE_PROVIDER: gemini (default) | local
    GOOGLE_API_KEY: Required for gemini
    LOCAL_MODEL_PATH: Required for local (cluster)
    MEDSAM2_CHECKPOINT_DIR: MedSAM-2 checkpoints (default: ./checkpoints)
    HF_HOME: HuggingFace cache for Med42 (default: ~/.cache/huggingface)
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from core.schemas import PatientCase
from services.mas_orchestrator import MASOrchestrator


async def run_diagnosis():
    """Run MAS diagnosis on a chest X-ray."""
    
    image_path = "./test_chest_xray.png"
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return None
    
    case = PatientCase(
        id="MAS-DEMO-001",
        history="""58-year-old male with:
        - Persistent productive cough for 3 weeks
        - Mild dyspnea on exertion
        - Low-grade fever for 2 days
        
        PMH: Hypertension, Type 2 DM, former smoker (20 pack-years, quit 5y ago)
        Family: Father - lung cancer at 72
        Social: Construction worker""",
        image_path=image_path,
        patient_age=58,
        patient_sex="M",
        modality="X-Ray",
        target_region="Chest",
    )
    
    # Compute bounding box from image dimensions
    with Image.open(image_path) as img:
        w, h = img.size
    margin_x, margin_y = int(w * 0.1), int(h * 0.1)
    anatomical_bbox = [margin_x, margin_y, w - margin_x, h - margin_y]
    
    print(f"\n{'='*60}")
    print(f"MAS Diagnosis: {case.id}")
    print(f"Patient: {case.patient_age}yo {case.patient_sex} | {case.modality} - {case.target_region}")
    print(f"{'='*60}\n")
    
    checkpoint_dir = os.environ.get("MEDSAM2_CHECKPOINT_DIR", "./checkpoints")
    orchestrator = MASOrchestrator(
        checkpoint_path=checkpoint_dir,
        low_memory_mode=True,
        log_level="INFO",
        log_file="./mas_diagnosis.log",
    )
    
    try:
        report, _ = await orchestrator.run_diagnosis(
            case=case,
            anatomical_bbox=anatomical_bbox,
        )
        
        print(f"\n{'='*60}")
        print("DIAGNOSIS RESULT")
        print(f"{'='*60}")
        print(f"Primary: {report.primary_diagnosis}")
        print(f"Severity: {report.severity} | Confidence: {report.confidence:.0%}")
        print(f"Urgency: {report.urgency}")
        print(f"\nRecommendations:")
        for i, rec in enumerate(report.recommended_actions, 1):
            print(f"  {i}. {rec}")
        print(f"\nVision Metrics:")
        for k, v in report.vision_findings.extracted_geometry.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print(f"{'='*60}")
        print(f"Full log: ./mas_diagnosis.log")
        
        return report
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        orchestrator.cleanup()


if __name__ == "__main__":
    provider = os.getenv("ACTIVE_PROVIDER", "gemini")
    if provider not in ("gemini", "local"):
        print(f"ERROR: ACTIVE_PROVIDER must be gemini or local (got: {provider})")
        sys.exit(1)
    if provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)
    if provider == "local":
        print(f"Provider: local | Model: {os.getenv('LOCAL_MODEL_PATH', 'not set')}")
    else:
        print("Provider: gemini")

    result = asyncio.run(run_diagnosis())
    sys.exit(0 if result else 1)

#!/usr/bin/env python
"""
MAS Diagnostic Pipeline - Multi-Agent Medical Diagnosis System.

Usage:
    python run_mas_diagnosis.py

Environment Variables:
    ACTIVE_PROVIDER: gemini|openai|anthropic|local (default: gemini)
    GOOGLE_API_KEY: Required if ACTIVE_PROVIDER=gemini
    LOCAL_MODEL_PATH: Required if ACTIVE_PROVIDER=local
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
    
    orchestrator = MASOrchestrator(
        checkpoint_path="./checkpoints",
        low_memory_mode=True,
        log_level="INFO",
        log_file="./mas_diagnosis.log",
    )
    
    try:
        report, discussion = await orchestrator.run_diagnosis(
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
    
    if provider == "local":
        print(f"Provider: local | Model: {os.getenv('LOCAL_MODEL_PATH', 'not set')}")
    elif provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)
    elif provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    elif provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)
    else:
        print(f"Provider: {provider}")
    
    result = asyncio.run(run_diagnosis())
    sys.exit(0 if result else 1)

#!/usr/bin/env python
"""
Example script to run the MedSAM-2 diagnostic pipeline.

This demonstrates the full workflow:
1. MedSAM-2 vision analysis with geometric metric extraction
2. Knowledge Graph retrieval
3. CAMEL-AI multi-agent consultation
4. Final diagnostic report generation
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.schemas import PatientCase
from services.manager import DiagnosisManager


async def run_example():
    """Run an example diagnosis on a chest X-ray."""
    
    print("=" * 70)
    print("MedSAM-2 Medical Diagnostic Pipeline - Example Run")
    print("=" * 70)
    
    # Check if we have a test image
    image_path = "./test_chest_xray.png"
    if not os.path.exists(image_path):
        print(f"ERROR: Test image not found at {image_path}")
        return
    
    print(f"\nTest Image: {image_path}")
    print(f"Image Size: {os.path.getsize(image_path) / 1024:.1f} KB")
    
    # Create a patient case
    case = PatientCase(
        id="DEMO-CHEST-001",
        history="58-year-old male patient presenting with persistent cough for 3 weeks, "
                "mild dyspnea on exertion, and occasional chest discomfort. "
                "History of smoking (20 pack-years, quit 5 years ago). "
                "No fever or weight loss reported.",
        image_path=image_path,
        patient_age=58,
        patient_sex="M",
        modality="X-Ray",
        target_region="Chest",
    )
    
    print(f"\nPatient Case ID: {case.id}")
    print(f"Modality: {case.modality}")
    print(f"Target Region: {case.target_region}")
    print(f"Patient: {case.patient_age}yo {case.patient_sex}")
    print(f"\nHistory: {case.history[:100]}...")
    
    # Initialize the diagnosis manager
    # Try MedSAM-2 first, fall back to placeholder if not available
    print("\n" + "-" * 70)
    print("Initializing Diagnostic System...")
    print("-" * 70)
    
    try:
        manager = DiagnosisManager(
            use_medsam2=True,
            checkpoint_path="./checkpoints",
            low_memory_mode=True,
            preload_vision_model=False,
        )
    except Exception as e:
        print(f"MedSAM-2 init failed: {e}")
        print("Falling back to placeholder vision provider...")
        manager = DiagnosisManager(
            use_medsam2=False,
            low_memory_mode=True,
        )
    
    # Define anatomical bounding box for chest X-ray
    # This targets the central lung/mediastinum region
    # Format: [x_min, y_min, x_max, y_max]
    # We'll use a generous box covering most of the chest
    anatomical_bbox = [100, 100, 900, 800]  # Adjust based on image size
    
    print("\n" + "-" * 70)
    print("Running Diagnosis...")
    print("-" * 70)
    
    # Run the diagnosis
    try:
        report = await manager.run_diagnosis(
            case=case,
            anatomical_bbox=anatomical_bbox,
        )
        
        # Print the clinical summary
        print("\n" + "=" * 70)
        print("DIAGNOSTIC REPORT")
        print("=" * 70)
        print(report.to_clinical_summary())
        
        # Print additional details
        print("\n" + "-" * 70)
        print("RAW VISION METRICS")
        print("-" * 70)
        print(f"Model ID: {report.vision_findings.model_id}")
        print(f"Risk Score: {report.vision_findings.risk_score:.2%}")
        print(f"\nExtracted Geometry ({len(report.vision_findings.extracted_geometry)} measurements):")
        for key, value in report.vision_findings.extracted_geometry.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\nConfidence Scores:")
        for key, value in report.vision_findings.confidence_scores.items():
            print(f"  {key}: {value:.2%}")
        
        return report
        
    except Exception as e:
        print(f"\nERROR during diagnosis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY not set - CAMEL-AI agents will fail")
        print("Set it with: $env:GOOGLE_API_KEY = 'your-key-here'")
    
    # Run the async example
    result = asyncio.run(run_example())
    
    if result:
        print("\n" + "=" * 70)
        print("Example completed successfully!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("Example failed - check errors above")
        print("=" * 70)

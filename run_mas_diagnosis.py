#!/usr/bin/env python
"""
Example script to run the Multi-Agent System (MAS) diagnostic pipeline.

This demonstrates the full MAS workflow:
1. ClinicalHistoryAgent parses patient symptoms
2. VisionAnalysisAgent runs MedSAM-2 segmentation
3. KGAgent queries Neo4j knowledge graph
4. RadiologistAgent drafts preliminary report
5. SpecialistAgent reviews and debates for consensus
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.schemas import PatientCase
from services.mas_orchestrator import MASOrchestrator, run_mas_diagnosis


async def run_example():
    """Run an example MAS diagnosis on a chest X-ray."""
    
    image_path = "./test_chest_xray.png"
    if not os.path.exists(image_path):
        print(f"ERROR: Test image not found at {image_path}")
        print("Please ensure the test image exists before running.")
        return None
    
    case = PatientCase(
        id="MAS-DEMO-001",
        history="""58-year-old male patient presenting with:
        - Persistent productive cough for 3 weeks
        - Mild dyspnea on exertion, worsening over past week
        - Occasional right-sided chest discomfort, non-radiating
        - Low-grade fever (99.5°F) for 2 days
        
        Past Medical History:
        - Hypertension, controlled on lisinopril
        - Type 2 Diabetes Mellitus, on metformin
        - Former smoker (20 pack-years, quit 5 years ago)
        
        Social History:
        - Works in construction
        - No recent travel
        - No known TB exposure
        
        Family History:
        - Father: lung cancer at age 72
        - Mother: hypertension, diabetes
        """,
        image_path=image_path,
        patient_age=58,
        patient_sex="M",
        modality="X-Ray",
        target_region="Chest",
    )
    
    print("=" * 70)
    print("Multi-Agent System (MAS) Diagnostic Pipeline")
    print("=" * 70)
    print(f"\nPatient Case: {case.id}")
    print(f"Age/Sex: {case.patient_age}yo {case.patient_sex}")
    print(f"Modality: {case.modality} - {case.target_region}")
    print("\n" + "-" * 70)
    
    anatomical_bbox = [100, 100, 900, 800]
    
    orchestrator = MASOrchestrator(
        checkpoint_path="./checkpoints",
        low_memory_mode=True,
        log_level="INFO",
        log_file="./mas_diagnosis.log",
    )
    
    try:
        report, discussion_messages = await orchestrator.run_diagnosis(
            case=case,
            anatomical_bbox=anatomical_bbox,
        )
        
        print("\n" + "=" * 70)
        print("DIAGNOSTIC REPORT")
        print("=" * 70)
        print(report.to_clinical_summary())
        
        print("\n" + "=" * 70)
        print("AGENT DISCUSSION LOG")
        print("=" * 70)
        for msg in discussion_messages:
            print(f"\n[{msg.agent_role}] (Iteration {msg.iteration})")
            print("-" * 40)
            print(msg.content[:1000])
            if len(msg.content) > 1000:
                print("... (truncated)")
        
        print("\n" + "=" * 70)
        print(f"Log file written to: ./mas_diagnosis.log")
        print("=" * 70)
        
        return report
        
    except Exception as e:
        print(f"\nERROR during MAS diagnosis: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        orchestrator.cleanup()


async def run_simple_example():
    """Run a simplified example using the convenience function."""
    
    image_path = "./test_chest_xray.png"
    if not os.path.exists(image_path):
        print(f"ERROR: Test image not found at {image_path}")
        return None
    
    case = PatientCase(
        id="MAS-SIMPLE-001",
        history="45-year-old female with shortness of breath and chest pain for 1 week.",
        image_path=image_path,
        patient_age=45,
        patient_sex="F",
        modality="X-Ray",
        target_region="Chest",
    )
    
    report, messages = await run_mas_diagnosis(
        case=case,
        log_level="INFO",
    )
    
    print(report.to_clinical_summary())
    return report


if __name__ == "__main__":
    # Check for required credentials based on provider
    active_provider = os.getenv("ACTIVE_PROVIDER", "gemini")
    
    if active_provider == "local":
        # Local HuggingFace model - no API key needed
        print(f"Using local model provider")
        local_path = os.getenv("LOCAL_MODEL_PATH", "")
        if local_path:
            print(f"Model path: {local_path}")
    elif active_provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            print("ERROR: GOOGLE_API_KEY environment variable not set")
            print("Set it with: export GOOGLE_API_KEY='your-key-here'")
            print("Or use local model: ACTIVE_PROVIDER=local LOCAL_MODEL_PATH=/path/to/model")
            sys.exit(1)
    elif active_provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY environment variable not set")
            sys.exit(1)
    elif active_provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY environment variable not set")
            sys.exit(1)
    
    result = asyncio.run(run_example())
    
    if result:
        print("\nMAS diagnosis completed successfully!")
    else:
        print("\nMAS diagnosis failed - check errors above")

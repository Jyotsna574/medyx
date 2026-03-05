"""
Diagnosis Manager - Main orchestrator for the medical diagnosis workflow.

This module coordinates the complete diagnosis pipeline:
1. Vision AI analyzes the medical image (placeholder backend)
2. Knowledge Graph retrieves relevant medical guidelines
3. Multi-expert consultation discusses the findings
4. Final diagnostic report is generated
"""

import asyncio
import gc
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv

from core.schemas import DiagnosticReport, PatientCase, VisionMetrics
from infrastructure.vision import VisionProvider
# Neo4j disabled for testing - uncomment to enable
# from infrastructure.rag.neo4j_retriever import Neo4jKnowledgeRetriever
from services.squad import run_consultation, ConsultationResult

# Load environment variables
load_dotenv()


def _clear_cuda_memory():
    """Clear CUDA memory cache if available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    except ImportError:
        pass


class DiagnosisManager:
    """
    Main orchestrator for the medical diagnosis workflow.
    
    Coordinates between:
    - Vision AI (placeholder backend)
    - Knowledge Graph (medical guidelines from Neo4j)
    - CAMEL-AI Expert Agents (multi-agent consultation)
    """

    def __init__(self, preload_vision_model: bool = False):
        """
        Initialize the DiagnosisManager.

        Args:
            preload_vision_model: If True, load vision backend at startup.
        """
        print("Initializing Vision AI Provider...")
        self.vision_provider = VisionProvider(preload_model=preload_vision_model)
        print("  Engine: Placeholder (Development)")

        # Neo4j disabled for testing - uncomment to enable knowledge graph
        # print("Initializing Knowledge Graph...")
        # self.kg_retriever = Neo4jKnowledgeRetriever()
        # self.kg_retriever.connect()  # Raises Neo4jConnectionError on failure
        # print("Knowledge Graph: Connected")
        self.kg_retriever = None
        self.kg_available = False
        print("DiagnosisManager: Ready (Neo4j disabled)")

    async def run_diagnosis(
        self,
        case: PatientCase,
        prompt_type: str = "auto",
        prompt_data: Optional[Any] = None,
        anatomical_bbox: Optional[list[int]] = None,
    ) -> DiagnosticReport:
        """
        Execute the complete diagnosis workflow with vision analysis.

        Args:
            case: Patient case with history and image path.
            prompt_type: Type of segmentation prompt ('auto', 'bbox', 'point').
            prompt_data: Prompt coordinates for segmentation.
            anatomical_bbox: Predefined anatomical bounding box [x_min, y_min, x_max, y_max].
            
        Returns:
            Comprehensive DiagnosticReport with expert consultation.
        """
        print(f"\n{'='*60}")
        print(f"Starting diagnosis for case: {case.id}")
        print(f"{'='*60}")
        
        # Step 1: Run Vision Analysis (Neo4j disabled - no KG query)
        print("\n[Step 1] Analyzing image...")
        print(f"  Modality: {case.modality}")
        print(f"  Target Region: {case.target_region}")

        vision_task = self.vision_provider.analyze(case.image_path)
        
        # Neo4j disabled - use empty guidelines
        # kg_query = f"Find symptoms, treatments, and clinical guidelines for: {case.history}"
        # kg_task = self.kg_retriever.search(kg_query)
        # metrics, guidelines = await asyncio.gather(vision_task, kg_task)
        metrics = await vision_task
        guidelines = "[Neo4j disabled - using vision metrics and clinical reasoning only]"
        
        print(f"  Vision Analysis: Risk Score = {metrics.risk_score:.1%}")
        print(f"  Extracted Geometry: {len(metrics.extracted_geometry)} measurements")
        print(f"  Guidelines: {len(guidelines)} chars (Neo4j disabled)")
        
        # Log key geometric metrics
        if metrics.extracted_geometry:
            print("  Key Measurements:")
            for key, value in list(metrics.extracted_geometry.items())[:5]:
                print(f"    - {key}: {value}")
        
        
        # Step 2: Run Multi-Expert Consultation with extracted geometry
        print("\n[Step 2] Running expert consultation...")
        print("  Passing geometric metrics to CAMEL-AI agents...")
        consultation = run_consultation(metrics, guidelines)
        
        print(f"  Primary Diagnosis: {consultation.primary_diagnosis}")
        print(f"  Severity: {consultation.severity}")
        print(f"  Confidence: {consultation.confidence:.1%}")
        
        # Step 3: Generate Comprehensive Report
        print("\n[Step 3] Generating diagnostic report...")
        report = self._create_report(case, metrics, consultation)
        
        print(f"\n{'='*60}")
        print("Diagnosis Complete")
        print(f"{'='*60}")
        
        return report
    
    def _create_report(
        self,
        case: PatientCase,
        metrics: VisionMetrics,
        consultation: ConsultationResult
    ) -> DiagnosticReport:
        """
        Create the final diagnostic report.
        
        Args:
            case: Original patient case.
            metrics: Vision analysis results.
            consultation: Expert consultation results.
            
        Returns:
            Complete DiagnosticReport.
        """
        # Generate report ID
        report_id = f"DX-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        
        # Determine limitations based on analysis
        limitations = [
            "AI-assisted analysis should be confirmed by clinical examination",
            "Image quality may affect detection accuracy",
        ]
        
        if not self.kg_available:
            limitations.append("Knowledge graph unavailable - using standard guidelines")
        
        return DiagnosticReport(
            report_id=report_id,
            generated_at=datetime.now(),
            patient_case_id=case.id,
            vision_findings=metrics,
            primary_diagnosis=consultation.primary_diagnosis,
            differential_diagnoses=consultation.differential_diagnoses,
            severity=consultation.severity,
            confidence=consultation.confidence,
            urgency=consultation.urgency,
            recommended_actions=consultation.recommended_actions,
            follow_up_timeline=consultation.follow_up_timeline,
            expert_discussion=consultation.discussion_transcript,
            consensus_reached=consultation.consensus_reached,
            clinical_notes=consultation.clinical_notes or None,
            limitations=limitations,
        )

    def __del__(self):
        """Clean up connections."""
        if hasattr(self, 'kg_retriever') and self.kg_retriever:
            self.kg_retriever.close()

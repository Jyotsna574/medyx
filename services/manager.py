"""
Diagnosis Manager - Main orchestrator for the medical diagnosis workflow.

This module coordinates the complete diagnosis pipeline:
1. MedSAM-2 Vision AI analyzes the medical image with real segmentation
2. Knowledge Graph retrieves relevant medical guidelines  
3. Multi-expert consultation discusses the findings with extracted geometry
4. Final diagnostic report is generated with metrics-driven reasoning
"""

import asyncio
import gc
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv

from core.schemas import DiagnosticReport, PatientCase, VisionMetrics
from infrastructure.vision.vision_provider import VisionProvider
from infrastructure.vision.medsam2_engine import (
    MedSAM2VisionProvider,
    DomainConfig,
)
from infrastructure.rag.neo4j_retriever import Neo4jKnowledgeRetriever
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
    - MedSAM-2 Vision AI (real segmentation and geometric analysis)
    - Knowledge Graph (medical guidelines from Neo4j)
    - CAMEL-AI Expert Agents (multi-agent consultation)
    """

    def __init__(
        self,
        preload_vision_model: bool = False,
        use_medsam2: bool = True,
        checkpoint_path: Optional[str] = None,
        low_memory_mode: bool = True,
    ):
        """
        Initialize the DiagnosisManager.
        
        Args:
            preload_vision_model: If True, load vision model at startup.
            use_medsam2: If True, use MedSAM-2 engine. Otherwise, use placeholder.
            checkpoint_path: Path to MedSAM-2 checkpoints.
            low_memory_mode: Enable memory optimization for limited GPU VRAM.
        """
        self.use_medsam2 = use_medsam2
        self.low_memory_mode = low_memory_mode
        
        # Initialize Vision AI Provider
        print("Initializing Vision AI Provider...")
        if use_medsam2:
            try:
                self.vision_provider = MedSAM2VisionProvider(
                    checkpoint_path=checkpoint_path or "./checkpoints",
                    preload_model=preload_vision_model,
                    low_memory_mode=low_memory_mode,
                )
                print("  Engine: MedSAM-2 (Production)")
            except Exception as e:
                print(f"  MedSAM-2 initialization failed: {e}")
                print("  Falling back to placeholder vision provider")
                self.vision_provider = VisionProvider(preload_model=preload_vision_model)
                self.use_medsam2 = False
        else:
            self.vision_provider = VisionProvider(preload_model=preload_vision_model)
            print("  Engine: Placeholder (Development)")

        # Initialize Knowledge Graph Retriever
        print("Initializing Knowledge Graph...")
        self.kg_retriever = Neo4jKnowledgeRetriever()
        
        if self.kg_retriever.connect():
            print("Knowledge Graph: Connected")
            self.kg_available = True
        else:
            print("Knowledge Graph: Using fallback guidelines")
            self.kg_available = False
        
        print("DiagnosisManager: Ready")

    async def run_diagnosis(
        self,
        case: PatientCase,
        prompt_type: str = "auto",
        prompt_data: Optional[Any] = None,
        anatomical_bbox: Optional[list[int]] = None,
    ) -> DiagnosticReport:
        """
        Execute the complete diagnosis workflow with MedSAM-2 vision analysis.
        
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
        
        # Determine the prompt to use
        if anatomical_bbox is not None:
            prompt_type = "bbox"
            prompt_data = anatomical_bbox
        
        # Create domain configuration based on case
        domain_config = self._create_domain_config(case)
        
        # Step 1: Run Vision Analysis and KG Query in parallel
        print("\n[Step 1] Analyzing image and retrieving guidelines...")
        print(f"  Modality: {case.modality}")
        print(f"  Target Region: {case.target_region}")
        
        # Prepare vision analysis parameters
        if self.use_medsam2:
            vision_task = self.vision_provider.analyze(
                image_path=case.image_path,
                prompt_type=prompt_type,
                prompt_data=prompt_data,
                domain_config=domain_config,
            )
        else:
            vision_task = self.vision_provider.analyze(case.image_path)
        
        kg_query = f"Find symptoms, treatments, and clinical guidelines for: {case.history}"
        kg_task = self.kg_retriever.search(kg_query)
        
        metrics, guidelines = await asyncio.gather(vision_task, kg_task)
        
        print(f"  Vision Analysis: Risk Score = {metrics.risk_score:.1%}")
        print(f"  Extracted Geometry: {len(metrics.extracted_geometry)} measurements")
        print(f"  Guidelines Retrieved: {len(guidelines)} characters")
        
        # Log key geometric metrics
        if metrics.extracted_geometry:
            print("  Key Measurements:")
            for key, value in list(metrics.extracted_geometry.items())[:5]:
                print(f"    - {key}: {value}")
        
        # Clear CUDA memory after vision analysis before agent inference
        if self.low_memory_mode:
            print("\n  Clearing GPU memory for agent inference...")
            if self.use_medsam2:
                self.vision_provider.unload_model()
            _clear_cuda_memory()
        
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
    
    def _create_domain_config(self, case: PatientCase) -> Optional[DomainConfig]:
        """Create domain configuration based on patient case."""
        if not self.use_medsam2:
            return None
        
        # Map modality and region to domain configuration
        modality_lower = case.modality.lower()
        region_lower = case.target_region.lower()
        
        if "fundus" in modality_lower or "eye" in region_lower or "retina" in region_lower:
            return DomainConfig(
                domain="ophthalmic",
                modality="fundoscopy",
                target_structure="optic_disc",
                compute_ratios=["cdr", "circularity"],
            )
        elif "xray" in modality_lower or "x-ray" in modality_lower:
            if "chest" in region_lower or "lung" in region_lower or "thorax" in region_lower:
                return DomainConfig(
                    domain="thoracic",
                    modality="xray",
                    target_structure="lung_region",
                    compute_ratios=["aspect_ratio", "circularity"],
                )
        elif "ct" in modality_lower:
            if "abdomen" in region_lower or "liver" in region_lower:
                return DomainConfig(
                    domain="abdominal",
                    modality="ct",
                    target_structure="organ",
                    pixel_spacing_mm=0.7,
                    slice_thickness_mm=5.0,
                    compute_ratios=["circularity"],
                )
        elif "mri" in modality_lower:
            return DomainConfig(
                domain="neurological" if "brain" in region_lower else "general",
                modality="mri",
                target_structure=case.target_region,
                compute_ratios=["circularity", "aspect_ratio"],
            )
        
        # Default generic configuration
        return DomainConfig(
            domain="general",
            modality=case.modality,
            target_structure=case.target_region,
            compute_ratios=["circularity", "aspect_ratio"],
        )

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

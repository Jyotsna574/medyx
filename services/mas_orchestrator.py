"""
Multi-Agent System (MAS) Orchestrator for Medical Diagnosis.

This module implements a fully interactive MAS using the CAMEL-AI framework,
coordinating specialized agents that process scans and symptoms, retrieve
knowledge, and debate to reach clinical consensus.

Supports both cloud APIs (Gemini, OpenAI, Anthropic) and local HuggingFace
models via the LLM factory. The backend is selected based on configuration
(models.yaml) and environment variables.

Architecture:
    1. ClinicalHistoryAgent - Parses symptoms into structured clinical history
    2. VisionAnalysisAgent - Executes MedSAM-2 pipeline for geometric metrics
    3. KGAgent - Queries Neo4j for medical guidelines
    4. RadiologistAgent - Drafts preliminary diagnostic report
    5. SpecialistAgent - Reviews and provides final consensus opinion
"""

import asyncio
import os
import re
import sys
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field

from camel.agents import ChatAgent
from camel.messages import BaseMessage

from core.schemas import DiagnosticReport, PatientCase, VisionMetrics
from infrastructure.vision.medsam2_engine import (
    MedSAM2VisionProvider,
    DomainConfig,
)
# Neo4j disabled for testing - uncomment to enable
# from infrastructure.rag.neo4j_retriever import (
#     Neo4jKnowledgeRetriever,
#     Neo4jConnectionError,
#     Neo4jQueryError,
# )
from infrastructure.llm_factory import get_llm_backend, get_provider_info


# =============================================================================
# HELPERS
# =============================================================================

def _extract_field(text: str, pattern: str, default: str = "") -> str:
    """Extract first regex match from text."""
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else default


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def configure_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Configure loguru logging for the MAS pipeline."""
    logger.remove()
    
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=True,
    )
    
    if log_file:
        logger.add(
            log_file,
            format=log_format,
            level=log_level,
            rotation="10 MB",
        )
    
    return logger


# =============================================================================
# PYDANTIC SCHEMAS FOR AGENT OUTPUTS
# =============================================================================

class ClinicalHistory(BaseModel):
    """Structured clinical history extracted from patient symptoms."""
    
    age: Optional[int] = Field(None, description="Patient age in years")
    sex: Optional[str] = Field(None, description="Patient biological sex (M/F)")
    chief_complaint: str = Field(..., description="Primary reason for visit")
    history_present_illness: str = Field(..., description="Details of current condition")
    comorbidities: list[str] = Field(default_factory=list, description="Existing conditions")
    medications: list[str] = Field(default_factory=list, description="Current medications")
    risk_factors: list[str] = Field(default_factory=list, description="Relevant risk factors")
    family_history: list[str] = Field(default_factory=list, description="Relevant family history")
    social_history: Optional[str] = Field(None, description="Social/lifestyle factors")
    raw_text: str = Field(..., description="Original symptom text")


class GeometricMetrics(BaseModel):
    """Geometric metrics extracted from medical image segmentation."""
    
    pixel_area: int = Field(..., description="Segmented region area in pixels")
    bbox: list[int] = Field(..., description="Bounding box [x_min, y_min, x_max, y_max]")
    centroid: list[float] = Field(..., description="Center of mass [x, y]")
    circularity: float = Field(..., ge=0, le=1, description="Shape circularity (0-1)")
    eccentricity: float = Field(..., ge=0, le=1, description="Shape elongation (0-1)")
    solidity: float = Field(..., ge=0, le=1, description="Convex hull fill ratio")
    num_components: int = Field(..., description="Number of disconnected regions")
    confidence_score: float = Field(..., ge=0, le=1, description="Segmentation confidence")
    additional_metrics: dict[str, Any] = Field(default_factory=dict)
    model_id: str = Field(..., description="Vision model identifier")
    
    def to_clinical_text(self) -> str:
        """Convert metrics to clinical description text."""
        lines = [
            f"Segmentation Area: {self.pixel_area:,} pixels",
            f"Bounding Box: {self.bbox}",
            f"Centroid: ({self.centroid[0]:.1f}, {self.centroid[1]:.1f})",
            f"Shape Circularity: {self.circularity:.3f}",
            f"Eccentricity: {self.eccentricity:.3f}",
            f"Solidity: {self.solidity:.3f}",
            f"Components: {self.num_components}",
            f"Confidence: {self.confidence_score:.1%}",
        ]
        
        for key, value in self.additional_metrics.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)


class KnowledgeContext(BaseModel):
    """Medical knowledge retrieved from the knowledge graph."""
    
    diseases: list[str] = Field(default_factory=list, description="Relevant diseases")
    treatments: list[str] = Field(default_factory=list, description="Treatment options")
    risk_factors: list[str] = Field(default_factory=list, description="Risk factors")
    differential_diagnoses: list[str] = Field(default_factory=list)
    guidelines: str = Field(..., description="Clinical guidelines text")
    sources: list[str] = Field(default_factory=list, description="Knowledge sources")


class PreliminaryReport(BaseModel):
    """Preliminary diagnostic report from RadiologistAgent."""
    
    primary_impression: str = Field(..., description="Primary diagnostic impression")
    findings: list[str] = Field(default_factory=list, description="Key findings")
    severity: str = Field(..., description="NORMAL/MILD/MODERATE/SEVERE/CRITICAL")
    confidence: float = Field(..., ge=0, le=1, description="Diagnostic confidence")
    recommendations: list[str] = Field(default_factory=list)
    supporting_metrics: dict[str, Any] = Field(default_factory=dict)
    questions_for_specialist: list[str] = Field(default_factory=list)


class ConsensusStatus(str, Enum):
    """Status of the diagnostic consensus."""
    PENDING = "PENDING"
    IN_DISCUSSION = "IN_DISCUSSION"
    CONSENSUS_REACHED = "CONSENSUS_REACHED"
    DISAGREEMENT = "DISAGREEMENT"
    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"


class DiscussionMessage(BaseModel):
    """A single message in the Radiologist-Specialist discussion."""
    
    agent_role: str = Field(..., description="Role of the speaking agent")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    iteration: int = Field(..., description="Discussion iteration number")


class FinalConsensus(BaseModel):
    """Final consensus output from the MAS pipeline."""
    
    status: ConsensusStatus = Field(..., description="Consensus status")
    primary_diagnosis: str = Field(..., description="Final primary diagnosis")
    severity: str = Field(..., description="Final severity assessment")
    confidence: float = Field(..., ge=0, le=1, description="Final confidence")
    urgency: str = Field(..., description="ROUTINE/SOON/URGENT/EMERGENCY")
    differential_diagnoses: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    follow_up_timeline: str = Field(..., description="Follow-up recommendation")
    clinical_notes: str = Field(default="", description="Additional clinical notes")
    metrics_justification: str = Field(..., description="Metrics supporting diagnosis")
    discussion_summary: str = Field(..., description="Summary of agent discussion")
    iterations_used: int = Field(..., description="Discussion iterations used")


# =============================================================================
# CUSTOM TOOLS FOR AGENTS
# =============================================================================

class MedSAMTool:
    """Custom tool for executing MedSAM-2 vision pipeline."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        low_memory_mode: bool = True,
    ):
        self.checkpoint_path = checkpoint_path or os.environ.get("MEDSAM2_CHECKPOINT_DIR", "./checkpoints")
        self.low_memory_mode = low_memory_mode
        self._provider: Optional[MedSAM2VisionProvider] = None
    
    def _ensure_provider(self) -> MedSAM2VisionProvider:
        """Lazily initialize the vision provider."""
        if self._provider is None:
            logger.info("Initializing MedSAM-2 Vision Provider")
            self._provider = MedSAM2VisionProvider(
                checkpoint_path=self.checkpoint_path,
                preload_model=False,
                low_memory_mode=self.low_memory_mode,
            )
        return self._provider
    
    async def run_medsam_pipeline(
        self,
        image_path: str,
        anatomical_bbox: Optional[list[int]] = None,
        domain_config: Optional[DomainConfig] = None,
    ) -> GeometricMetrics:
        """
        Execute MedSAM-2 segmentation and extract geometric metrics.
        
        Args:
            image_path: Path to the medical image.
            anatomical_bbox: Optional bounding box [x_min, y_min, x_max, y_max].
            domain_config: Domain-specific configuration.
            
        Returns:
            GeometricMetrics with extracted measurements.
        """
        logger.info(f"Running MedSAM-2 pipeline on: {image_path}")
        
        provider = self._ensure_provider()
        
        prompt_type = "bbox" if anatomical_bbox else "auto"
        prompt_data = anatomical_bbox
        
        result = await provider.analyze(
            image_path=image_path,
            prompt_type=prompt_type,
            prompt_data=prompt_data,
            domain_config=domain_config,
        )
        
        # Check for analysis errors
        if not result.extracted_geometry and any("Analysis Error" in f for f in result.findings):
            logger.error(f"[VisionAnalysisAgent] Vision Analysis FAILED: {result.findings}")
        elif not result.extracted_geometry:
            logger.warning("[VisionAnalysisAgent] Vision Analysis returned NO geometry (possible empty mask)")
        
        logger.debug(f"Vision analysis complete: {len(result.extracted_geometry)} metrics")
        
        geometry = result.extracted_geometry
        
        metrics = GeometricMetrics(
            pixel_area=geometry.get("pixel_area", 0),
            bbox=geometry.get("bbox", [0, 0, 0, 0]),
            centroid=geometry.get("centroid", [0.0, 0.0]),
            circularity=geometry.get("circularity", 0.0),
            eccentricity=geometry.get("eccentricity", 0.0),
            solidity=geometry.get("solidity", 0.0),
            num_components=geometry.get("num_components", 1),
            confidence_score=result.confidence_scores.get("segmentation", 0.5),
            additional_metrics={
                k: v for k, v in geometry.items()
                if k not in ["pixel_area", "bbox", "centroid", "circularity", 
                            "eccentricity", "solidity", "num_components"]
            },
            model_id=result.model_id,
        )

        return metrics
    
    def unload(self):
        """Unload the vision model to free GPU memory."""
        if self._provider:
            self._provider.unload_model()
            logger.info("MedSAM-2 model unloaded")


# Neo4j disabled - entire class commented out. Uncomment to enable.
# class Neo4jTool:
#     """Custom tool for querying Neo4j knowledge graph."""
#
#     def __init__(self):
#         self._retriever: Optional[Neo4jKnowledgeRetriever] = None
#         self._connected = False
#
#     def _ensure_retriever(self) -> Neo4jKnowledgeRetriever:
#         """Lazily initialize the Neo4j retriever. Raises Neo4jConnectionError if connect fails."""
#         if self._retriever is None:
#             logger.info("Initializing Neo4j Knowledge Retriever")
#             self._retriever = Neo4jKnowledgeRetriever()
#             self._retriever.connect()  # Raises Neo4jConnectionError on failure
#             self._connected = True
#             logger.info("Neo4j connection established")
#         return self._retriever
#
#     async def query_medical_knowledge(
#         self,
#         clinical_history: ClinicalHistory,
#         geometric_metrics: GeometricMetrics,
#     ) -> KnowledgeContext:
#         retriever = self._ensure_retriever()
#         query_parts = [
#             clinical_history.chief_complaint,
#             " ".join(clinical_history.comorbidities),
#             " ".join(clinical_history.risk_factors),
#         ]
#         query = " ".join(query_parts)
#         logger.info(f"Querying knowledge graph: '{query[:100]}...'")
#         guidelines = await retriever.search(query)
#         logger.debug(f"Retrieved {len(guidelines)} characters of guidelines")
#         context = KnowledgeContext(
#             guidelines=guidelines,
#             sources=["Neo4j Medical KG"],
#         )
#         logger.info(f"Knowledge Context Retrieved: {len(context.sources)} sources")
#         return context
#
#     def close(self):
#         """Close the Neo4j connection."""
#         if self._retriever:
#             self._retriever.close()
#             logger.info("Neo4j connection closed")


# =============================================================================
# AGENT SYSTEM PROMPTS
# =============================================================================

CLINICAL_HISTORY_AGENT_PROMPT = """You are a Clinical History Analyst specializing in extracting structured medical information from patient narratives.

YOUR TASK:
Parse the raw patient symptoms text and extract a structured clinical history.

REQUIRED OUTPUT FORMAT (use EXACTLY these labels):
AGE: [number or "Unknown"]
SEX: [M/F/Unknown]
CHIEF_COMPLAINT: [main reason for visit in one sentence]
HISTORY_PRESENT_ILLNESS: [detailed description of current condition]
COMORBIDITIES: [comma-separated list of existing conditions]
MEDICATIONS: [comma-separated list of current medications]
RISK_FACTORS: [comma-separated list of relevant risk factors]
FAMILY_HISTORY: [comma-separated list of relevant family conditions]
SOCIAL_HISTORY: [lifestyle factors like smoking, alcohol, occupation]

RULES:
- Extract only what is explicitly stated or strongly implied
- Use "Unknown" or "None reported" when information is not available
- Be concise but complete
- Do not invent information not present in the input"""

VISION_ANALYSIS_AGENT_PROMPT = """You are a Medical Vision Analysis Agent responsible for interpreting geometric metrics from AI segmentation.

YOUR ROLE:
You receive geometric metrics extracted from a medical image by the MedSAM-2 segmentation model.
Interpret these metrics in clinical context.

METRICS YOU WILL RECEIVE:
- pixel_area: Size of segmented region
- bbox: Bounding box coordinates
- circularity: How circular the region is (0-1, where 1 = perfect circle)
- eccentricity: How elongated the shape is (0-1)
- solidity: How filled/solid the region is (0-1)
- num_components: Number of disconnected regions (higher = more fragmented)
- confidence_score: Model confidence in segmentation

YOUR OUTPUT MUST INCLUDE:
1. Quality assessment of the segmentation
2. Clinical interpretation of the shape metrics
3. Any concerns or anomalies detected
4. Recommendations for whether manual review is needed"""

KG_AGENT_PROMPT = """You are a Medical Knowledge Graph Agent with access to clinical databases and guidelines.

YOUR ROLE:
Given clinical history and imaging metrics, synthesize relevant medical knowledge.

YOUR OUTPUT MUST INCLUDE:
1. Relevant disease entities from the knowledge base
2. Risk factors applicable to this case
3. Potential differential diagnoses
4. Applicable clinical guidelines and thresholds
5. Treatment considerations if pathology is confirmed

IMPORTANT:
- Cite specific thresholds (e.g., "CDR > 0.7 indicates high glaucoma risk")
- Reference guideline sources when available
- Be evidence-based, not speculative"""

RADIOLOGIST_AGENT_PROMPT = """You are a Senior Radiologist with 25 years of experience drafting diagnostic reports.

YOUR ROLE:
Synthesize inputs from the Clinical History, Vision Analysis, and Knowledge Graph agents to draft a preliminary diagnostic report.

YOUR OUTPUT FORMAT:
PRIMARY_IMPRESSION: [Most likely diagnosis based on evidence]
KEY_FINDINGS:
- [Finding 1 with supporting metric]
- [Finding 2 with supporting metric]
SEVERITY: [NORMAL/MILD/MODERATE/SEVERE/CRITICAL]
CONFIDENCE: [0.XX]
RECOMMENDATIONS:
1. [Recommendation 1]
2. [Recommendation 2]
QUESTIONS_FOR_SPECIALIST:
- [Any uncertainties or areas needing senior review]

CRITICAL RULES:
1. ALWAYS cite specific numeric metrics to support findings
2. Do not make claims unsupported by the provided measurements
3. Acknowledge limitations and uncertainties
4. Flag any metrics that fall outside normal ranges"""

SPECIALIST_AGENT_PROMPT = """You are a Senior Specialist Diagnostician responsible for final clinical review.

YOUR ROLE:
Review the Radiologist's preliminary report for clinical accuracy and consistency with the raw metrics.

YOUR RESPONSIBILITIES:
1. Verify that all diagnostic claims are supported by the provided metrics
2. Check for logical inconsistencies
3. Assess if severity rating matches the measurements
4. Provide corrections or approval

YOUR OUTPUT FORMAT (choose one):

IF APPROVED:
CONSENSUS_REACHED
FINAL_DIAGNOSIS: [Diagnosis]
FINAL_SEVERITY: [Level]
FINAL_CONFIDENCE: [0.XX]
FINAL_URGENCY: [ROUTINE/SOON/URGENT/EMERGENCY]
CLINICAL_NOTES: [Any additional observations]
METRICS_JUSTIFICATION: [List specific measurements supporting this diagnosis]

IF NEEDS REVISION:
REVISION_REQUIRED
ISSUES:
- [Issue 1]
- [Issue 2]
QUESTIONS:
- [Question for Radiologist]
SUGGESTED_CORRECTIONS: [What should be changed]

IMPORTANT:
- You MUST eventually reach consensus or clearly state DISAGREEMENT
- After {max_iterations} iterations, you must provide a final assessment
- Use "CONSENSUS_REACHED" token when you approve the diagnosis"""


# =============================================================================
# MAS ORCHESTRATOR
# =============================================================================

class MASOrchestrator:
    """
    Multi-Agent System Orchestrator for Medical Diagnosis.
    
    Coordinates five specialized agents through a structured workflow
    with a bounded discussion loop for cross-verification.
    """
    
    TERMINATION_TOKEN = "CONSENSUS_REACHED"
    MAX_DISCUSSION_ITERATIONS = 3
    
    def __init__(
        self,
        checkpoint_path: str | None = None,
        low_memory_mode: bool = True,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
    ):
        """
        Initialize the MAS Orchestrator.
        
        Args:
            checkpoint_path: Path to MedSAM-2 checkpoints. Defaults to MEDSAM2_CHECKPOINT_DIR env.
            low_memory_mode: Enable GPU memory optimization.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
            log_file: Optional file path for logging output.
        """
        configure_logging(log_level, log_file)
        
        ckpt = checkpoint_path or os.environ.get("MEDSAM2_CHECKPOINT_DIR", "./checkpoints")
        
        logger.info("=" * 60)
        logger.info("Initializing Multi-Agent System Orchestrator")
        logger.info("=" * 60)
        
        self.medsam_tool = MedSAMTool(ckpt, low_memory_mode)
        # Neo4j disabled for testing - uncomment to enable knowledge graph
        # self.neo4j_tool = Neo4jTool()
        self.neo4j_tool = None  # No Neo4j - vision + CAMEL agents only
        self._init_agents()
        
        logger.info("MAS Orchestrator initialized successfully")
    
    def _init_agents(self):
        """Initialize all CAMEL-AI agents.
        
        Uses the LLM factory to get the appropriate backend based on
        configuration (models.yaml) and environment variables.
        """
        logger.info("Initializing CAMEL-AI agents...")
        
        # Get provider info for logging
        provider_info = get_provider_info()
        logger.info(f"LLM Provider: {provider_info['active_provider']}")
        if provider_info['is_local']:
            local_config = provider_info.get('local_model_config', {})
            logger.info(f"Local Model: {local_config.get('model_name', 'unknown')}")
        
        # Get the LLM backend from factory
        self.model = get_llm_backend()
        
        self.clinical_history_agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Clinical History Analyst",
                content=CLINICAL_HISTORY_AGENT_PROMPT,
            ),
            model=self.model,
        )
        logger.debug("ClinicalHistoryAgent initialized")
        
        self.vision_analysis_agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Vision Analysis Agent",
                content=VISION_ANALYSIS_AGENT_PROMPT,
            ),
            model=self.model,
        )
        logger.debug("VisionAnalysisAgent initialized")
        
        self.kg_agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Knowledge Graph Agent",
                content=KG_AGENT_PROMPT,
            ),
            model=self.model,
        )
        logger.debug("KGAgent initialized")
        
        self.radiologist_agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Senior Radiologist",
                content=RADIOLOGIST_AGENT_PROMPT,
            ),
            model=self.model,
        )
        logger.debug("RadiologistAgent initialized")
        
        specialist_prompt = SPECIALIST_AGENT_PROMPT.format(
            max_iterations=self.MAX_DISCUSSION_ITERATIONS
        )
        self.specialist_agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Senior Specialist",
                content=specialist_prompt,
            ),
            model=self.model,
        )
        logger.debug("SpecialistAgent initialized")
        
        logger.info("All 5 agents initialized successfully")
    
    async def run_diagnosis(
        self,
        case: PatientCase,
        anatomical_bbox: Optional[list[int]] = None,
    ) -> tuple[DiagnosticReport, list[DiscussionMessage]]:
        """
        Execute the full MAS diagnostic pipeline.
        
        Args:
            case: Patient case with symptoms and image path.
            anatomical_bbox: Optional bounding box for targeted segmentation.
            
        Returns:
            Tuple of (DiagnosticReport, discussion_messages).
        """
        logger.info("=" * 60)
        logger.info(f"Starting MAS Diagnosis for Case: {case.id}")
        logger.info("=" * 60)
        
        # Step 1: Parse Clinical History
        clinical_history = await self._run_clinical_history_agent(case)
        
        # Step 2: Run Vision Analysis
        geometric_metrics = await self._run_vision_analysis_agent(
            case.image_path, anatomical_bbox, case
        )

        # Clear GPU memory after vision analysis
        self.medsam_tool.unload()

        # Pre-compute metrics text once (used by KG, Radiologist, Specialist)
        metrics_text = geometric_metrics.to_clinical_text()

        # Step 3: Query Knowledge Graph
        knowledge_context = await self._run_kg_agent(
            clinical_history, geometric_metrics, metrics_text
        )

        # Step 4: Generate Preliminary Report
        preliminary_report = await self._run_radiologist_agent(
            clinical_history, geometric_metrics, knowledge_context, metrics_text
        )

        # Step 5: Discussion & Consensus Loop
        final_consensus, discussion_messages = await self._run_consensus_loop(
            preliminary_report, clinical_history, geometric_metrics, knowledge_context, metrics_text
        )
        
        # Step 6: Generate Final Report
        report = self._generate_final_report(
            case, geometric_metrics, final_consensus, discussion_messages
        )
        
        logger.info("=" * 60)
        logger.info("MAS Diagnosis Complete")
        logger.info(f"Final Diagnosis: {report.primary_diagnosis}")
        logger.info(f"Consensus Status: {final_consensus.status.value}")
        logger.info("=" * 60)
        
        return report, discussion_messages
    
    async def _run_clinical_history_agent(
        self, case: PatientCase
    ) -> ClinicalHistory:
        """Run the Clinical History Agent to parse symptoms."""
        logger.info("[Agent: ClinicalHistoryAgent] Parsing clinical history...")
        
        message = BaseMessage.make_user_message(
            role_name="Case Coordinator",
            content=f"""Parse the following patient information into structured clinical history:

PATIENT INFO:
Age: {case.patient_age or 'Not provided'}
Sex: {case.patient_sex or 'Not provided'}
Modality: {case.modality}
Target Region: {case.target_region}

SYMPTOMS/HISTORY:
{case.history}

Provide structured output using the required format.""",
        )
        
        response = self.clinical_history_agent.step(message)
        parsed = self._parse_clinical_history(response.msg.content, case)
        chief = parsed.chief_complaint or "N/A"
        logger.info(f"[ClinicalHistoryAgent] Chief: {chief[:80]}{'...' if len(chief) > 80 else ''}")

        return parsed
    
    def _parse_clinical_history(
        self, response: str, case: PatientCase
    ) -> ClinicalHistory:
        """Parse the agent response into ClinicalHistory schema."""

        def extract_list(pattern: str) -> list[str]:
            value = _extract_field(response, pattern, "")
            if value and value.lower() not in ["none", "unknown", "none reported"]:
                return [item.strip() for item in value.split(",") if item.strip()]
            return []

        age = case.patient_age
        age_str = _extract_field(response, r"AGE:\s*(\d+)")
        if age_str and age_str.isdigit():
            age = int(age_str)
        
        sex = case.patient_sex or _extract_field(response, r"SEX:\s*([MF])", "Unknown")

        return ClinicalHistory(
            age=age,
            sex=sex,
            chief_complaint=_extract_field(response, r"CHIEF_COMPLAINT:\s*(.+?)(?=\n[A-Z]|$)", case.history[:200]),
            history_present_illness=_extract_field(response, r"HISTORY_PRESENT_ILLNESS:\s*(.+?)(?=\n[A-Z]|$)", case.history),
            comorbidities=extract_list(r"COMORBIDITIES:\s*(.+?)(?=\n[A-Z]|$)"),
            medications=extract_list(r"MEDICATIONS:\s*(.+?)(?=\n[A-Z]|$)"),
            risk_factors=extract_list(r"RISK_FACTORS:\s*(.+?)(?=\n[A-Z]|$)"),
            family_history=extract_list(r"FAMILY_HISTORY:\s*(.+?)(?=\n[A-Z]|$)"),
            social_history=_extract_field(response, r"SOCIAL_HISTORY:\s*(.+?)(?=\n[A-Z]|$)"),
            raw_text=case.history,
        )
    
    async def _run_vision_analysis_agent(
        self,
        image_path: str,
        anatomical_bbox: Optional[list[int]],
        case: PatientCase,
    ) -> GeometricMetrics:
        """Run the Vision Analysis Agent with MedSAM-2 tool."""
        logger.info("[Agent: VisionAnalysisAgent] Running MedSAM-2 pipeline...")
        
        domain_config = self._create_domain_config(case)
        
        metrics = await self.medsam_tool.run_medsam_pipeline(
            image_path=image_path,
            anatomical_bbox=anatomical_bbox,
            domain_config=domain_config,
        )
        
        # Log metrics - warn if all zeros (indicates failed segmentation)
        if metrics.pixel_area == 0:
            logger.error(f"[VisionAnalysisAgent] FAILED - pixel_area=0, confidence={metrics.confidence_score:.1%}")
            logger.error("[VisionAnalysisAgent] Check: torchvision installed? MedSAM installed (MEDSAM_ROOT set)? Checkpoint exists? Image valid?")
        else:
            logger.info(f"[VisionAnalysisAgent] OK - area={metrics.pixel_area:,}px, circ={metrics.circularity:.3f}, conf={metrics.confidence_score:.1%}")
        
        return metrics
    
    def _create_domain_config(self, case: PatientCase) -> Optional[DomainConfig]:
        """Create domain configuration from patient case."""
        modality_lower = case.modality.lower()
        region_lower = case.target_region.lower()
        
        if "fundus" in modality_lower or "eye" in region_lower:
            return DomainConfig(
                domain="ophthalmic",
                modality="fundoscopy",
                target_structure="optic_disc",
                compute_ratios=["cdr", "circularity"],
            )
        elif "xray" in modality_lower or "x-ray" in modality_lower:
            return DomainConfig(
                domain="thoracic",
                modality="xray",
                target_structure="lung_region",
                compute_ratios=["aspect_ratio", "circularity"],
            )
        
        return DomainConfig(
            domain="general",
            modality=case.modality,
            target_structure=case.target_region,
            compute_ratios=["circularity", "aspect_ratio"],
        )
    
    async def _run_kg_agent(
        self,
        clinical_history: ClinicalHistory,
        geometric_metrics: GeometricMetrics,
        metrics_text: str,
    ) -> KnowledgeContext:
        """Run the Knowledge Graph Agent."""
        logger.info("[Agent: KGAgent] Querying knowledge graph...")
        knowledge_context = KnowledgeContext(
            guidelines="[Neo4j disabled - agents using clinical history and vision metrics only]",
            sources=[],
        )

        message = BaseMessage.make_user_message(
            role_name="Case Coordinator",
            content=f"""Given this clinical case, synthesize the relevant medical knowledge:

CLINICAL HISTORY:
- Chief Complaint: {clinical_history.chief_complaint}
- Comorbidities: {', '.join(clinical_history.comorbidities) or 'None'}
- Risk Factors: {', '.join(clinical_history.risk_factors) or 'None'}

IMAGING METRICS:
{metrics_text}

KNOWLEDGE BASE RESULTS:
{knowledge_context.guidelines}

Provide relevant disease associations, risk assessments, and applicable guidelines.""",
        )
        
        response = self.kg_agent.step(message)
        logger.info(f"[KGAgent] Guidelines: {len(knowledge_context.guidelines)} chars")

        return knowledge_context

    async def _run_radiologist_agent(
        self,
        clinical_history: ClinicalHistory,
        geometric_metrics: GeometricMetrics,
        knowledge_context: KnowledgeContext,
        metrics_text: str,
    ) -> str:
        """Run the Radiologist Agent to draft preliminary report."""
        logger.info("[Agent: RadiologistAgent] Drafting preliminary report...")

        message = BaseMessage.make_user_message(
            role_name="Case Coordinator",
            content=f"""Draft a preliminary diagnostic report based on the following inputs:

CLINICAL HISTORY:
- Age: {clinical_history.age or 'Unknown'}
- Sex: {clinical_history.sex or 'Unknown'}
- Chief Complaint: {clinical_history.chief_complaint}
- Comorbidities: {', '.join(clinical_history.comorbidities) or 'None reported'}
- Risk Factors: {', '.join(clinical_history.risk_factors) or 'None identified'}

GEOMETRIC METRICS FROM VISION ANALYSIS:
{metrics_text}

KNOWLEDGE GRAPH CONTEXT:
{knowledge_context.guidelines[:2000]}

Draft your preliminary report using the required format.
CRITICAL: All findings must cite specific metrics from the vision analysis.""",
        )
        
        response = self.radiologist_agent.step(message)
        preliminary_report = response.msg.content
        
        logger.info("[RadiologistAgent] Preliminary Report Generated")
        logger.debug(f"Report Preview: {preliminary_report[:500]}...")
        
        return preliminary_report
    
    async def _run_consensus_loop(
        self,
        preliminary_report: str,
        clinical_history: ClinicalHistory,
        geometric_metrics: GeometricMetrics,
        knowledge_context: KnowledgeContext,
        metrics_text: str,
    ) -> tuple[FinalConsensus, list[DiscussionMessage]]:
        """
        Run the bounded discussion loop between Radiologist and Specialist.
        Terminates on CONSENSUS_REACHED token or iteration limit.
        """
        logger.info(f"[Consensus] Starting loop (max {self.MAX_DISCUSSION_ITERATIONS} iterations)")

        discussion_messages: list[DiscussionMessage] = []
        current_report = preliminary_report
        iteration = 0
        consensus_reached = False
        specialist_response = ""

        while iteration < self.MAX_DISCUSSION_ITERATIONS and not consensus_reached:
            iteration += 1
            is_final = iteration == self.MAX_DISCUSSION_ITERATIONS

            specialist_message = BaseMessage.make_user_message(
                role_name="Case Coordinator",
                content=f"""Review this diagnostic report from the Radiologist:

RADIOLOGIST'S REPORT:
{current_report}

ORIGINAL METRICS FOR VERIFICATION:
{metrics_text}

ITERATION: {iteration} of {self.MAX_DISCUSSION_ITERATIONS}

{"This is the FINAL iteration. You MUST provide a final assessment with CONSENSUS_REACHED or DISAGREEMENT." if is_final else "Review the report and provide feedback or approval."}""",
            )
            
            specialist_resp = self.specialist_agent.step(specialist_message)
            specialist_response = specialist_resp.msg.content
            
            discussion_messages.append(DiscussionMessage(
                agent_role="Senior Specialist",
                content=specialist_response,
                iteration=iteration,
            ))
            
            if self.TERMINATION_TOKEN in specialist_response.upper():
                consensus_reached = True
                logger.info(f"[CONSENSUS REACHED] at iteration {iteration}")
                break
            
            if "DISAGREEMENT" in specialist_response.upper():
                logger.warning("[DISAGREEMENT] Specialist disagrees with diagnosis")
                break
            
            if iteration < self.MAX_DISCUSSION_ITERATIONS:
                radiologist_message = BaseMessage.make_user_message(
                    role_name="Case Coordinator",
                    content=f"""The Specialist has provided the following feedback:

SPECIALIST'S FEEDBACK:
{specialist_response}

ORIGINAL METRICS:
{metrics_text}

Respond to the Specialist's concerns and revise your report if needed.""",
                )
                
                radiologist_resp = self.radiologist_agent.step(radiologist_message)
                current_report = radiologist_resp.msg.content
                
                discussion_messages.append(DiscussionMessage(
                    agent_role="Senior Radiologist",
                    content=current_report,
                    iteration=iteration,
                ))
        
        if iteration >= self.MAX_DISCUSSION_ITERATIONS and not consensus_reached:
            logger.warning(f"[MAX ITERATIONS REACHED] Forcing final assessment")
        
        final_consensus = self._parse_consensus(
            specialist_response, 
            current_report, 
            iteration,
            consensus_reached,
        )
        
        logger.info(f"[Final Consensus] Status: {final_consensus.status.value}")
        logger.info(f"[Final Consensus] Diagnosis: {final_consensus.primary_diagnosis}")
        logger.info(f"[Final Consensus] Iterations Used: {final_consensus.iterations_used}")
        
        return final_consensus, discussion_messages
    
    def _parse_consensus(
        self,
        specialist_response: str,
        current_report: str,
        iterations: int,
        consensus_reached: bool,
    ) -> FinalConsensus:
        """Parse the final consensus from specialist response."""
        combined = specialist_response + "\n" + current_report

        status = ConsensusStatus.CONSENSUS_REACHED if consensus_reached else (
            ConsensusStatus.DISAGREEMENT if "DISAGREEMENT" in specialist_response.upper()
            else ConsensusStatus.MAX_ITERATIONS_REACHED
        )

        primary_diagnosis = _extract_field(
            combined,
            r"(?:FINAL_DIAGNOSIS|PRIMARY_IMPRESSION|PRIMARY_DIAGNOSIS):\s*(.+?)(?:\n|$)",
            "Requires further evaluation"
        )
        
        severity = _extract_field(
            combined,
            r"(?:FINAL_SEVERITY|SEVERITY):\s*(\w+)",
            "MODERATE"
        )
        if severity not in ["NORMAL", "MILD", "MODERATE", "SEVERE", "CRITICAL"]:
            severity = "MODERATE"
        
        confidence_str = _extract_field(combined, r"(?:FINAL_)?CONFIDENCE:\s*([\d.]+)", "0.7")
        try:
            confidence = float(confidence_str)
            confidence = min(1.0, max(0.0, confidence))
        except ValueError:
            confidence = 0.7
        
        urgency = _extract_field(
            combined,
            r"(?:FINAL_)?URGENCY:\s*(\w+)",
            "ROUTINE"
        )
        if urgency not in ["ROUTINE", "SOON", "URGENT", "EMERGENCY"]:
            urgency = "ROUTINE"
        
        metrics_justification = _extract_field(
            combined,
            r"METRICS_JUSTIFICATION:\s*(.+?)(?=\n[A-Z_]+:|$)",
            "See geometric metrics in report"
        )
        
        return FinalConsensus(
            status=status,
            primary_diagnosis=primary_diagnosis,
            severity=severity,
            confidence=confidence,
            urgency=urgency,
            differential_diagnoses=[],
            recommended_actions=[],
            follow_up_timeline="As clinically indicated",
            clinical_notes=_extract_field(combined, r"CLINICAL_NOTES:\s*(.+?)(?:\n|$)", ""),
            metrics_justification=metrics_justification,
            discussion_summary=f"Consensus reached after {iterations} iteration(s)",
            iterations_used=iterations,
        )
    
    def _generate_final_report(
        self,
        case: PatientCase,
        metrics: GeometricMetrics,
        consensus: FinalConsensus,
        discussion: list[DiscussionMessage],
    ) -> DiagnosticReport:
        """Generate the final DiagnosticReport from consensus."""
        report_id = f"MAS-{datetime.now().strftime('%Y%m%d')}-{uuid4().hex[:8].upper()}"
        
        vision_metrics = VisionMetrics(
            risk_score=min(1.0, max(0.0, 1.0 - metrics.confidence_score)),
            findings=[
                f"Segmentation confidence: {metrics.confidence_score:.1%}",
                f"Shape circularity: {metrics.circularity:.3f}",
                f"Region solidity: {metrics.solidity:.3f}",
                f"Components detected: {metrics.num_components}",
            ],
            extracted_geometry={
                "pixel_area": metrics.pixel_area,
                "bbox": metrics.bbox,
                "centroid": metrics.centroid,
                "circularity": metrics.circularity,
                "eccentricity": metrics.eccentricity,
                "solidity": metrics.solidity,
                "num_components": metrics.num_components,
                **metrics.additional_metrics,
            },
            confidence_scores={"segmentation": metrics.confidence_score},
            model_id=metrics.model_id,
        )
        
        discussion_transcript = "\n\n".join([
            f"[{msg.agent_role}] (Iteration {msg.iteration}):\n{msg.content}"
            for msg in discussion
        ])
        
        return DiagnosticReport(
            report_id=report_id,
            generated_at=datetime.now(),
            patient_case_id=case.id,
            vision_findings=vision_metrics,
            primary_diagnosis=consensus.primary_diagnosis,
            differential_diagnoses=consensus.differential_diagnoses,
            severity=consensus.severity,
            confidence=consensus.confidence,
            urgency=consensus.urgency,
            recommended_actions=consensus.recommended_actions or [
                "Review by attending physician",
                "Follow-up imaging as indicated",
            ],
            follow_up_timeline=consensus.follow_up_timeline,
            expert_discussion=discussion_transcript,
            consensus_reached=consensus.status == ConsensusStatus.CONSENSUS_REACHED,
            clinical_notes=consensus.clinical_notes or None,
            limitations=[
                "AI-assisted analysis requires clinical correlation",
                "Segmentation-based metrics may not capture all pathology",
                f"Analysis completed with {consensus.iterations_used} discussion iteration(s)",
            ],
        )
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up MAS resources...")
        self.medsam_tool.unload()
        # if self.neo4j_tool:
        #     self.neo4j_tool.close()
        logger.info("MAS cleanup complete")

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def run_mas_diagnosis(
    case: PatientCase,
    anatomical_bbox: Optional[list[int]] = None,
    checkpoint_path: str | None = None,
    low_memory_mode: bool = True,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> tuple[DiagnosticReport, list[DiscussionMessage]]:
    """
    Convenience function to run a full MAS diagnosis.

    Args:
        case: Patient case with symptoms and image.
        anatomical_bbox: Optional targeted bounding box.
        checkpoint_path: Path to MedSAM-2 checkpoints.
        low_memory_mode: Enable GPU memory optimization.
        log_level: Logging verbosity.
        log_file: Optional log file path.

    Returns:
        Tuple of (DiagnosticReport, discussion_messages).
    """
    ckpt = checkpoint_path or os.environ.get("MEDSAM2_CHECKPOINT_DIR", "./checkpoints")
    orchestrator = MASOrchestrator(
        checkpoint_path=ckpt,
        low_memory_mode=low_memory_mode,
        log_level=log_level,
        log_file=log_file,
    )
    
    try:
        return await orchestrator.run_diagnosis(case, anatomical_bbox)
    finally:
        orchestrator.cleanup()


"""
Multi-Agent Medical Consultation System.

This module implements a sophisticated multi-expert consultation using 
CAMEL-AI agents. Supports both cloud APIs (Gemini, OpenAI, Anthropic) and
local HuggingFace models via the LLM factory.

The backend is selected based on configuration (models.yaml) and
environment variables (ACTIVE_PROVIDER, LOCAL_ACTIVE_MODEL, LOCAL_MODEL_PATH).
"""

import re
from dataclasses import dataclass
from typing import Optional

from camel.agents import ChatAgent
from camel.messages import BaseMessage

from core.schemas import VisionMetrics
from infrastructure.llm_factory import get_llm_backend, get_provider_info


@dataclass
class ConsultationResult:
    """Result of the multi-agent medical consultation."""
    
    primary_diagnosis: str
    severity: str
    confidence: float
    urgency: str
    differential_diagnoses: list[str]
    recommended_actions: list[str]
    follow_up_timeline: str
    discussion_transcript: str
    consensus_reached: bool
    clinical_notes: str


# Expert System Prompts
RADIOLOGIST_PROMPT = """You are a Senior Radiologist with 25 years of experience in medical imaging interpretation.

YOUR EXPERTISE:
- Medical image interpretation across all modalities (X-ray, CT, MRI, Fundoscopy)
- Pattern recognition and pathology detection
- Quantitative analysis of segmentation-derived measurements
- Correlating imaging findings with clinical presentation

YOUR ROLE IN THIS CONSULTATION:
- Analyze the AI-extracted GEOMETRIC MEASUREMENTS from the segmentation
- Interpret specific metrics: pixel area, circularity, CDR (if ophthalmic), aspect ratios
- Explain the clinical significance of each measurement
- Assess whether measurements fall within normal or pathological ranges
- Identify any findings that may require additional imaging

CRITICAL: Base your analysis strictly on the PROVIDED NUMERICAL MEASUREMENTS.
Do not speculate about findings not supported by the extracted geometry.

COMMUNICATION STYLE:
- Be precise and evidence-based
- Reference specific numeric values from the measurements
- Clearly state confidence levels based on measurement quality
- Use standardized medical terminology"""


PULMONOLOGIST_PROMPT = """You are a Senior Clinical Specialist with 20 years of experience in diagnostic medicine.

YOUR EXPERTISE:
- Multi-organ system diagnostics across specialties
- Interpretation of quantitative imaging biomarkers
- Treatment planning based on objective measurements
- Understanding disease progression through measurable indicators

YOUR ROLE IN THIS CONSULTATION:
- Evaluate the clinical significance of the EXTRACTED GEOMETRIC METRICS
- Map specific measurements to clinical thresholds (e.g., CDR > 0.7 = glaucoma risk)
- Provide differential diagnoses SUPPORTED BY the numeric findings
- Recommend appropriate treatment or further workup based on measurements
- Assess urgency and severity using objective measurement criteria

CRITICAL: Your clinical interpretation must be anchored to the specific 
numeric values provided. Justify each clinical conclusion with measurements.

COMMUNICATION STYLE:
- Focus on measurement-to-outcome relationships
- Provide actionable recommendations tied to metrics
- Reference specific threshold values for pathology
- Quantify severity based on measured deviations from normal"""


MEDICAL_DIRECTOR_PROMPT = """You are the Medical Director overseeing this diagnostic consultation.

YOUR EXPERTISE:
- Clinical decision-making and quality assurance
- Evidence-based medicine and guideline compliance
- Risk assessment and patient safety
- Synthesizing expert opinions into actionable plans

YOUR ROLE IN THIS CONSULTATION:
- Review the analyses from the Radiologist and Pulmonologist
- Synthesize findings into a coherent diagnostic conclusion
- Ensure all relevant differential diagnoses are considered
- Formulate the final recommendations and follow-up plan
- Assign severity, urgency, and confidence ratings

CRITICAL REQUIREMENTS:
1. Your diagnosis MUST be grounded in the EXTRACTED GEOMETRIC MEASUREMENTS provided
2. Cite specific numeric values (e.g., pixel area, CDR, circularity) to justify your conclusions
3. Do NOT speculate beyond what the measurements support
4. If measurements are insufficient, explicitly state the limitations

You must provide your final assessment in the following structured format:

PRIMARY_DIAGNOSIS: [Single most likely diagnosis]
SEVERITY: [NORMAL/MILD/MODERATE/SEVERE/CRITICAL]
CONFIDENCE: [0.0-1.0 as decimal]
URGENCY: [ROUTINE/SOON/URGENT/EMERGENCY]
DIFFERENTIAL_DIAGNOSES: [Comma-separated list]
RECOMMENDED_ACTIONS: [Numbered list, one per line]
FOLLOW_UP: [Timeline recommendation]
CLINICAL_NOTES: [Any additional observations]
METRICS_JUSTIFICATION: [Cite specific measurements that support your diagnosis]
CONSENSUS: [YES/NO]"""


def run_consultation(metrics: VisionMetrics, guidelines: str) -> ConsultationResult:
    """
    Run a multi-expert medical consultation.
    
    Three experts discuss the case:
    1. Senior Radiologist - Analyzes imaging findings
    2. Senior Pulmonologist - Provides clinical interpretation
    3. Medical Director - Synthesizes and finalizes diagnosis
    
    The LLM backend is determined by configuration (models.yaml) and
    environment variables (ACTIVE_PROVIDER, LOCAL_ACTIVE_MODEL).
    
    Args:
        metrics: Vision metrics from AI image analysis.
        guidelines: Medical guidelines from knowledge base.
        
    Returns:
        ConsultationResult with diagnosis and full discussion.
    """
    # Get the LLM backend from factory (Gemini, OpenAI, local HuggingFace, etc.)
    provider_info = get_provider_info()
    print(f"[Squad] Using LLM provider: {provider_info['active_provider']}")
    
    model = get_llm_backend()

    # Format the vision findings
    findings_text = _format_findings(metrics)
    
    # Create expert agents
    radiologist = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Senior Radiologist",
            content=RADIOLOGIST_PROMPT,
        ),
        model=model,
    )
    
    pulmonologist = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Senior Pulmonologist",
            content=PULMONOLOGIST_PROMPT,
        ),
        model=model,
    )
    
    medical_director = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Medical Director",
            content=MEDICAL_DIRECTOR_PROMPT,
        ),
        model=model,
    )
    
    # Build the consultation transcript
    transcript = []
    transcript.append("=" * 70)
    transcript.append("MULTI-EXPERT MEDICAL CONSULTATION")
    transcript.append("=" * 70)
    transcript.append("\nEXPERT PANEL:")
    transcript.append("  1. Senior Radiologist - Imaging Specialist")
    transcript.append("  2. Senior Pulmonologist - Respiratory Medicine")
    transcript.append("  3. Medical Director - Final Review & Decision")
    transcript.append("\n" + "=" * 70)
    
    # === PHASE 1: Radiologist Analysis ===
    transcript.append("\n[PHASE 1] RADIOLOGICAL ANALYSIS")
    transcript.append("-" * 50)
    
    radiology_request = f"""
Please analyze the following AI-detected imaging findings:

AUTOMATED VISION ANALYSIS:
{findings_text}

REFERENCE GUIDELINES:
{guidelines[:2000]}  # Truncate if too long

Provide your expert interpretation of these findings, including:
1. Assessment of each detected pathology
2. Clinical significance
3. Reliability of the automated findings
4. Any additional imaging recommendations
"""
    
    rad_message = BaseMessage.make_user_message(
        role_name="Case Coordinator",
        content=radiology_request,
    )
    rad_response = radiologist.step(rad_message)
    radiology_analysis = rad_response.msg.content
    
    transcript.append("\nSENIOR RADIOLOGIST:")
    transcript.append(radiology_analysis)
    
    # === PHASE 2: Pulmonologist Clinical Interpretation ===
    transcript.append("\n" + "-" * 50)
    transcript.append("[PHASE 2] CLINICAL INTERPRETATION")
    transcript.append("-" * 50)
    
    pulm_request = f"""
Based on the radiologist's analysis, please provide your clinical interpretation:

ORIGINAL FINDINGS:
{findings_text}

RADIOLOGIST'S ASSESSMENT:
{radiology_analysis}

Please provide:
1. Clinical significance of these findings
2. Differential diagnoses to consider
3. Recommended treatment approach
4. Urgency assessment
"""
    
    pulm_message = BaseMessage.make_user_message(
        role_name="Case Coordinator",
        content=pulm_request,
    )
    pulm_response = pulmonologist.step(pulm_message)
    clinical_interpretation = pulm_response.msg.content
    
    transcript.append("\nSENIOR PULMONOLOGIST:")
    transcript.append(clinical_interpretation)
    
    # === PHASE 3: Medical Director Final Review ===
    transcript.append("\n" + "-" * 50)
    transcript.append("[PHASE 3] FINAL DIAGNOSTIC REVIEW")
    transcript.append("-" * 50)
    
    director_request = f"""
Please synthesize the expert analyses and provide the final diagnostic conclusion.

CRITICAL: Your diagnosis MUST be strictly grounded in the EXTRACTED GEOMETRIC MEASUREMENTS.
Cite specific numeric values to justify every clinical conclusion.

ORIGINAL VISION FINDINGS WITH EXTRACTED METRICS:
{findings_text}

RADIOLOGIST'S ANALYSIS:
{radiology_analysis}

PULMONOLOGIST'S INTERPRETATION:
{clinical_interpretation}

Provide your final structured assessment using EXACTLY this format:

PRIMARY_DIAGNOSIS: [diagnosis]
SEVERITY: [NORMAL/MILD/MODERATE/SEVERE/CRITICAL]
CONFIDENCE: [0.XX]
URGENCY: [ROUTINE/SOON/URGENT/EMERGENCY]
DIFFERENTIAL_DIAGNOSES: [diagnosis1, diagnosis2, ...]
RECOMMENDED_ACTIONS:
1. [action 1]
2. [action 2]
3. [action 3]
FOLLOW_UP: [timeline]
CLINICAL_NOTES: [observations]
METRICS_JUSTIFICATION: [List the specific measurements (e.g., pixel_area=X, cdr=Y, circularity=Z) that support your diagnosis]
CONSENSUS: [YES/NO]
"""
    
    dir_message = BaseMessage.make_user_message(
        role_name="Case Coordinator",
        content=director_request,
    )
    dir_response = medical_director.step(dir_message)
    final_assessment = dir_response.msg.content
    
    transcript.append("\nMEDICAL DIRECTOR (Final Assessment):")
    transcript.append(final_assessment)
    
    # Closing
    transcript.append("\n" + "=" * 70)
    transcript.append("CONSULTATION COMPLETE")
    transcript.append("=" * 70)
    
    # Parse the structured response
    result = _parse_director_response(final_assessment, metrics)
    result.discussion_transcript = "\n".join(transcript)
    
    return result


def _format_findings(metrics: VisionMetrics) -> str:
    """Format vision metrics for the experts with extracted geometry."""
    lines = [
        f"Overall Risk Score: {metrics.risk_score:.1%}",
        f"Model Used: {metrics.model_id}",
        "",
    ]
    
    # Format extracted geometric measurements
    if metrics.extracted_geometry:
        lines.append("EXTRACTED GEOMETRIC MEASUREMENTS:")
        lines.append("-" * 40)
        for key, value in metrics.extracted_geometry.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")
    
    # Format confidence scores
    if metrics.confidence_scores:
        lines.append("CONFIDENCE SCORES:")
        for key, score in metrics.confidence_scores.items():
            lines.append(f"  {key}: {score:.1%}")
        lines.append("")
    
    # Format detected findings
    lines.append("DETECTED FINDINGS:")
    for finding in metrics.findings:
        lines.append(f"  - {finding}")
    
    return "\n".join(lines)


def _parse_director_response(response: str, metrics: VisionMetrics) -> ConsultationResult:
    """Parse the Medical Director's structured response."""
    
    def extract_field(pattern: str, default: str = "") -> str:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else default
    
    def extract_list(pattern: str) -> list[str]:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            items = match.group(1).strip()
            # Split by commas or numbered lines
            if "\n" in items:
                return [re.sub(r'^\d+\.\s*', '', line.strip()) 
                        for line in items.split("\n") if line.strip()]
            else:
                return [item.strip() for item in items.split(",") if item.strip()]
        return []
    
    # Extract fields with defaults based on risk score
    primary_diagnosis = extract_field(r"PRIMARY_DIAGNOSIS:\s*(.+?)(?:\n|$)", "Findings require further evaluation")
    
    severity = extract_field(r"SEVERITY:\s*(\w+)", "MODERATE")
    if severity not in ["NORMAL", "MILD", "MODERATE", "SEVERE", "CRITICAL"]:
        severity = "MODERATE" if metrics.risk_score > 0.3 else "MILD"
    
    confidence_str = extract_field(r"CONFIDENCE:\s*([\d.]+)", "0.7")
    try:
        confidence = float(confidence_str)
        confidence = min(1.0, max(0.0, confidence))
    except ValueError:
        confidence = 0.7
    
    urgency = extract_field(r"URGENCY:\s*(\w+)", "ROUTINE")
    if urgency not in ["ROUTINE", "SOON", "URGENT", "EMERGENCY"]:
        urgency = "URGENT" if metrics.risk_score > 0.5 else "ROUTINE"
    
    differential = extract_list(r"DIFFERENTIAL_DIAGNOSES:\s*(.+?)(?=RECOMMENDED|FOLLOW|CLINICAL|CONSENSUS|$)")
    
    # Extract recommended actions (handle numbered list)
    actions_match = re.search(
        r"RECOMMENDED_ACTIONS:\s*\n?((?:\d+\..+\n?)+)",
        response,
        re.IGNORECASE
    )
    if actions_match:
        actions = [re.sub(r'^\d+\.\s*', '', line.strip()) 
                   for line in actions_match.group(1).split("\n") if line.strip()]
    else:
        actions = ["Schedule follow-up consultation", "Review imaging findings"]
    
    follow_up = extract_field(r"FOLLOW_UP:\s*(.+?)(?:\n|$)", "As clinically indicated")
    clinical_notes = extract_field(r"CLINICAL_NOTES:\s*(.+?)(?=CONSENSUS|$)", "")
    
    consensus_str = extract_field(r"CONSENSUS:\s*(\w+)", "YES")
    consensus = consensus_str.upper() == "YES"
    
    return ConsultationResult(
        primary_diagnosis=primary_diagnosis,
        severity=severity,
        confidence=confidence,
        urgency=urgency,
        differential_diagnoses=differential,
        recommended_actions=actions,
        follow_up_timeline=follow_up,
        discussion_transcript="",  # Will be set by caller
        consensus_reached=consensus,
        clinical_notes=clinical_notes,
    )

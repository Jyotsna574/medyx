"""Pydantic Models - The Data Contract for MedicalAgentDiagnosis-MAD."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class PatientCase(BaseModel):
    """Represents a patient case for diagnosis."""

    id: str = Field(..., description="Unique identifier for the patient case")
    history: str = Field(..., description="Patient medical history and symptoms")
    image_path: str = Field(..., description="Path to the medical scan image")
    patient_age: Optional[int] = Field(None, description="Patient age in years")
    patient_sex: Optional[str] = Field(None, description="Patient sex (M/F)")
    modality: str = Field(
        default="unknown",
        description="Imaging modality (e.g., 'X-Ray', 'CT', 'MRI', 'Ultrasound', 'Fundoscopy')"
    )
    target_region: str = Field(
        default="unknown",
        description="Anatomical target region (e.g., 'Chest', 'Abdomen', 'Brain', 'Eye')"
    )


class VisionMetrics(BaseModel):
    """Metrics extracted from AI vision analysis of medical scans.
    
    This schema is domain-agnostic and supports any imaging modality.
    Specific geometric measurements are stored in extracted_geometry.
    """

    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall pathology risk score (0.0 = low, 1.0 = high)",
    )
    findings: list[str] = Field(
        default_factory=list,
        description="List of detected pathologies and clinical findings",
    )
    extracted_geometry: dict = Field(
        default_factory=dict,
        description="Dynamic geometric measurements extracted from the image. "
                    "Examples: {'optic_cup_area_px': 1234, 'cdr': 0.65}, "
                    "{'liver_volume_cm3': 1450.5}, {'lesion_diameter_mm': 12.3}",
    )
    confidence_scores: dict = Field(
        default_factory=dict,
        description="Per-finding confidence scores. Keys are finding names, values are 0.0-1.0.",
    )
    model_id: str = Field(
        default="unknown",
        description="Identifier of the vision model that produced these metrics",
    )


class ExpertOpinion(BaseModel):
    """Opinion from a single medical expert in the consultation."""
    
    expert_role: str = Field(..., description="Role of the expert (e.g., 'Senior Radiologist')")
    assessment: str = Field(..., description="The expert's detailed assessment")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in assessment")
    recommendations: list[str] = Field(default_factory=list, description="Specific recommendations")


class DiagnosticReport(BaseModel):
    """Comprehensive diagnostic report with expert consultation."""

    # Report metadata
    report_id: str = Field(..., description="Unique report identifier")
    generated_at: datetime = Field(default_factory=datetime.now, description="Report generation timestamp")
    
    # Patient info
    patient_case_id: str = Field(..., description="Reference to the patient case")
    
    # Vision analysis
    vision_findings: VisionMetrics = Field(..., description="AI vision analysis results")
    
    # Expert consultation
    primary_diagnosis: str = Field(..., description="Primary diagnosis conclusion")
    differential_diagnoses: list[str] = Field(default_factory=list, description="Alternative diagnoses considered")
    
    # Risk assessment
    severity: str = Field(..., description="Severity level: NORMAL, MILD, MODERATE, SEVERE, CRITICAL")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall diagnostic confidence")
    urgency: str = Field(..., description="Urgency: ROUTINE, SOON, URGENT, EMERGENCY")
    
    # Recommendations
    recommended_actions: list[str] = Field(default_factory=list, description="Recommended next steps")
    follow_up_timeline: Optional[str] = Field(None, description="Suggested follow-up timeline")
    
    # Expert discussion
    expert_discussion: str = Field(..., description="Full transcript of expert consultation")
    consensus_reached: bool = Field(True, description="Whether experts reached consensus")
    
    # Additional notes
    clinical_notes: Optional[str] = Field(None, description="Additional clinical observations")
    limitations: list[str] = Field(default_factory=list, description="Limitations of this analysis")

    def to_clinical_summary(self) -> str:
        """Generate a clean clinical summary for display."""
        lines = [
            "=" * 70,
            "DIAGNOSTIC REPORT",
            "=" * 70,
            f"Report ID: {self.report_id}",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Patient Case: {self.patient_case_id}",
            "",
            "-" * 70,
            "VISION ANALYSIS FINDINGS",
            "-" * 70,
            f"Risk Score: {self.vision_findings.risk_score:.1%}",
            f"Model: {self.vision_findings.model_id}",
            "",
        ]
        
        if self.vision_findings.extracted_geometry:
            lines.append("Extracted Measurements:")
            for key, value in self.vision_findings.extracted_geometry.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        lines.append("Detected Findings:")
        
        for finding in self.vision_findings.findings:
            lines.append(f"  * {finding}")
        
        lines.extend([
            "",
            "-" * 70,
            "DIAGNOSIS",
            "-" * 70,
            f"PRIMARY: {self.primary_diagnosis}",
            "",
            f"Severity: {self.severity}",
            f"Confidence: {self.confidence:.1%}",
            f"Urgency: {self.urgency}",
        ])
        
        if self.differential_diagnoses:
            lines.append("")
            lines.append("Differential Diagnoses:")
            for dd in self.differential_diagnoses:
                lines.append(f"  - {dd}")
        
        lines.extend([
            "",
            "-" * 70,
            "RECOMMENDATIONS",
            "-" * 70,
        ])
        
        for i, action in enumerate(self.recommended_actions, 1):
            lines.append(f"  {i}. {action}")
        
        if self.follow_up_timeline:
            lines.append(f"\nFollow-up: {self.follow_up_timeline}")
        
        if self.limitations:
            lines.extend([
                "",
                "-" * 70,
                "LIMITATIONS",
                "-" * 70,
            ])
            for lim in self.limitations:
                lines.append(f"  * {lim}")
        
        lines.extend([
            "",
            "=" * 70,
            f"Consensus Reached: {'Yes' if self.consensus_reached else 'No - Review Required'}",
            "=" * 70,
        ])
        
        return "\n".join(lines)

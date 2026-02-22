"""Services module containing business logic for MedicalAgentDiagnosis-MAD."""

from .squad import run_consultation
from .manager import DiagnosisManager
from .mas_orchestrator import (
    MASOrchestrator,
    run_mas_diagnosis,
    ClinicalHistory,
    GeometricMetrics,
    KnowledgeContext,
    FinalConsensus,
    DiscussionMessage,
    ConsensusStatus,
)

__all__ = [
    # Original exports
    "run_consultation",
    "DiagnosisManager",
    # MAS Orchestrator
    "MASOrchestrator",
    "run_mas_diagnosis",
    # MAS Schemas
    "ClinicalHistory",
    "GeometricMetrics",
    "KnowledgeContext",
    "FinalConsensus",
    "DiscussionMessage",
    "ConsensusStatus",
]

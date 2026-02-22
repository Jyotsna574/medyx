"""Core module containing schemas and interfaces for MedicalAgentDiagnosis-MAD."""

from .schemas import DiagnosticReport, PatientCase, VisionMetrics, ExpertOpinion
from .interfaces import AbstractKnowledgeRetriever, AbstractVisionProvider

__all__ = [
    "PatientCase",
    "VisionMetrics",
    "DiagnosticReport",
    "ExpertOpinion",
    "AbstractVisionProvider",
    "AbstractKnowledgeRetriever",
]

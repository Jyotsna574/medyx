"""Vision Analysis Module.

Provides AI-powered medical image analysis through configurable backends.

Components:
- VisionProvider: Main provider for image analysis (placeholder)
- MedSAM2VisionProvider: Production MedSAM-2 segmentation provider
- MedSAM2VisionEngine: Low-level MedSAM-2 inference engine
- VisionBackend: Abstract base class for implementing custom backends
- VisionAnalysisResult: Standard result format from any backend
- DomainConfig: Domain-specific configuration for metric extraction
"""

from .vision_provider import (
    VisionProvider,
    VisionBackend,
    VisionAnalysisResult,
    PlaceholderVisionBackend,
)

from .medsam2_engine import (
    MedSAM2VisionEngine,
    MedSAM2VisionProvider,
    DomainConfig,
    PromptType,
    SegmentationPrompt,
    SegmentationResult,
)

__all__ = [
    # Base classes
    "VisionProvider",
    "VisionBackend",
    "VisionAnalysisResult",
    "PlaceholderVisionBackend",
    # MedSAM-2 Production
    "MedSAM2VisionEngine",
    "MedSAM2VisionProvider",
    "DomainConfig",
    "PromptType",
    "SegmentationPrompt",
    "SegmentationResult",
]

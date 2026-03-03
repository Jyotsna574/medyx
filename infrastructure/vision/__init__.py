"""Vision Analysis Module.

Provides AI-powered medical image analysis through configurable backends.

Components:
- VisionProvider: Main provider for image analysis (placeholder)
- MedSAMVisionProvider: Production MedSAM segmentation provider (bowang-lab)
- MedSAMVisionEngine: Low-level MedSAM inference engine
- VisionBackend: Abstract base class for implementing custom backends
- VisionAnalysisResult: Standard result format from any backend
- DomainConfig: Domain-specific configuration for metric extraction

Reference:
    Ma et al., "Segment Anything in Medical Images", Nature Communications 2024
    https://github.com/bowang-lab/MedSAM
"""

from .vision_provider import (
    VisionProvider,
    VisionBackend,
    VisionAnalysisResult,
    PlaceholderVisionBackend,
)

from .medsam2_engine import (
    MedSAMVisionEngine,
    MedSAMVisionProvider,
    MedSAM2VisionEngine,  # backward compat alias
    MedSAM2VisionProvider,  # backward compat alias
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
    # MedSAM Production (bowang-lab)
    "MedSAMVisionEngine",
    "MedSAMVisionProvider",
    # Backward compatibility aliases
    "MedSAM2VisionEngine",
    "MedSAM2VisionProvider",
    # Helpers
    "DomainConfig",
    "PromptType",
    "SegmentationPrompt",
    "SegmentationResult",
]

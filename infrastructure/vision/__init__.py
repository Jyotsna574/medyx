"""Vision Analysis Module.

Provides AI-powered medical image analysis through configurable backends.

Components:
- VisionProvider: Configurable provider (default: placeholder)
- VisionBackend: Abstract base class for custom backends
- VisionAnalysisResult: Standard result format
- PlaceholderVisionBackend: Mock backend for agent pipeline testing
- DomainConfig: Domain-specific configuration (when re-adding segmentation)
"""

from .vision_provider import (
    VisionProvider,
    VisionBackend,
    VisionAnalysisResult,
    PlaceholderVisionBackend,
    DomainConfig,
)

__all__ = [
    "VisionProvider",
    "VisionBackend",
    "VisionAnalysisResult",
    "PlaceholderVisionBackend",
    "DomainConfig",
]

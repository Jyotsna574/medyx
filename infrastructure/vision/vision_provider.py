"""
Vision Provider - Abstract vision analysis provider.

This module defines the base vision provider interface and a configurable
implementation that can work with different vision backends (local models,
cloud APIs, etc.).
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, field

from core.interfaces import AbstractVisionProvider
from core.schemas import VisionMetrics


@dataclass
class VisionAnalysisResult:
    """Generic result from any vision analysis backend."""
    
    findings: list[str] = field(default_factory=list)
    confidence_scores: dict[str, float] = field(default_factory=dict)
    extracted_geometry: dict[str, Any] = field(default_factory=dict)
    overall_risk_score: float = 0.0
    model_id: str = ""
    image_processed: bool = False
    error: Optional[str] = None
    raw_output: Optional[dict] = None


class VisionBackend(ABC):
    """Abstract base class for vision analysis backends.
    
    Implement this interface to add new vision analysis capabilities
    (e.g., torchxrayvision, cloud APIs, custom models).
    """
    
    @abstractmethod
    def load(self) -> bool:
        """Load the vision model/initialize the backend.
        
        Returns:
            True if successfully loaded, False otherwise.
        """
        ...
    
    @abstractmethod
    def analyze(self, image_path: str) -> VisionAnalysisResult:
        """Analyze a medical image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            VisionAnalysisResult with findings and metrics.
        """
        ...
    
    @abstractmethod
    def get_info(self) -> dict:
        """Get information about this backend.
        
        Returns:
            Dict with backend metadata (model_id, capabilities, etc.)
        """
        ...


class PlaceholderVisionBackend(VisionBackend):
    """Placeholder backend for development/testing.
    
    Returns mock results. Replace with real implementation.
    """
    
    def __init__(self, model_id: str = "placeholder-v1"):
        self.model_id = model_id
        self._loaded = False
    
    def load(self) -> bool:
        print(f"[PlaceholderVisionBackend] Initialized (model_id={self.model_id})")
        self._loaded = True
        return True
    
    def analyze(self, image_path: str) -> VisionAnalysisResult:
        if not self._loaded:
            self.load()
        
        if not os.path.exists(image_path):
            return VisionAnalysisResult(
                error=f"Image file not found: {image_path}",
                model_id=self.model_id
            )
        
        return VisionAnalysisResult(
            findings=[
                "Placeholder analysis - implement a real VisionBackend",
                f"Image: {os.path.basename(image_path)}",
            ],
            confidence_scores={"placeholder_finding": 0.5},
            extracted_geometry={},
            overall_risk_score=0.5,
            model_id=self.model_id,
            image_processed=True,
        )
    
    def get_info(self) -> dict:
        return {
            "model_id": self.model_id,
            "type": "placeholder",
            "loaded": self._loaded,
            "description": "Placeholder backend for development. Implement a real VisionBackend.",
        }


class VisionProvider(AbstractVisionProvider):
    """
    Configurable Vision Provider.
    
    Works with any VisionBackend implementation. Default uses a placeholder
    backend that should be replaced with a real implementation.
    """
    
    def __init__(
        self,
        backend: Optional[VisionBackend] = None,
        preload_model: bool = False
    ):
        """
        Initialize the Vision Provider.
        
        Args:
            backend: VisionBackend implementation to use. Defaults to PlaceholderVisionBackend.
            preload_model: If True, load the backend immediately at startup.
        """
        self.backend = backend or PlaceholderVisionBackend()
        self._loaded = False
        
        if preload_model:
            self._load()
    
    def _load(self) -> bool:
        """Load the vision backend."""
        if not self._loaded:
            print("Loading Vision Backend...")
            success = self.backend.load()
            if success:
                print(f"Vision Backend Ready: {self.backend.get_info().get('model_id', 'unknown')}")
                self._loaded = True
            else:
                print("Vision Backend: Failed to load")
            return success
        return True
    
    async def analyze(self, image_path: str) -> VisionMetrics:
        """
        Analyze a medical image and return vision metrics.
        
        Args:
            image_path: Path to the medical image file.
            
        Returns:
            VisionMetrics containing analysis results.
        """
        if not self._loaded:
            self._load()
        
        result = self.backend.analyze(image_path)
        return self._convert_to_metrics(result, image_path)
    
    def _convert_to_metrics(
        self,
        result: VisionAnalysisResult,
        image_path: str
    ) -> VisionMetrics:
        """
        Convert backend result to VisionMetrics format.
        
        Args:
            result: Raw result from the vision backend.
            image_path: Original image path for context.
            
        Returns:
            VisionMetrics object with analysis results.
        """
        if result.error:
            return VisionMetrics(
                risk_score=0.0,
                findings=[
                    f"Analysis Error: {result.error}",
                    "Manual review required",
                    f"Image: {os.path.basename(image_path)}"
                ],
                extracted_geometry={},
                confidence_scores={},
                model_id=result.model_id,
            )
        
        findings = result.findings.copy()
        findings.append(f"Model: {result.model_id}")
        findings.append(f"Image: {os.path.basename(image_path)}")
        
        return VisionMetrics(
            risk_score=min(1.0, max(0.0, result.overall_risk_score)),
            findings=findings,
            extracted_geometry=result.extracted_geometry,
            confidence_scores=result.confidence_scores,
            model_id=result.model_id,
        )
    
    def get_model_info(self) -> dict:
        """Get information about the vision backend."""
        return self.backend.get_info()

"""Abstract Base Classes defining the interfaces for MedicalAgentDiagnosis-MAD components."""

from abc import ABC, abstractmethod

from .schemas import VisionMetrics


class AbstractVisionProvider(ABC):
    """Abstract interface for vision analysis providers."""

    @abstractmethod
    async def analyze(self, image_path: str) -> VisionMetrics:
        """Analyze a retinal scan image and extract vision metrics.

        Args:
            image_path: Path to the retinal scan image file.

        Returns:
            VisionMetrics containing CDR, risk score, and findings.
        """
        ...


class AbstractKnowledgeRetriever(ABC):
    """Abstract interface for knowledge/guideline retrieval systems."""

    @abstractmethod
    async def search(self, query: str) -> str:
        """Search for relevant medical guidelines and knowledge.

        Args:
            query: The search query describing the medical condition or topic.

        Returns:
            A string containing relevant medical guidelines and information.
        """
        ...

"""
MedSAM Vision Engine - Medical image segmentation using bowang-lab/MedSAM.

This module provides MedSAM inference for medical image analysis,
extracting geometric metrics from segmentation masks for downstream
multi-agent diagnostic reasoning.

Based on: https://github.com/bowang-lab/MedSAM
Paper: "Segment Anything in Medical Images" (Nature Communications 2024)

Environment (set these in your SLURM script):
    MEDSAM_CHECKPOINT_PATH: Full path to checkpoint .pth file (required on cluster)
                            e.g. /scratch/ed21b031/models/medsam_checkpoints/sam_vit_b_01ec64.pth
    MEDSAM_ROOT: Path to MedSAM repo (where segment_anything module lives)
                 e.g. /home/ddp/medyx/MedSAM

No auto-download - checkpoints must be pre-downloaded for offline cluster use.
"""

import gc
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from loguru import logger
import numpy as np
from PIL import Image

from core.schemas import VisionMetrics
from infrastructure.vision.vision_provider import VisionBackend, VisionAnalysisResult

# Lazy imports for PyTorch and MedSAM to avoid loading CUDA until needed
torch = None
segment_anything = None


def _lazy_import_torch():
    """Lazily import PyTorch to defer CUDA initialization."""
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    return torch


def _get_checkpoint_path() -> str:
    """
    Resolve checkpoint path from environment.
    
    Priority:
    1. MEDSAM_CHECKPOINT_PATH - full path to .pth file
    2. MEDSAM_CHECKPOINT_DIR + sam_vit_b_01ec64.pth
    3. ./checkpoints/sam_vit_b_01ec64.pth (fallback)
    """
    # Direct path to checkpoint file (preferred for cluster)
    ckpt_path = os.environ.get("MEDSAM_CHECKPOINT_PATH")
    if ckpt_path:
        return ckpt_path
    
    # Directory-based (legacy)
    ckpt_dir = os.environ.get("MEDSAM_CHECKPOINT_DIR", "./checkpoints")
    return os.path.join(ckpt_dir, "sam_vit_b_01ec64.pth")


def _lazy_import_medsam():
    """Lazily import MedSAM/segment_anything modules."""
    global segment_anything
    if segment_anything is None:
        # Add MedSAM repo to path if specified
        medsam_root = os.environ.get("MEDSAM_ROOT")
        if medsam_root:
            import sys
            if medsam_root not in sys.path:
                sys.path.insert(0, medsam_root)
                logger.info(f"[MedSAM] Added MEDSAM_ROOT to sys.path: {medsam_root}")
        try:
            import torchvision  # noqa: F401 - required by segment_anything
            from segment_anything import sam_model_registry, SamPredictor
            segment_anything = {
                "sam_model_registry": sam_model_registry,
                "SamPredictor": SamPredictor,
            }
            logger.info("[MedSAM] segment_anything loaded successfully")
        except ImportError as e:
            missing = str(e).replace("No module named ", "").strip("'")
            logger.error(f"[MedSAM] Import FAILED - missing: {missing}")
            if "torchvision" in str(e):
                logger.error("[MedSAM] Fix: pip install torchvision")
            elif "segment_anything" in str(e):
                logger.error("[MedSAM] Fix: Set MEDSAM_ROOT to your MedSAM repo path")
                logger.error("[MedSAM]      e.g. export MEDSAM_ROOT=/home/ddp/medyx/MedSAM")
            segment_anything = None
    return segment_anything


class PromptType(Enum):
    """Types of spatial prompts for segmentation."""
    BOUNDING_BOX = "bbox"
    POINT = "point"
    POINTS = "points"
    AUTOMATIC = "auto"


@dataclass
class SegmentationPrompt:
    """Spatial prompt for guiding segmentation."""
    prompt_type: PromptType
    data: Any
    label: int = 1  # 1 for foreground, 0 for background


@dataclass 
class DomainConfig:
    """Configuration for domain-specific metric extraction."""
    domain: str  # e.g., "ophthalmic", "thoracic", "abdominal"
    modality: str  # e.g., "fundoscopy", "xray", "ct"
    target_structure: str  # e.g., "optic_disc", "lung", "liver"
    pixel_spacing_mm: Optional[float] = None  # mm per pixel
    slice_thickness_mm: Optional[float] = None  # for 3D volumes
    compute_ratios: list[str] = field(default_factory=list)  # e.g., ["cdr"]
    custom_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationResult:
    """Result from MedSAM segmentation."""
    mask: np.ndarray  # Binary segmentation mask (H, W) or (D, H, W)
    confidence_score: float  # Model confidence
    logits: Optional[np.ndarray] = None  # Raw logits before thresholding
    prompt_used: Optional[SegmentationPrompt] = None


class MedSAMVisionEngine(VisionBackend):
    """
    Production MedSAM Vision Engine (bowang-lab/MedSAM).
    
    Performs real segmentation inference using SAM ViT-B architecture
    fine-tuned on medical images. Extracts geometric metrics from masks
    and returns validated VisionMetrics for downstream multi-agent processing.
    
    Reference:
        Ma et al., "Segment Anything in Medical Images", Nature Communications (2024)
        https://github.com/bowang-lab/MedSAM
    """
    
    # SAM ViT-B model type (works with both SAM and MedSAM checkpoints)
    MODEL_TYPE = "vit_b"
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        use_float16: bool = True,
        low_memory_mode: bool = False,
    ):
        """
        Initialize the MedSAM Vision Engine.
        
        Args:
            checkpoint_path: Full path to checkpoint .pth file. Defaults to
                            MEDSAM_CHECKPOINT_PATH env var.
            device: Device to run inference on ('cuda', 'cpu', or None for auto).
            use_float16: Use FP16 for reduced memory on GPU.
            low_memory_mode: Enable aggressive memory optimization for limited VRAM.
        """
        self.checkpoint_path = checkpoint_path or _get_checkpoint_path()
        self.use_float16 = use_float16
        self.low_memory_mode = low_memory_mode
        
        # Determine device
        if device is None:
            _torch = _lazy_import_torch()
            self.device = "cuda" if _torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self._model = None
        self._predictor = None
        self._loaded = False
        self._current_image_set = False
        
        logger.info(f"[MedSAMVisionEngine] Initialized | Model: medsam_vit_b | Device: {self.device} | FP16: {use_float16} | Low Memory: {low_memory_mode}")
    
    def load(self) -> bool:
        """
        Load the MedSAM model into memory.
        
        Returns:
            True if successfully loaded, False otherwise.
        """
        if self._loaded:
            return True
        
        _torch = _lazy_import_torch()
        _medsam = _lazy_import_medsam()
        
        if _medsam is None:
            logger.error("[MedSAMVisionEngine] segment_anything not available.")
            logger.error("[MedSAMVisionEngine] Set MEDSAM_ROOT to your MedSAM repo path.")
            return False
        
        try:
            # Checkpoint must exist - no auto-download on cluster
            checkpoint_file = Path(self.checkpoint_path) if self.checkpoint_path else None
            if not checkpoint_file or not checkpoint_file.is_file():
                logger.error(f"[MedSAMVisionEngine] Checkpoint NOT FOUND: {checkpoint_file}")
                logger.error("[MedSAMVisionEngine] Set MEDSAM_CHECKPOINT_PATH to the MedSAM fine-tuned .pth file:")
                logger.error("[MedSAMVisionEngine]   export MEDSAM_CHECKPOINT_PATH=/scratch/ed21b031/models/medsam_checkpoints/medsam_vit_b.pth")
                logger.error("[MedSAMVisionEngine]   (download with: bash download_ckpts.sh)")
                return False

            logger.info(f"[MedSAMVisionEngine] Loading checkpoint: {checkpoint_file}")
            
            # For low-memory mode on limited VRAM GPUs
            if self.low_memory_mode and self.device == "cuda":
                _torch.cuda.empty_cache()
                gc.collect()
            
            # Build model using sam_model_registry (works with SAM or MedSAM checkpoint)
            self._model = _medsam["sam_model_registry"][self.MODEL_TYPE](
                checkpoint=str(checkpoint_file)
            )
            self._model = self._model.to(self.device)
            
            # Use FP16 for memory savings on GPU
            if self.use_float16 and self.device == "cuda":
                self._model = self._model.half()
            
            # Set to eval mode
            self._model.eval()
            
            # Create predictor
            self._predictor = _medsam["SamPredictor"](self._model)
            
            self._loaded = True
            logger.info(f"[MedSAMVisionEngine] Model loaded successfully")
            
            # Report memory usage
            if self.device == "cuda":
                allocated = _torch.cuda.memory_allocated() / 1024**3
                logger.debug(f"  GPU Memory Used: {allocated:.2f} GB")
            
            return True
            
        except Exception as e:
            logger.exception(f"[MedSAMVisionEngine] Failed to load model: {e}")
            return False
    
    def analyze(
        self,
        image_path: str,
        prompt_type: Union[str, PromptType] = PromptType.AUTOMATIC,
        prompt_data: Optional[Any] = None,
        domain_config: Optional[DomainConfig] = None,
    ) -> VisionAnalysisResult:
        """
        Analyze a medical image using MedSAM segmentation.
        
        Args:
            image_path: Path to the medical image file.
            prompt_type: Type of spatial prompt ('bbox', 'point', 'points', 'auto').
            prompt_data: Prompt coordinates:
                - bbox: [x_min, y_min, x_max, y_max]
                - point: [x, y]
                - points: [[x1, y1], [x2, y2], ...]
                - auto: None (automatic prompt generation)
            domain_config: Domain-specific configuration for metric extraction.
            
        Returns:
            VisionAnalysisResult with segmentation findings and geometry.
        """
        if not self._loaded:
            if not self.load():
                return VisionAnalysisResult(
                    error="Failed to load MedSAM model",
                    model_id="medsam_vit_b",
                )
        
        _torch = _lazy_import_torch()
        
        # Validate image path
        if not os.path.exists(image_path):
            return VisionAnalysisResult(
                error=f"Image file not found: {image_path}",
                model_id="medsam_vit_b",
            )
        
        try:
            # Load and preprocess image
            image = self._load_image(image_path)
            
            # Set image in predictor
            self._set_image(image)
            
            # Prepare prompt
            prompt = self._prepare_prompt(prompt_type, prompt_data, image.shape)
            
            # Run segmentation inference
            seg_result = self._run_inference(prompt)
            
            # Extract geometric metrics
            if domain_config is None:
                domain_config = self._infer_domain_config(image_path)
            
            geometry = self.extract_geometric_metrics(seg_result.mask, domain_config)
            
            # Generate findings
            findings = self._generate_findings(seg_result, geometry, domain_config)
            
            # Calculate risk score from geometry
            risk_score = self._calculate_risk_score(geometry, domain_config)
            
            # Clear image from predictor to free memory
            self._clear_image()
            
            return VisionAnalysisResult(
                findings=findings,
                confidence_scores={"segmentation": seg_result.confidence_score},
                extracted_geometry=geometry,
                overall_risk_score=risk_score,
                model_id="medsam_vit_b",
                image_processed=True,
                raw_output={
                    "mask_shape": list(seg_result.mask.shape),
                    "prompt_type": prompt.prompt_type.value if prompt else "auto",
                },
            )
            
        except Exception as e:
            logger.exception(f"[MedSAMVisionEngine] Analysis error: {e}")
            return VisionAnalysisResult(
                error=str(e),
                model_id="medsam_vit_b",
            )
        finally:
            # Aggressive memory cleanup for low-memory mode
            if self.low_memory_mode and self.device == "cuda":
                self._clear_cuda_cache()
    
    def extract_geometric_metrics(
        self,
        mask: np.ndarray,
        domain_config: DomainConfig,
    ) -> dict[str, Any]:
        """
        Extract geometric metrics from a segmentation mask.
        
        Args:
            mask: Binary segmentation mask (H, W) or (D, H, W).
            domain_config: Domain-specific configuration.
            
        Returns:
            Dictionary of computed geometric metrics.
        """
        geometry = {}
        
        # Basic pixel-level metrics
        mask_binary = (mask > 0).astype(np.uint8)
        pixel_area = int(np.sum(mask_binary))
        geometry["pixel_area"] = pixel_area

        if pixel_area == 0:
            raise ValueError(
                "Segmentation produced an empty mask. "
                "Ensure the anatomical bbox is within image bounds and contains "
                "the target structure. Use image dimensions (e.g. [0.1*w, 0.1*h, 0.9*w, 0.9*h])."
            )
        
        # Convert to physical units if spacing is available
        if domain_config.pixel_spacing_mm is not None:
            area_mm2 = pixel_area * (domain_config.pixel_spacing_mm ** 2)
            geometry["area_mm2"] = round(area_mm2, 4)
        
        # 2D metrics
        if mask_binary.ndim == 2:
            geometry.update(self._extract_2d_metrics(mask_binary, domain_config))
        # 3D metrics
        elif mask_binary.ndim == 3:
            geometry.update(self._extract_3d_metrics(mask_binary, domain_config))
        
        # Domain-specific ratio calculations
        for ratio_name in domain_config.compute_ratios:
            ratio_value = self._compute_ratio(ratio_name, mask_binary, geometry, domain_config)
            if ratio_value is not None:
                geometry[ratio_name] = ratio_value
        
        return geometry
    
    def _extract_2d_metrics(
        self,
        mask: np.ndarray,
        domain_config: DomainConfig,
    ) -> dict[str, Any]:
        """Extract 2D geometric metrics."""
        metrics = {}
        
        try:
            from skimage import measure
            from skimage.measure import regionprops
        except ImportError:
            metrics["warning"] = "scikit-image not available for detailed metrics"
            return metrics
        
        # Find connected components
        labeled = measure.label(mask, connectivity=2)
        regions = regionprops(labeled)
        
        if not regions:
            return metrics
        
        # Use largest region
        largest_region = max(regions, key=lambda r: r.area)
        
        # Bounding box
        minr, minc, maxr, maxc = largest_region.bbox
        metrics["bbox"] = [int(minc), int(minr), int(maxc), int(maxr)]
        metrics["bbox_width_px"] = int(maxc - minc)
        metrics["bbox_height_px"] = int(maxr - minr)
        
        # Centroid
        cy, cx = largest_region.centroid
        metrics["centroid"] = [round(cx, 2), round(cy, 2)]
        
        # Shape metrics
        metrics["perimeter_px"] = round(largest_region.perimeter, 2)
        metrics["eccentricity"] = round(largest_region.eccentricity, 4)
        metrics["solidity"] = round(largest_region.solidity, 4)
        
        # Equivalent diameter (diameter of circle with same area)
        equiv_diam = getattr(largest_region, 'equivalent_diameter_area', None)
        if equiv_diam is None:
            equiv_diam = largest_region.equivalent_diameter
        metrics["equivalent_diameter_px"] = round(equiv_diam, 2)
        
        # Major/minor axis lengths
        major_axis = getattr(largest_region, 'axis_major_length', None)
        if major_axis is None:
            major_axis = largest_region.major_axis_length
        minor_axis = getattr(largest_region, 'axis_minor_length', None)
        if minor_axis is None:
            minor_axis = largest_region.minor_axis_length
        metrics["major_axis_px"] = round(major_axis, 2)
        metrics["minor_axis_px"] = round(minor_axis, 2)
        
        # Convert to mm if pixel spacing available
        if domain_config.pixel_spacing_mm is not None:
            ps = domain_config.pixel_spacing_mm
            metrics["bbox_width_mm"] = round(metrics["bbox_width_px"] * ps, 2)
            metrics["bbox_height_mm"] = round(metrics["bbox_height_px"] * ps, 2)
            metrics["perimeter_mm"] = round(metrics["perimeter_px"] * ps, 2)
            metrics["equivalent_diameter_mm"] = round(metrics["equivalent_diameter_px"] * ps, 2)
            metrics["major_axis_mm"] = round(metrics["major_axis_px"] * ps, 2)
            metrics["minor_axis_mm"] = round(metrics["minor_axis_px"] * ps, 2)
        
        # Number of connected components
        metrics["num_components"] = len(regions)
        
        return metrics
    
    def _extract_3d_metrics(
        self,
        mask: np.ndarray,
        domain_config: DomainConfig,
    ) -> dict[str, Any]:
        """Extract 3D volumetric metrics."""
        metrics = {}
        
        voxel_count = int(np.sum(mask))
        metrics["voxel_count"] = voxel_count
        
        # Calculate volume in mm³ if spacing available
        if domain_config.pixel_spacing_mm and domain_config.slice_thickness_mm:
            voxel_volume = (domain_config.pixel_spacing_mm ** 2) * domain_config.slice_thickness_mm
            volume_mm3 = voxel_count * voxel_volume
            metrics["volume_mm3"] = round(volume_mm3, 2)
            metrics["volume_cm3"] = round(volume_mm3 / 1000, 4)
        
        # Surface area estimation (marching cubes)
        try:
            from skimage import measure
            verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
            surface_area = measure.mesh_surface_area(verts, faces)
            metrics["surface_area_voxels"] = round(surface_area, 2)
            
            if domain_config.pixel_spacing_mm:
                ps = domain_config.pixel_spacing_mm
                metrics["surface_area_mm2"] = round(surface_area * ps * ps, 2)
        except Exception:
            pass
        
        return metrics
    
    def _compute_ratio(
        self,
        ratio_name: str,
        mask: np.ndarray,
        geometry: dict,
        domain_config: DomainConfig,
    ) -> Optional[float]:
        """Compute domain-specific ratios."""
        
        if ratio_name == "cdr":
            # Cup-to-Disc Ratio for ophthalmic imaging
            # Requires separate cup and disc segmentation
            # This is a simplified estimation based on mask geometry
            if "solidity" in geometry and "eccentricity" in geometry:
                # Estimate CDR from shape characteristics
                # Lower solidity + higher eccentricity suggests larger cup
                estimated_cdr = 1.0 - geometry["solidity"]
                return round(min(1.0, max(0.0, estimated_cdr)), 3)
        
        elif ratio_name == "aspect_ratio":
            if "major_axis_px" in geometry and "minor_axis_px" in geometry:
                if geometry["minor_axis_px"] > 0:
                    return round(geometry["major_axis_px"] / geometry["minor_axis_px"], 3)
        
        elif ratio_name == "circularity":
            if "perimeter_px" in geometry and "pixel_area" in geometry:
                perimeter = geometry["perimeter_px"]
                area = geometry["pixel_area"]
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    return round(min(1.0, circularity), 4)
        
        return None
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess an image for MedSAM."""
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return np.array(image)
    
    def _set_image(self, image: np.ndarray) -> None:
        """Set the current image in the predictor."""
        _torch = _lazy_import_torch()
        
        with _torch.no_grad():
            self._predictor.set_image(image)
        self._current_image_set = True
    
    def _clear_image(self) -> None:
        """Clear the current image from predictor."""
        self._current_image_set = False
        # Reset predictor state
        if self._predictor is not None:
            self._predictor.reset_image()
    
    def _prepare_prompt(
        self,
        prompt_type: Union[str, PromptType],
        prompt_data: Optional[Any],
        image_shape: tuple,
    ) -> Optional[SegmentationPrompt]:
        """Prepare segmentation prompt from input. Clamps bbox to image bounds."""
        h, w = image_shape[:2]

        if isinstance(prompt_type, str):
            prompt_type = PromptType(prompt_type)

        if prompt_type == PromptType.AUTOMATIC:
            # Generate automatic center-region prompt
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            prompt_data = [margin_x, margin_y, w - margin_x, h - margin_y]
            prompt_type = PromptType.BOUNDING_BOX
        elif prompt_type == PromptType.BOUNDING_BOX and prompt_data is not None:
            # Clamp bbox to image dimensions - prevents empty mask from out-of-bounds coords
            x_min, y_min, x_max, y_max = prompt_data[:4]
            x_min = max(0, min(int(x_min), w - 1))
            y_min = max(0, min(int(y_min), h - 1))
            x_max = max(x_min + 1, min(int(x_max), w))
            y_max = max(y_min + 1, min(int(y_max), h))
            prompt_data = [x_min, y_min, x_max, y_max]

        return SegmentationPrompt(
            prompt_type=prompt_type,
            data=prompt_data,
        )
    
    def _run_inference(self, prompt: SegmentationPrompt) -> SegmentationResult:
        """Run MedSAM inference with the given prompt."""
        _torch = _lazy_import_torch()
        
        with _torch.no_grad():
            if prompt.prompt_type == PromptType.BOUNDING_BOX:
                box = np.array(prompt.data, dtype=np.float32)
                masks, scores, logits = self._predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box,
                    multimask_output=False,
                )
            
            elif prompt.prompt_type == PromptType.POINT:
                point = np.array([prompt.data], dtype=np.float32)
                label = np.array([prompt.label], dtype=np.int32)
                masks, scores, logits = self._predictor.predict(
                    point_coords=point,
                    point_labels=label,
                    multimask_output=False,
                )
            
            elif prompt.prompt_type == PromptType.POINTS:
                points = np.array(prompt.data, dtype=np.float32)
                labels = np.ones(len(points), dtype=np.int32) * prompt.label
                masks, scores, logits = self._predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=False,
                )
            
            else:
                raise ValueError(f"Unsupported prompt type: {prompt.prompt_type}")
        
        # Get the best mask
        best_idx = np.argmax(scores) if len(scores) > 1 else 0
        best_mask = masks[best_idx]
        best_score = float(scores[best_idx])
        best_logits = logits[best_idx] if logits is not None else None
        
        return SegmentationResult(
            mask=best_mask,
            confidence_score=best_score,
            logits=best_logits,
            prompt_used=prompt,
        )
    
    def _infer_domain_config(self, image_path: str) -> DomainConfig:
        """Infer domain configuration from image path/metadata."""
        path_lower = image_path.lower()
        
        # Try to infer from filename
        if any(x in path_lower for x in ["fundus", "retina", "eye", "optic"]):
            return DomainConfig(
                domain="ophthalmic",
                modality="fundoscopy",
                target_structure="optic_disc",
                compute_ratios=["cdr", "circularity"],
            )
        elif any(x in path_lower for x in ["chest", "xray", "lung", "thorax"]):
            return DomainConfig(
                domain="thoracic",
                modality="xray",
                target_structure="lung_region",
                compute_ratios=["aspect_ratio", "circularity"],
            )
        elif any(x in path_lower for x in ["ct", "liver", "abdomen"]):
            return DomainConfig(
                domain="abdominal",
                modality="ct",
                target_structure="organ",
                compute_ratios=["circularity"],
            )
        
        # Default generic config
        return DomainConfig(
            domain="general",
            modality="unknown",
            target_structure="region_of_interest",
            compute_ratios=["circularity", "aspect_ratio"],
        )
    
    def _generate_findings(
        self,
        seg_result: SegmentationResult,
        geometry: dict,
        domain_config: DomainConfig,
    ) -> list[str]:
        """Generate clinical findings from segmentation results."""
        findings = []
        
        # Segmentation quality
        if seg_result.confidence_score >= 0.9:
            findings.append(f"High-confidence segmentation of {domain_config.target_structure}")
        elif seg_result.confidence_score >= 0.7:
            findings.append(f"Moderate-confidence segmentation of {domain_config.target_structure}")
        else:
            findings.append(f"Low-confidence segmentation - manual verification recommended")
        
        # Size findings
        if "area_mm2" in geometry:
            findings.append(f"Segmented area: {geometry['area_mm2']:.2f} mm²")
        elif "pixel_area" in geometry:
            findings.append(f"Segmented area: {geometry['pixel_area']} pixels")
        
        # Shape findings
        if "circularity" in geometry:
            circ = geometry["circularity"]
            if circ > 0.9:
                findings.append("Shape: highly circular/regular")
            elif circ > 0.7:
                findings.append("Shape: moderately circular")
            else:
                findings.append("Shape: irregular/non-circular")
        
        # Domain-specific findings
        if domain_config.domain == "ophthalmic" and "cdr" in geometry:
            cdr = geometry["cdr"]
            if cdr >= 0.7:
                findings.append(f"Elevated Cup-to-Disc Ratio ({cdr:.2f}) - glaucoma risk indicator")
            elif cdr >= 0.5:
                findings.append(f"Borderline Cup-to-Disc Ratio ({cdr:.2f}) - monitor recommended")
            else:
                findings.append(f"Normal Cup-to-Disc Ratio ({cdr:.2f})")
        
        # Component findings
        if "num_components" in geometry and geometry["num_components"] > 1:
            findings.append(f"Multiple disconnected regions detected ({geometry['num_components']})")
        
        return findings
    
    def _calculate_risk_score(
        self,
        geometry: dict,
        domain_config: DomainConfig,
    ) -> float:
        """Calculate overall risk score from geometric metrics."""
        risk_factors = []
        
        # CDR-based risk (ophthalmic)
        if "cdr" in geometry:
            cdr = geometry["cdr"]
            if cdr >= 0.8:
                risk_factors.append(0.9)
            elif cdr >= 0.7:
                risk_factors.append(0.7)
            elif cdr >= 0.5:
                risk_factors.append(0.4)
            else:
                risk_factors.append(0.1)
        
        # Shape irregularity risk
        if "circularity" in geometry:
            circ = geometry["circularity"]
            irregularity_risk = 1.0 - circ
            risk_factors.append(irregularity_risk * 0.5)
        
        # Solidity risk (gaps/holes suggest pathology)
        if "solidity" in geometry:
            solidity = geometry["solidity"]
            if solidity < 0.8:
                risk_factors.append(0.3)
        
        # Calculate weighted average
        if risk_factors:
            return min(1.0, sum(risk_factors) / len(risk_factors))
        
        return 0.3  # Default moderate risk for unknown cases
    
    def _clear_cuda_cache(self) -> None:
        """Clear CUDA memory cache."""
        _torch = _lazy_import_torch()
        if self.device == "cuda":
            _torch.cuda.empty_cache()
            gc.collect()
    
    def unload(self) -> None:
        """Unload the model and free all GPU memory."""
        _torch = _lazy_import_torch()
        
        self._predictor = None
        self._model = None
        self._loaded = False
        self._current_image_set = False
        
        if self.device == "cuda":
            _torch.cuda.empty_cache()
            gc.collect()
        
        logger.info("[MedSAMVisionEngine] Model unloaded")
    
    def get_info(self) -> dict:
        """Get information about this backend."""
        _torch = _lazy_import_torch()
        
        info = {
            "model_id": "medsam_vit_b",
            "type": "MedSAM (bowang-lab)",
            "architecture": "ViT-B",
            "loaded": self._loaded,
            "device": self.device,
            "fp16": self.use_float16,
            "low_memory_mode": self.low_memory_mode,
            "reference": "Ma et al., Nature Communications 2024",
        }
        
        if self._loaded and self.device == "cuda":
            info["gpu_memory_allocated_gb"] = round(_torch.cuda.memory_allocated() / 1024**3, 2)
            info["gpu_memory_reserved_gb"] = round(_torch.cuda.memory_reserved() / 1024**3, 2)
        
        return info


# Backward compatibility aliases
MedSAM2VisionEngine = MedSAMVisionEngine


class MedSAMVisionProvider:
    """
    High-level Vision Provider using MedSAM engine.
    
    Implements the AbstractVisionProvider interface for seamless
    integration with the diagnostic pipeline.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        preload_model: bool = False,
        low_memory_mode: bool = True,
    ):
        """
        Initialize the MedSAM Vision Provider.
        
        Args:
            checkpoint_path: Path to model checkpoints.
            preload_model: Load model immediately at startup.
            low_memory_mode: Enable memory optimization for limited VRAM.
        """
        self.engine = MedSAMVisionEngine(
            checkpoint_path=checkpoint_path,
            low_memory_mode=low_memory_mode,
        )
        
        if preload_model:
            self.engine.load()
    
    async def analyze(
        self,
        image_path: str,
        prompt_type: str = "auto",
        prompt_data: Optional[Any] = None,
        domain_config: Optional[DomainConfig] = None,
    ) -> VisionMetrics:
        """
        Analyze a medical image and return vision metrics.
        
        Args:
            image_path: Path to the medical image file.
            prompt_type: Type of prompt ('auto', 'bbox', 'point', 'points').
            prompt_data: Prompt coordinates if not automatic.
            domain_config: Domain-specific configuration.
            
        Returns:
            VisionMetrics containing analysis results.
        """
        result = self.engine.analyze(
            image_path=image_path,
            prompt_type=prompt_type,
            prompt_data=prompt_data,
            domain_config=domain_config,
        )
        
        return self._convert_to_metrics(result, image_path)
    
    def _convert_to_metrics(
        self,
        result: VisionAnalysisResult,
        image_path: str,
    ) -> VisionMetrics:
        """Convert engine result to VisionMetrics."""
        if result.error:
            logger.error(f"[MedSAM] Vision analysis FAILED: {result.error}")
            return VisionMetrics(
                risk_score=0.0,
                findings=[f"Analysis Error: {result.error}"],
                extracted_geometry={},
                confidence_scores={},
                model_id=result.model_id,
            )
        
        return VisionMetrics(
            risk_score=min(1.0, max(0.0, result.overall_risk_score)),
            findings=result.findings,
            extracted_geometry=result.extracted_geometry,
            confidence_scores=result.confidence_scores,
            model_id=result.model_id,
        )
    
    def unload_model(self) -> None:
        """Unload the model to free GPU memory."""
        self.engine.unload()
    
    def get_model_info(self) -> dict:
        """Get information about the vision engine."""
        return self.engine.get_info()


# Backward compatibility alias
MedSAM2VisionProvider = MedSAMVisionProvider

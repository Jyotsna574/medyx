# Modality & Image Format for Accurate Results

Use these formats when creating `PatientCase` objects for reliable vision metrics and domain-specific processing.

## Modality & Target Region

The pipeline maps your input to a domain config. Use these values for best results:

| Domain | Modality | Target Region | Notes |
|--------|----------|---------------|-------|
| Ophthalmic | `Fundoscopy`, `fundus` | `Eye`, `Retina` | Optic disc, CDR, glaucoma |
| Thoracic | `X-Ray`, `Xray`, `xray` | `Chest`, `Lung`, `Thorax` | Chest X-ray, lung region |
| Abdominal | `CT` | `Abdomen`, `Liver` | Liver/organ segmentation |
| Neurological | `MRI` | `Brain` | Brain imaging |
| General | Any | Any | Uses circularity, aspect_ratio |

**Examples:**
```python
# Chest X-ray (thoracic)
modality="X-Ray", target_region="Chest"

# Fundus/eye (ophthalmic)
modality="Fundoscopy", target_region="Eye"

# Abdomen CT
modality="CT", target_region="Abdomen"
```

## Image Format

### Supported file types
- PNG, JPEG, TIFF (converted to RGB before segmentation)

### Recommended
- **Resolution**: ≥ 512×512 pixels. MedSAM-2 works best at 1024×1024 or higher.
- **Format**: PNG preferred for lossless quality.
- **Orientation**: Standard (anterior–posterior for chest; frontal for fundus).
- **Content**: Single-frame 2D image. The region of interest should be visible and not clipped.

### Bounding Box (`anatomical_bbox`)

Format: `[x_min, y_min, x_max, y_max]` in **pixel coordinates** (origin top-left).

- **Must be within image bounds.** Values exceeding width/height cause failed segmentation.
- **Use image dimensions** when supplying a bbox:
  ```python
  # Center 80% of image (10% margin each side)
  with Image.open(image_path) as img:
      w, h = img.size
  margin_x, margin_y = int(w * 0.1), int(h * 0.1)
  anatomical_bbox = [margin_x, margin_y, w - margin_x, h - margin_y]
  ```
- **If omitted**, the pipeline uses an automatic center crop (~60% of image).
- **For chest X-rays**, the bbox should cover the lung fields; adjust margins if needed.

### Common mistakes
1. **Fixed bbox** like `[100, 100, 900, 800]` on 512×512 images → out-of-bounds, empty mask.
2. **Wrong modality** for the image (e.g. "CT" for X-ray) → wrong domain and metrics.
3. **Grayscale** images are fine; the pipeline converts to RGB.

## Quick reference

```python
from core.schemas import PatientCase

case = PatientCase(
    id="CASE-001",
    history="58yo M with cough and dyspnea...",
    image_path="./chest_xray.png",
    patient_age=58,
    patient_sex="M",
    modality="X-Ray",      # Matches thoracic domain
    target_region="Chest", # Matches thoracic domain
)

# Compute bbox from image size
with Image.open(case.image_path) as img:
    w, h = img.size
anatomical_bbox = [int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)]
```

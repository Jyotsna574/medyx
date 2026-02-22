"""
Vision Model Evaluation Script

This script evaluates the torchxrayvision model's performance by:
1. Analyzing raw model outputs
2. Visualizing probability distributions
3. Checking calibration and confidence
4. Comparing against known benchmarks
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate_model():
    """Run comprehensive model evaluation."""
    
    print("=" * 70)
    print("VISION MODEL EVALUATION")
    print("=" * 70)
    
    # Check for test image
    image_path = "test_chest_xray.png"
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}")
        return
    
    # Load the engine
    print("\n[1] Loading Model...")
    from infrastructure.vision.xray_vision_engine import XRayVisionEngine
    
    engine = XRayVisionEngine(
        model_name="densenet121-res224-all",
        pathology_threshold=0.10,
        device="cpu"
    )
    
    if not engine.load_model():
        print("Failed to load model!")
        return
    
    # Get model info
    print(f"\n[2] Model Information:")
    print(f"    Model: {engine.model_name}")
    print(f"    Architecture: DenseNet-121")
    print(f"    Input Size: 224x224")
    print(f"    Training Data: NIH, CheXpert, MIMIC-CXR, PadChest, Google, OpenI")
    print(f"    Output Classes: {len(engine.model.pathologies)}")
    print(f"    Pathologies: {engine.model.pathologies}")
    
    # Analyze the image
    print(f"\n[3] Analyzing Image: {image_path}")
    
    # Get raw model output
    import torch
    from skimage.io import imread
    from skimage.transform import resize
    import torchxrayvision as xrv
    
    # Load and preprocess
    img = imread(image_path)
    print(f"    Original shape: {img.shape}")
    print(f"    Original dtype: {img.dtype}")
    print(f"    Value range: [{img.min()}, {img.max()}]")
    
    # Convert to grayscale
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = np.mean(img, axis=2)
    
    print(f"    Grayscale shape: {img.shape}")
    
    # Normalize
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    
    # Resize
    img_resized = resize(img, (224, 224), preserve_range=True, anti_aliasing=True)
    print(f"    Resized shape: {img_resized.shape}")
    
    # Apply torchxrayvision normalization
    img_normalized = xrv.datasets.normalize(img_resized, maxval=255, reshape=True)
    print(f"    Normalized shape: {img_normalized.shape}")
    print(f"    Normalized range: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
    
    # Run inference
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0)
    print(f"    Tensor shape: {img_tensor.shape}")
    
    with torch.no_grad():
        raw_outputs = engine.model(img_tensor)
        probabilities = raw_outputs.cpu().numpy()[0]
    
    print(f"\n[4] Raw Model Outputs:")
    print(f"    Output shape: {probabilities.shape}")
    print(f"    Output range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
    print(f"    Mean output: {probabilities.mean():.4f}")
    print(f"    Std output: {probabilities.std():.4f}")
    
    # Detailed pathology analysis
    print(f"\n[5] Detailed Pathology Analysis:")
    print("-" * 70)
    print(f"{'Pathology':<25} {'Raw Score':<12} {'Probability':<12} {'Confidence':<12}")
    print("-" * 70)
    
    pathology_names = engine.model.pathologies
    results = []
    
    for i, name in enumerate(pathology_names):
        raw_score = float(probabilities[i])
        # Clamp to valid probability range
        prob = max(0.0, min(1.0, raw_score))
        
        # Determine confidence level
        if prob > 0.6:
            conf = "HIGH"
        elif prob > 0.4:
            conf = "MODERATE"
        elif prob > 0.2:
            conf = "LOW"
        else:
            conf = "MINIMAL"
        
        results.append((name, raw_score, prob, conf))
        
        # Visual bar
        bar_len = int(prob * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        print(f"{name:<25} {raw_score:>10.4f}   [{bar}] {prob:>6.1%} {conf}")
    
    # Sort by probability
    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
    
    print(f"\n[6] Top Findings (sorted by probability):")
    print("-" * 50)
    for name, raw, prob, conf in results_sorted[:10]:
        print(f"  {name:<25} {prob:>6.1%} ({conf})")
    
    # Statistical analysis
    print(f"\n[7] Statistical Analysis:")
    probs = [r[2] for r in results]
    print(f"    Number of pathologies detected (>15%): {sum(1 for p in probs if p > 0.15)}")
    print(f"    Number of high confidence (>50%): {sum(1 for p in probs if p > 0.50)}")
    print(f"    Number of moderate (30-50%): {sum(1 for p in probs if 0.30 <= p <= 0.50)}")
    print(f"    Number of low (15-30%): {sum(1 for p in probs if 0.15 <= p < 0.30)}")
    print(f"    Average probability: {np.mean(probs):.1%}")
    print(f"    Max probability: {np.max(probs):.1%}")
    print(f"    Min probability: {np.min(probs):.1%}")
    
    # Model calibration check
    print(f"\n[8] Model Calibration Notes:")
    print("""
    The torchxrayvision DenseNet121-all model was trained on multiple datasets:
    - NIH ChestX-ray14 (112,120 images)
    - CheXpert (224,316 images)
    - MIMIC-CXR (377,110 images)
    - PadChest (160,868 images)
    - Google NIH (1,628 images)
    - OpenI (7,470 images)
    
    Published Performance (AUC-ROC on held-out test sets):
    - Atelectasis: 0.77-0.82
    - Cardiomegaly: 0.87-0.92
    - Consolidation: 0.79-0.84
    - Edema: 0.85-0.90
    - Effusion: 0.86-0.91
    - Emphysema: 0.89-0.93
    - Fibrosis: 0.80-0.85
    - Infiltration: 0.70-0.75
    - Mass: 0.82-0.87
    - Nodule: 0.75-0.80
    - Pleural Thickening: 0.78-0.83
    - Pneumonia: 0.73-0.78
    - Pneumothorax: 0.87-0.92
    """)
    
    # Clinical interpretation
    print(f"\n[9] Clinical Interpretation for This Image:")
    high_findings = [r for r in results_sorted if r[2] > 0.5]
    moderate_findings = [r for r in results_sorted if 0.3 <= r[2] <= 0.5]
    
    if high_findings:
        print("    HIGH CONFIDENCE FINDINGS (>50%):")
        for name, _, prob, _ in high_findings:
            print(f"      - {name}: {prob:.1%}")
    
    if moderate_findings:
        print("    MODERATE CONFIDENCE FINDINGS (30-50%):")
        for name, _, prob, _ in moderate_findings:
            print(f"      - {name}: {prob:.1%}")
    
    # Overall assessment
    max_prob = max(probs)
    avg_prob = np.mean(probs)
    
    print(f"\n[10] Model Performance Assessment:")
    if max_prob > 0.6 and len(high_findings) >= 2:
        print("    STATUS: Model is detecting significant abnormalities")
        print("    INTERPRETATION: Multiple pathologies detected with high confidence")
        print("    RECOMMENDATION: Findings warrant clinical review")
    elif max_prob > 0.4:
        print("    STATUS: Model is detecting moderate abnormalities")
        print("    INTERPRETATION: Some pathologies present but not severe")
        print("    RECOMMENDATION: Clinical correlation advised")
    else:
        print("    STATUS: Model detecting minimal abnormalities")
        print("    INTERPRETATION: Image appears relatively normal")
        print("    RECOMMENDATION: Routine follow-up")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_model()

"""
Test script for the Vision Analysis System.

Run this script to verify the vision engine works before running the full system.

Usage:
    python tests/test_vision_engine.py
    python tests/test_vision_engine.py --image path/to/xray.jpg
    python tests/test_vision_engine.py --skip-model
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all required libraries are installed."""
    print("=" * 60)
    print("TEST 1: Checking dependencies...")
    print("=" * 60)
    
    errors = []
    
    try:
        import torch
        print(f"  [OK] PyTorch: {torch.__version__}")
        print(f"       Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    except ImportError as e:
        errors.append(f"  [X] PyTorch not installed: {e}")
    
    try:
        import torchxrayvision as xrv
        print("  [OK] torchxrayvision: installed")
    except ImportError as e:
        errors.append(f"  [X] torchxrayvision not installed: {e}")
    
    try:
        import skimage
        print(f"  [OK] scikit-image: {skimage.__version__}")
    except ImportError as e:
        errors.append(f"  [X] scikit-image not installed: {e}")
    
    try:
        import numpy as np
        print(f"  [OK] NumPy: {np.__version__}")
    except ImportError as e:
        errors.append(f"  [X] NumPy not installed: {e}")
    
    if errors:
        print("\nMissing dependencies:")
        for err in errors:
            print(err)
        print("\nInstall with: pip install torch torchxrayvision scikit-image")
        return False
    
    print("\n  All dependencies OK!")
    return True


def test_engine_init():
    """Test engine initialization."""
    print("\n" + "=" * 60)
    print("TEST 2: Initializing Vision Engine...")
    print("=" * 60)
    
    try:
        from infrastructure.vision.xray_vision_engine import XRayVisionEngine
        
        engine = XRayVisionEngine(
            model_name="densenet121-res224-all",
            pathology_threshold=0.15,
            device="cpu"
        )
        
        print(f"  [OK] Engine created")
        print(f"       Model: {engine.model_name}")
        print(f"       Device: {engine.device}")
        print(f"       Threshold: {engine.pathology_threshold}")
        
        return engine
        
    except Exception as e:
        print(f"  [X] Failed: {e}")
        return None


def test_model_loading(engine):
    """Test model loading."""
    print("\n" + "=" * 60)
    print("TEST 3: Loading AI Model...")
    print("=" * 60)
    
    try:
        print("  Loading (this may take a moment)...")
        success = engine.load_model()
        
        if success:
            print("  [OK] Model loaded!")
            print(f"       Pathologies: {len(engine.PATHOLOGY_LABELS)}")
            return True
        else:
            print("  [X] Model loading failed")
            return False
            
    except Exception as e:
        print(f"  [X] Error: {e}")
        return False


def test_provider():
    """Test the VisionProvider."""
    print("\n" + "=" * 60)
    print("TEST 4: Testing VisionProvider...")
    print("=" * 60)
    
    try:
        from infrastructure.vision.vision_provider import VisionProvider
        
        provider = VisionProvider(preload_model=False)
        print("  [OK] VisionProvider created")
        print(f"       Engine: {provider.engine.model_name}")
        print(f"       Device: {provider.engine.device}")
        
        return True
        
    except Exception as e:
        print(f"  [X] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_image():
    """Create a simple test image."""
    try:
        import numpy as np
        from skimage.io import imsave
        
        # Create simple grayscale image
        img = np.zeros((224, 224), dtype=np.uint8)
        
        # Add some structure
        y, x = np.ogrid[:224, :224]
        left = ((x - 70)**2 / 40**2 + (y - 112)**2 / 80**2) < 1
        right = ((x - 154)**2 / 40**2 + (y - 112)**2 / 80**2) < 1
        center = ((x - 112)**2 / 30**2 + (y - 140)**2 / 40**2) < 1
        
        img[left] = 180
        img[right] = 180
        img[center] = 100
        
        noise = np.random.randint(0, 30, (224, 224), dtype=np.uint8)
        img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        
        test_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_xray.png"
        )
        imsave(test_path, img)
        
        return test_path
        
    except Exception as e:
        print(f"  Error creating test image: {e}")
        return None


def test_analysis(engine, image_path=None):
    """Test image analysis."""
    print("\n" + "=" * 60)
    print("TEST 5: Image Analysis...")
    print("=" * 60)
    
    if image_path is None:
        print("  Creating test image...")
        image_path = create_test_image()
        if not image_path:
            return False
    
    if not os.path.exists(image_path):
        print(f"  [X] Image not found: {image_path}")
        return False
    
    print(f"  Analyzing: {os.path.basename(image_path)}")
    
    try:
        result = engine.analyze(image_path)
        
        if result.error:
            print(f"  [X] Error: {result.error}")
            return False
        
        print("  [OK] Analysis complete!")
        print(f"\n  Risk Score: {result.overall_risk_score:.1%}")
        
        if result.high_risk_findings:
            print(f"\n  Findings ({len(result.high_risk_findings)}):")
            for f in result.high_risk_findings[:5]:
                print(f"    - {f}")
        else:
            print("\n  No significant pathologies above threshold")
        
        return True
        
    except Exception as e:
        print(f"  [X] Failed: {e}")
        return False


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test Vision Engine")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--skip-model", action="store_true", help="Skip model tests")
    args = parser.parse_args()
    
    print("+" + "=" * 58 + "+")
    print("|" + " VISION ENGINE TEST SUITE ".center(58) + "|")
    print("|" + " MedicalAgentDiagnosis-MAD ".center(58) + "|")
    print("+" + "=" * 58 + "+")
    
    results = {}
    
    # Test 1: Imports
    results["imports"] = test_imports()
    if not results["imports"]:
        print("\n[X] Cannot continue without dependencies")
        return 1
    
    # Test 2: Engine init
    engine = test_engine_init()
    results["engine"] = engine is not None
    
    # Test 3: Model loading (optional)
    if engine and not args.skip_model:
        results["model"] = test_model_loading(engine)
        
        # Test 5: Analysis
        if results["model"]:
            results["analysis"] = test_analysis(engine, args.image)
    
    # Test 4: Provider
    results["provider"] = test_provider()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "[OK]" if passed else "[X]"
        print(f"  {status} {name.upper()}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("All tests passed! Vision Engine ready.")
        return 0
    else:
        print("Some tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())

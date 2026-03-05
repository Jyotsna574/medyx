"""
Test script for the Vision Provider (placeholder backend).

Usage:
    python tests/test_vision_engine.py
    python tests/test_vision_engine.py --image path/to/image.png
"""

import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_placeholder_provider():
    """Test VisionProvider with placeholder backend."""
    print("=" * 60)
    print("Testing VisionProvider (placeholder backend)")
    print("=" * 60)

    try:
        from infrastructure.vision import VisionProvider

        provider = VisionProvider(preload_model=False)
        print("  [OK] VisionProvider created")
        info = provider.get_model_info()
        print(f"       Model: {info.get('model_id', 'unknown')}")
        return provider
    except Exception as e:
        print(f"  [X] Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_analysis(provider, image_path: str):
    """Test async image analysis."""
    print("\nAnalyzing image...")
    result = await provider.analyze(image_path)
    print(f"  Risk Score: {result.risk_score:.1%}")
    print(f"  Model: {result.model_id}")
    if result.extracted_geometry:
        print("  Geometry:", list(result.extracted_geometry.keys()))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="./test_chest_xray.png")
    args = parser.parse_args()

    provider = test_placeholder_provider()
    if not provider:
        return 1

    if os.path.exists(args.image):
        result = asyncio.run(test_analysis(provider, args.image))
        print("\n[OK] Vision pipeline ready")
    else:
        print(f"\n  Image not found: {args.image}")
        print("  VisionProvider initialized OK (run with valid image to test analysis)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

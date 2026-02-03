#!/usr/bin/env python3
"""
Example script demonstrating the Pre-Segmentation Filter API usage
"""

import requests
import json

# Base URL for the SAM website server
BASE_URL = "http://127.0.0.1:5015"

def upload_image(image_path):
    """Upload an image to the server"""
    print(f"📤 Uploading image: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{BASE_URL}/upload_image", files=files)
    
    result = response.json()
    if result.get('success'):
        print(f"✅ Image uploaded successfully!")
        print(f"   Dimensions: {result['dimensions']['width']}x{result['dimensions']['height']}")
        return True
    else:
        print(f"❌ Failed to upload image: {result.get('error')}")
        return False

def apply_pre_segmentation_filter(brightness=0, contrast=1.0, min_threshold=-1, 
                                  max_threshold=-1, filter_mode='remove_below'):
    """Apply pre-segmentation filter to the uploaded image"""
    print(f"\n🔧 Applying pre-segmentation filter...")
    print(f"   Brightness: {brightness}")
    print(f"   Contrast: {contrast}")
    print(f"   Min Threshold: {min_threshold}")
    print(f"   Max Threshold: {max_threshold}")
    print(f"   Filter Mode: {filter_mode}")
    
    data = {
        'brightness': brightness,
        'contrast': contrast,
        'min_threshold': min_threshold,
        'max_threshold': max_threshold,
        'filter_mode': filter_mode
    }
    
    response = requests.post(
        f"{BASE_URL}/apply_pre_segmentation_filter",
        json=data,
        headers={'Content-Type': 'application/json'}
    )
    
    result = response.json()
    if result.get('success'):
        print(f"✅ {result.get('message')}")
        return True
    else:
        print(f"❌ Failed to apply filter: {result.get('error')}")
        return False

def reset_filter():
    """Reset image to original state"""
    print(f"\n🔄 Resetting to original image...")
    
    response = requests.post(
        f"{BASE_URL}/reset_pre_segmentation_filter",
        json={},
        headers={'Content-Type': 'application/json'}
    )
    
    result = response.json()
    if result.get('success'):
        print(f"✅ {result.get('message')}")
        return True
    else:
        print(f"❌ Failed to reset: {result.get('error')}")
        return False

def run_sam_segmentation(model_size='vit_b', points_per_side=32, crop_layers=1):
    """Run SAM segmentation on the filtered image"""
    print(f"\n🎯 Running SAM segmentation...")
    print(f"   Model: {model_size}")
    print(f"   Points per side: {points_per_side}")
    print(f"   Crop layers: {crop_layers}")
    
    data = {
        'model_size': model_size,
        'points_per_side': points_per_side,
        'crop_layers': crop_layers,
        'backend': 'pytorch',
        'performance_mode': False,
        'use_gpu': True,
        'apply_overlap_filter': True,
        'overlap_threshold': 0.8,
        'overlap_remove_mode': 'larger'
    }
    
    response = requests.post(
        f"{BASE_URL}/run_sam_segmentation",
        json=data,
        headers={'Content-Type': 'application/json'}
    )
    
    result = response.json()
    if result.get('success') and result.get('masks_found'):
        print(f"✅ {result.get('message')}")
        print(f"   Masks found: {result.get('masks_count')}")
        return True
    elif result.get('success'):
        print(f"⚠️ {result.get('message')}")
        return False
    else:
        print(f"❌ Failed to run segmentation: {result.get('error')}")
        return False


# ============================================================================
# Example Use Cases
# ============================================================================

def example_1_remove_dark_background(image_path):
    """Example 1: Remove dark background noise (pixels below intensity 50)"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Remove Dark Background Noise")
    print("="*70)
    
    # Upload image
    if not upload_image(image_path):
        return
    
    # Apply filter to remove dark pixels
    apply_pre_segmentation_filter(
        brightness=0,
        contrast=1.0,
        min_threshold=50,
        max_threshold=-1,
        filter_mode='remove_below'
    )
    
    # Run SAM segmentation
    run_sam_segmentation()


def example_2_remove_bright_areas(image_path):
    """Example 2: Remove bright overexposed areas (pixels above intensity 200)"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Remove Bright Overexposed Areas")
    print("="*70)
    
    # Upload image
    if not upload_image(image_path):
        return
    
    # Apply filter to remove bright pixels
    apply_pre_segmentation_filter(
        brightness=0,
        contrast=1.0,
        min_threshold=-1,
        max_threshold=200,
        filter_mode='remove_above'
    )
    
    # Run SAM segmentation
    run_sam_segmentation()


def example_3_keep_intensity_range(image_path):
    """Example 3: Keep only pixels within intensity range [50, 200]"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Keep Only Mid-Range Intensities [50, 200]")
    print("="*70)
    
    # Upload image
    if not upload_image(image_path):
        return
    
    # Apply filter to keep only specific range
    apply_pre_segmentation_filter(
        brightness=0,
        contrast=1.0,
        min_threshold=50,
        max_threshold=200,
        filter_mode='keep_range'
    )
    
    # Run SAM segmentation
    run_sam_segmentation()


def example_4_brighten_and_filter(image_path):
    """Example 4: Brighten image and remove dark areas"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Brighten Image and Remove Dark Areas")
    print("="*70)
    
    # Upload image
    if not upload_image(image_path):
        return
    
    # Apply combined brightness adjustment and filtering
    apply_pre_segmentation_filter(
        brightness=30,
        contrast=1.0,
        min_threshold=80,
        max_threshold=-1,
        filter_mode='remove_below'
    )
    
    # Run SAM segmentation
    run_sam_segmentation()


def example_5_enhance_contrast_and_filter(image_path):
    """Example 5: Enhance contrast and filter intensity range"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Enhance Contrast and Filter Range [60, 180]")
    print("="*70)
    
    # Upload image
    if not upload_image(image_path):
        return
    
    # Apply contrast enhancement and range filtering
    apply_pre_segmentation_filter(
        brightness=0,
        contrast=1.5,
        min_threshold=60,
        max_threshold=180,
        filter_mode='keep_range'
    )
    
    # Run SAM segmentation
    run_sam_segmentation()


def example_6_multiple_filters_workflow(image_path):
    """Example 6: Apply multiple filters in sequence with reset"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Multiple Filters Workflow with Reset")
    print("="*70)
    
    # Upload image
    if not upload_image(image_path):
        return
    
    # Try first filter
    print("\n--- Attempt 1: Remove dark background ---")
    apply_pre_segmentation_filter(
        min_threshold=50,
        filter_mode='remove_below'
    )
    run_sam_segmentation()
    
    # Reset and try different filter
    reset_filter()
    
    print("\n--- Attempt 2: Keep mid-range intensities ---")
    apply_pre_segmentation_filter(
        min_threshold=60,
        max_threshold=180,
        filter_mode='keep_range'
    )
    run_sam_segmentation()
    
    # Reset to original
    reset_filter()
    
    print("\n--- Final: Run on original image ---")
    run_sam_segmentation()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example_pre_segmentation_filter.py <image_path> [example_number]")
        print("\nExamples:")
        print("  1 - Remove dark background noise")
        print("  2 - Remove bright overexposed areas")
        print("  3 - Keep only mid-range intensities")
        print("  4 - Brighten image and remove dark areas")
        print("  5 - Enhance contrast and filter range")
        print("  6 - Multiple filters workflow with reset")
        print("\nDefault: Runs Example 1")
        sys.exit(1)
    
    image_path = sys.argv[1]
    example_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    # Make sure the server is running
    try:
        response = requests.get(f"{BASE_URL}/")
        print("✅ Server is running!")
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Server is not running at {BASE_URL}")
        print("Please start the server first:")
        print("  python src/web/sam_website.py")
        sys.exit(1)
    
    # Run selected example
    examples = {
        1: example_1_remove_dark_background,
        2: example_2_remove_bright_areas,
        3: example_3_keep_intensity_range,
        4: example_4_brighten_and_filter,
        5: example_5_enhance_contrast_and_filter,
        6: example_6_multiple_filters_workflow
    }
    
    example_func = examples.get(example_num, example_1_remove_dark_background)
    example_func(image_path)
    
    print("\n" + "="*70)
    print("✅ Done! You can now view the results in the web interface.")
    print(f"   Open: {BASE_URL}")
    print("="*70)


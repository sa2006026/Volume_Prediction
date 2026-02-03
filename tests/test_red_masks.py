#!/usr/bin/env python3
"""
Test script to verify the new red mask visualization with dashed lines for removed masks
"""

import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from core.sam_analyzer import SAMAnalyzer

def create_test_image():
    """Create a simple test image with some circles"""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background
    
    # Draw some test circles to segment
    cv2.circle(img, (150, 150), 40, (100, 100, 100), -1)  # Gray circle 1
    cv2.circle(img, (300, 150), 35, (80, 80, 80), -1)     # Gray circle 2  
    cv2.circle(img, (450, 150), 45, (120, 120, 120), -1)  # Gray circle 3
    cv2.circle(img, (200, 280), 30, (90, 90, 90), -1)     # Gray circle 4
    cv2.circle(img, (400, 280), 50, (110, 110, 110), -1)  # Gray circle 5
    
    return img

def test_mask_visualization():
    """Test the red mask visualization"""
    print("🧪 Testing red mask visualization...")
    
    # Create test image
    test_img = create_test_image()
    
    # Save test image
    cv2.imwrite('test_input.png', test_img)
    print("✅ Created test input image: test_input.png")
    
    # Initialize SAM analyzer
    analyzer = SAMAnalyzer()
    analyzer.load_image(test_img)
    
    # Create some mock masks for testing
    print("🔍 Creating mock masks for testing...")
    
    # Create circular masks at known positions
    masks = []
    mask_stats = []
    
    positions = [(150, 150, 40), (300, 150, 35), (450, 150, 45), (200, 280, 30), (400, 280, 50)]
    
    for i, (x, y, r) in enumerate(positions):
        # Create circular mask
        mask = np.zeros((400, 600), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        masks.append(mask > 0)
        
        # Create mock statistics
        area = np.pi * r * r
        stats = {
            'center_x': x,
            'center_y': y,
            'diameter': r * 2,
            'area': area,
            'circularity': 1.0,  # Perfect circles
            'aspect_ratio': 1.0,
            'mask_id': i
        }
        mask_stats.append(stats)
    
    # Set the masks and statistics
    analyzer.masks = masks
    analyzer.mask_statistics = mask_stats
    analyzer.mask_states = ['active'] * len(masks)  # All initially active
    
    # Test 1: All masks active (red, solid)
    print("📸 Test 1: All masks active (red, solid)")
    overlay_active = analyzer.create_mask_overlay(show_labels=True, alpha=0.5)
    cv2.imwrite('test_output_all_active.png', overlay_active)
    print("✅ Saved: test_output_all_active.png")
    
    # Test 2: Some masks removed (red, dashed)
    print("📸 Test 2: Some masks removed (red, dashed)")
    analyzer.mask_states = ['active', 'removed', 'active', 'removed', 'active']
    overlay_mixed = analyzer.create_mask_overlay(show_labels=True, alpha=0.5)
    cv2.imwrite('test_output_mixed_states.png', overlay_mixed)
    print("✅ Saved: test_output_mixed_states.png")
    
    # Test 3: All masks removed (red, dashed)
    print("📸 Test 3: All masks removed (red, dashed)")
    analyzer.mask_states = ['removed'] * len(masks)
    overlay_removed = analyzer.create_mask_overlay(show_labels=True, alpha=0.5)
    cv2.imwrite('test_output_all_removed.png', overlay_removed)
    print("✅ Saved: test_output_all_removed.png")
    
    print("\n🎉 Test completed! Check the output images:")
    print("   - test_input.png: Original test image")
    print("   - test_output_all_active.png: All masks red and solid")
    print("   - test_output_mixed_states.png: Mixed active (solid) and removed (dashed)")
    print("   - test_output_all_removed.png: All masks red and dashed")
    
    return True

if __name__ == '__main__':
    try:
        test_mask_visualization()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

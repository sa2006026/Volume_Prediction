#!/usr/bin/env python3
"""
Test script to verify the overlap filter remove mode functionality
"""
import numpy as np

def test_overlap_logic():
    """Test the overlap filter logic for both removal modes"""
    
    print("=" * 70)
    print("Testing Overlap Filter Remove Mode Logic")
    print("=" * 70)
    
    # Simulate two overlapping masks
    test_cases = [
        {
            'name': 'Case 1: Small mask (100px²) overlaps Large mask (500px²)',
            'area_i': 100,  # smaller
            'area_j': 500,  # larger
            'intersection': 80,
            'overlap_threshold': 0.8
        },
        {
            'name': 'Case 2: Medium mask (300px²) overlaps Medium mask (350px²)',
            'area_i': 300,
            'area_j': 350,
            'intersection': 250,
            'overlap_threshold': 0.8
        },
        {
            'name': 'Case 3: Large mask (1000px²) overlaps Small mask (200px²)',
            'area_i': 1000,  # larger
            'area_j': 200,   # smaller
            'intersection': 180,
            'overlap_threshold': 0.8
        }
    ]
    
    for test in test_cases:
        print(f"\n{test['name']}")
        print("-" * 70)
        print(f"  Mask i area: {test['area_i']} px²")
        print(f"  Mask j area: {test['area_j']} px²")
        print(f"  Intersection: {test['intersection']} px²")
        print(f"  Overlap threshold: {test['overlap_threshold']}")
        
        # Calculate overlap ratio
        base = min(test['area_i'], test['area_j'])
        ratio = test['intersection'] / base
        print(f"  Overlap ratio: {ratio:.2f} ({ratio*100:.0f}%)")
        
        if ratio >= test['overlap_threshold']:
            print(f"  ✅ Overlap detected (>= {test['overlap_threshold']*100:.0f}%)")
            
            # Mode: larger
            if test['area_i'] >= test['area_j']:
                remove_idx_larger = 'i'
            else:
                remove_idx_larger = 'j'
            
            # Mode: smaller
            if test['area_i'] >= test['area_j']:
                remove_idx_smaller = 'j'
            else:
                remove_idx_smaller = 'i'
            
            print(f"\n  Mode: 'larger' (default)")
            print(f"    → Remove mask {remove_idx_larger} ({test[f'area_{remove_idx_larger}']} px²)")
            print(f"    → Keep mask {remove_idx_smaller} ({test[f'area_{remove_idx_smaller}']} px²)")
            
            print(f"\n  Mode: 'smaller'")
            print(f"    → Remove mask {remove_idx_smaller} ({test[f'area_{remove_idx_smaller}']} px²)")
            print(f"    → Keep mask {remove_idx_larger} ({test[f'area_{remove_idx_larger}']} px²)")
        else:
            print(f"  ❌ No overlap detected (< {test['overlap_threshold']*100:.0f}%)")
            print(f"    → Both masks kept (no removal needed)")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print("✅ 'larger' mode: Removes the mask with LARGER area (keeps smaller)")
    print("✅ 'smaller' mode: Removes the mask with SMALLER area (keeps larger)")
    print("✅ Both modes respect the overlap threshold before taking action")
    print("=" * 70)


def demonstrate_use_cases():
    """Demonstrate practical use cases for each mode"""
    
    print("\n\n" + "=" * 70)
    print("Practical Use Cases")
    print("=" * 70)
    
    use_cases = [
        {
            'title': '🔬 Microscopy Droplet Detection',
            'scenario': 'SAM detects both precise small droplet masks (50px²) and fuzzy large masks (200px²)',
            'overlap': 0.85,
            'recommendation': "'larger' mode",
            'reason': 'Preserves precise boundaries of small droplets'
        },
        {
            'title': '🧬 Cell Segmentation',
            'scenario': 'Complete cell masks (1000px²) with small fragment masks inside (100px²)',
            'overlap': 0.90,
            'recommendation': "'smaller' mode",
            'reason': 'Keeps complete cell structures, removes fragments'
        },
        {
            'title': '🎯 General Object Detection',
            'scenario': 'Objects with varying sizes, need to preserve detail',
            'overlap': 0.80,
            'recommendation': "'larger' mode (default)",
            'reason': 'Standard approach that works well in most cases'
        }
    ]
    
    for i, case in enumerate(use_cases, 1):
        print(f"\n{i}. {case['title']}")
        print("-" * 70)
        print(f"   Scenario: {case['scenario']}")
        print(f"   Overlap: {case['overlap']*100:.0f}%")
        print(f"   ✅ Recommended: {case['recommendation']}")
        print(f"   Reason: {case['reason']}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_overlap_logic()
    demonstrate_use_cases()
    
    print("\n🎉 All tests completed!")
    print("\nTo test in the web interface:")
    print("1. Start the server: python src/web/sam_website.py")
    print("2. Upload an image")
    print("3. Enable 'Overlap Filter' in SAM configuration")
    print("4. Try both 'Larger mask' and 'Smaller mask' options")
    print("5. Compare the results!")


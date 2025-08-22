#!/usr/bin/env python3
"""
Test script for the pixel removal server
"""

import requests
import time
import os

def test_server():
    """Test if the server is running and responding"""
    
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Pixel Removal Server...")
    
    # Wait a moment for server to start
    time.sleep(2)
    
    try:
        # Test 1: Check if server responds
        print("1. Testing server connection...")
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ Server is responding!")
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return False
    
    try:
        # Test 2: Test image loading endpoint
        print("2. Testing image loading...")
        test_data = {
            "image_path": "/home/jimmy/code/2Dto3D_2/data/input/BF image/overfocus.jpg"
        }
        
        response = requests.post(f"{base_url}/load_image", 
                               json=test_data, 
                               timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Image loading works!")
                print(f"   Image dimensions: {result.get('dimensions', {})}")
            else:
                print(f"❌ Image loading failed: {result.get('error')}")
                return False
        else:
            print(f"❌ Image loading request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing image loading: {e}")
        return False
    
    print("\n🎉 All tests passed! Server is working correctly.")
    print(f"🌐 Access the application at: {base_url}")
    print("\n🎯 Server Features:")
    print("   • Interactive area selection (Rectangle, Circle, Polygon)")
    print("   • Multiple removal methods (Black, White, Transparent, Blur, Noise)")
    print("   • Real-time preview")
    print("   • Operation history")
    print("   • Save results")
    print("   • Reset to original")
    
    return True

if __name__ == "__main__":
    test_server()

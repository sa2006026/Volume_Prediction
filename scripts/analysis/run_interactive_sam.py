#!/usr/bin/env python3
"""
Startup script for Interactive SAM Segmentation Server
"""

import os
import sys
import subprocess

def main():
    """Start the Interactive SAM server"""
    print("🚀 Starting Interactive SAM Segmentation Server...")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the server script
    server_script = os.path.join(script_dir, 'src', 'web', 'interactive_sam_server.py')
    
    if not os.path.exists(server_script):
        print(f"❌ Server script not found: {server_script}")
        return 1
    
    try:
        # Run the server
        print(f"📂 Server script: {server_script}")
        print("🌐 Server will be available at: http://localhost:5004")
        print("📝 Interface: Interactive SAM Segmentation with Mask Removal")
        print("=" * 60)
        
        # Execute the server script
        subprocess.run([sys.executable, server_script], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

#!/usr/bin/env python3
"""
Installation script for Finnhub integration.

This script helps set up the Finnhub API integration for the Smart ESG Investment Advisor.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a Python package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Main installation function."""
    print("🚀 Installing Finnhub Integration for Smart ESG Investment Advisor")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  No virtual environment detected. Consider creating one:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On Unix/MacOS")
        print("   venv\\Scripts\\activate     # On Windows")
        print()
    
    # Install required packages
    packages = [
        "finnhub-python",
        "fastapi",
        "uvicorn[standard]",
        "pandas",
        "numpy",
        "requests"
    ]
    
    print("📦 Installing required packages...")
    failed_packages = []
    
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Failed to install packages: {', '.join(failed_packages)}")
        print("Please install them manually:")
        for package in failed_packages:
            print(f"   pip install {package}")
        return False
    
    print("\n✅ All packages installed successfully!")
    
    # Check configuration
    print("\n🔧 Checking configuration...")
    config_path = "config/settings.json"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print("Please create the config directory and settings.json file")
        return False
    
    try:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        finnhub_key = config.get("FINNHUB_API_KEY")
        if not finnhub_key or finnhub_key == "your_finnhub_api_key_here":
            print("⚠️  FINNHUB_API_KEY not configured or using placeholder value")
            print("Please update config/settings.json with your actual Finnhub API key")
        else:
            print(f"✅ FINNHUB_API_KEY configured: {finnhub_key[:10]}...")
            
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False
    
    # Test Finnhub connection
    print("\n🧪 Testing Finnhub connection...")
    try:
        import finnhub
        
        if not finnhub_key or finnhub_key == "your_finnhub_api_key_here":
            print("⚠️  Skipping connection test - API key not configured")
        else:
            client = finnhub.Client(api_key=finnhub_key)
            
            # Test a simple API call
            quote = client.quote("AAPL")
            if quote and 'c' in quote:
                print(f"✅ Finnhub connection successful! AAPL price: ${quote['c']:.2f}")
            else:
                print("❌ Finnhub connection test failed")
                return False
                
    except ImportError:
        print("❌ finnhub-python package not available")
        return False
    except Exception as e:
        print(f"❌ Finnhub connection test failed: {e}")
        return False
    
    print("\n🎉 Installation completed successfully!")
    print("\n📋 Next steps:")
    print("1. Update your Finnhub API key in config/settings.json")
    print("2. Run the test script: python backend/test_finnhub.py")
    print("3. Start the backend: cd backend && python main.py")
    print("4. Start the frontend: cd frontend && streamlit run app.py")
    
    print("\n🔗 Useful links:")
    print("- Finnhub API Documentation: https://finnhub.io/docs/api")
    print("- Get your API key: https://finnhub.io/register")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

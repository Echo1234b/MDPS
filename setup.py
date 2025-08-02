"""
MDPS Setup Script
Initializes the environment and checks dependencies
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import MetaTrader5
        import pandas
        import numpy
        import tensorflow
        import torch
        print("✅ Core dependencies found")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

def setup_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed",
        "data/features",
        "data/models",
        "logs",
        "models",
        "configs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def check_mt5_installation():
    """Check MetaTrader 5 installation"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            print("❌ MetaTrader 5 initialization failed")
            print("Please check MT5 installation and credentials in .env file")
            return False
        mt5.shutdown()
        print("✅ MetaTrader 5 connection successful")
        return True
    except Exception as e:
        print(f"❌ MetaTrader 5 error: {e}")
        return False

def setup_environment():
    """Setup environment variables and configurations"""
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        print("Please create .env file with necessary credentials")
        return False
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded")
        return True
    except Exception as e:
        print(f"❌ Environment setup error: {e}")
        return False

def main():
    """Main setup function"""
    print("Starting MDPS Setup...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    # Run setup steps
    check_dependencies()
    setup_directories()
    setup_environment()
    check_mt5_installation()
    
    print("\nSetup Complete! You can now run MDPS.")
    print("\nTo start the system:")
    print("1. Configure your .env file with your credentials")
    print("2. Run 'python -m mdps' to start the system")

if __name__ == "__main__":
    main()

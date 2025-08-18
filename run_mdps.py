#!/usr/bin/env python3
"""
MDPS Runner Script
Entry point for the Market Data Processing System
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Prefer the unified mdps package entrypoint
try:
    from mdps.main import main as mdps_main  # type: ignore
except Exception as e:
    print(f"Failed to import mdps entrypoint: {e}")
    sys.exit(1)

def main():
    """Entry point"""
    print("🚀 Market Data Processing System (MDPS)")
    print("=" * 50)
    
    # Check if we can import everything properly
    try:
        mdps_main()
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        logging.error(f"Startup failed: {e}")

if __name__ == "__main__":
    main()

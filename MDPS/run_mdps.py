#!/usr/bin/env python3
"""
MDPS System Runner
Main entry point for running the MDPS system with command line options
"""

import os
import sys
import argparse
import asyncio
import signal
import logging
from pathlib import Path
from typing import Optional

# Add MDPS root to path
sys.path.insert(0, str(Path(__file__).parent))

from main import MDPSSystem
from config import get_config
from logging import setup_logging
from error_handling import get_error_handler, ErrorCategory, ErrorSeverity

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="MDPS - Market Data Processing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mdps.py                    # Run with default settings
  python run_mdps.py --config custom.yaml  # Use custom config file
  python run_mdps.py --debug            # Enable debug logging
  python run_mdps.py --no-ui            # Run without UI
  python run_mdps.py --backtest         # Run in backtest mode
  python run_mdps.py --validate         # Validate configuration only
        """
    )
    
    # Basic options
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Configuration file path (default: config.yaml)'
    )
    
    parser.add_argument(
        '--log-level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='mdps.log',
        help='Log file path (default: mdps.log)'
    )
    
    # System options
    parser.add_argument(
        '--no-ui',
        action='store_true',
        help='Run without user interface'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run in headless mode (no UI, no interactive prompts)'
    )
    
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run as daemon process'
    )
    
    # Operation modes
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run in backtest mode'
    )
    
    parser.add_argument(
        '--paper-trading',
        action='store_true',
        help='Run in paper trading mode'
    )
    
    parser.add_argument(
        '--live-trading',
        action='store_true',
        help='Run in live trading mode (use with caution)'
    )
    
    parser.add_argument(
        '--data-collection-only',
        action='store_true',
        help='Run only data collection components'
    )
    
    parser.add_argument(
        '--ml-only',
        action='store_true',
        help='Run only ML prediction components'
    )
    
    # Validation and testing
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate configuration and exit'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run system tests and exit'
    )
    
    parser.add_argument(
        '--check-dependencies',
        action='store_true',
        help='Check system dependencies and exit'
    )
    
    # Performance options
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )
    
    parser.add_argument(
        '--memory-limit',
        type=str,
        help='Memory limit (e.g., 2GB, 512MB)'
    )
    
    # Development options
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Enable development mode'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    return parser

def validate_environment() -> bool:
    """Validate system environment"""
    print("Validating system environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        return False
        
    # Check required directories
    required_dirs = ['config', 'data', 'logs', 'models']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            print(f"Creating directory: {dir_name}")
            dir_path.mkdir(exist_ok=True)
            
    # Check configuration file
    config_file = Path('config.yaml')
    if not config_file.exists():
        print("WARNING: config.yaml not found, using default configuration")
        
    print("Environment validation completed")
    return True

def check_dependencies() -> bool:
    """Check system dependencies"""
    print("Checking system dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn',
        'asyncio', 'aiohttp', 'sqlite3', 'yaml', 'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
            
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install missing packages with: pip install " + " ".join(missing_packages))
        return False
        
    print("All required packages are available")
    return True

def setup_signal_handlers(system: MDPSSystem):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        asyncio.create_task(system.shutdown())
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # On Windows, also handle CTRL+C
    if os.name == 'nt':
        signal.signal(signal.CTRL_C_EVENT, signal_handler)

def print_banner():
    """Print MDPS system banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    MDPS - Market Data Processing System      ║
║                                                              ║
║  Advanced market analysis, ML predictions, and trading      ║
║  strategy execution with real-time data processing          ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_system_info():
    """Print system information"""
    print("System Information:")
    print(f"  Python Version: {sys.version}")
    print(f"  Platform: {sys.platform}")
    print(f"  Working Directory: {os.getcwd()}")
    print(f"  MDPS Root: {Path(__file__).parent.absolute()}")
    print()

def run_validation_mode(args) -> int:
    """Run in validation mode"""
    print("Running in validation mode...")
    
    # Validate environment
    if not validate_environment():
        return 1
        
    # Check dependencies
    if not check_dependencies():
        return 1
        
    # Validate configuration
    try:
        config = get_config()
        if config.validate_config():
            print("✓ Configuration validation passed")
        else:
            print("✗ Configuration validation failed")
            return 1
    except Exception as e:
        print(f"✗ Configuration validation error: {e}")
        return 1
        
    print("All validations passed successfully!")
    return 0

def run_test_mode(args) -> int:
    """Run in test mode"""
    print("Running system tests...")
    
    # This would run the actual test suite
    # For now, just run basic functionality tests
    try:
        # Test configuration
        config = get_config()
        print("✓ Configuration loading")
        
        # Test error handling
        error_handler = get_error_handler()
        print("✓ Error handling system")
        
        # Test logging
        logger = setup_logging(level=args.log_level, log_file=args.log_file)
        print("✓ Logging system")
        
        print("Basic system tests passed!")
        return 0
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return 1

async def run_main_system(args) -> int:
    """Run the main MDPS system"""
    try:
        # Initialize system
        system = MDPSSystem()
        
        # Setup signal handlers
        setup_signal_handlers(system)
        
        # Initialize system
        print("Initializing MDPS system...")
        if not await system.initialize():
            print("Failed to initialize system")
            return 1
            
        # Start system
        print("Starting MDPS system...")
        if not await system.start():
            print("Failed to start system")
            return 1
            
        print("MDPS system is running...")
        print("Press Ctrl+C to stop")
        
        # Keep system running
        while system.running:
            await asyncio.sleep(1)
            
        return 0
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
        return 0
    except Exception as e:
        print(f"System error: {e}")
        return 1
    finally:
        if 'system' in locals():
            await system.shutdown()

def main() -> int:
    """Main entry point"""
    try:
        # Parse arguments
        parser = setup_argument_parser()
        args = parser.parse_args()
        
        # Print banner
        print_banner()
        print_system_info()
        
        # Handle special modes
        if args.validate:
            return run_validation_mode(args)
        elif args.test:
            return run_test_mode(args)
        elif args.check_dependencies:
            return 0 if check_dependencies() else 1
            
        # Validate environment
        if not validate_environment():
            return 1
            
        # Setup logging
        logger = setup_logging(level=args.log_level, log_file=args.log_file)
        
        # Log startup
        logger.info("MDPS system starting")
        logger.info(f"Arguments: {vars(args)}")
        
        # Run main system
        return asyncio.run(run_main_system(args))
        
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
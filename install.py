import os
import sys
import venv
import subprocess
from pathlib import Path

def create_virtual_environment():
    """Create a virtual environment for MDPS"""
    print("Creating virtual environment...")
    venv.create("venv", with_pip=True)

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    python_path = os.path.join("venv", "Scripts", "python.exe")
    subprocess.check_call([python_path, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([python_path, "-m", "pip", "install", "-r", "requirements.txt"])

def create_desktop_shortcut():
    """Create desktop shortcut for MDPS"""
    try:
        # Install winshell and pywin32 first
        python_path = os.path.join("venv", "Scripts", "python.exe")
        subprocess.check_call([python_path, "-m", "pip", "install", "winshell", "pywin32"])
        
        import winshell
        from win32com.client import Dispatch
        
        desktop = Path(winshell.desktop())
        path = os.path.abspath("Start_MDPS.bat")
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(str(desktop / "MDPS.lnk"))
        shortcut.Targetpath = path
        shortcut.WorkingDirectory = os.path.dirname(path)
        shortcut.save()
        print("Desktop shortcut created successfully")
    except Exception as e:
        print(f"Warning: Could not create desktop shortcut: {str(e)}")
        print("You can still run the application using Start_MDPS.bat")

def setup_data_directories():
    """Create necessary data directories"""
    directories = [
        "data/raw",
        "data/processed",
        "data/models",
        "logs",
        "config"
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info >= (3, 13) or sys.version_info < (3, 9):
        print("Error: This project requires Python version between 3.9 and 3.12")
        sys.exit(1)
    print(f"Python version {sys.version_info.major}.{sys.version_info.minor} is compatible")

def main():
    try:
        print("Starting MDPS Installation...")
        check_python_version()
        
        # Create virtual environment
        create_virtual_environment()
        
        # Install requirements
        install_requirements()
        
        # Setup directories
        setup_data_directories()
        
        # Create desktop shortcut
        create_desktop_shortcut()
        
        print("\nInstallation complete!")
        print("\nYou can now run MDPS by:")
        print("1. Double-clicking the MDPS shortcut on your desktop")
        print("2. Running Start_MDPS.bat in the installation directory")
        
    except Exception as e:
        print(f"Error during installation: {e}")
        
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()

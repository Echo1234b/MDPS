import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import main window from trading_ui
from trading_ui.ui.main_window import MainWindow
from config import MDPSConfig

def main():
    try:
        # Initialize config
        config = MDPSConfig()
        
        # Create Qt Application
        app = QApplication(sys.argv)
        
        # Set application style and icon
        app.setStyle('Fusion')
        icon_path = os.path.join(project_root, 'UI', 'resources', 'icons', 'mdps.ico')
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
            
        # Create and show main window
        window = MainWindow(config)
        window.show()
        
        # Start application event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        import traceback
        print("Error starting MDPS:")
        print("".join(traceback.format_tb(e.__traceback__)))
        print(f"Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()

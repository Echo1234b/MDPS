import json
from pathlib import Path

class Settings:
    def __init__(self):
        self.config_path = Path(__file__).parent / 'system_config.json'
        self.ui_config_path = Path(__file__).parent / 'ui_config.json'
        self.load_settings()

    def load_settings(self):
        with open(self.config_path) as f:
            self.system_config = json.load(f)
        with open(self.ui_config_path) as f:
            self.ui_config = json.load(f)

    def save_settings(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.system_config, f, indent=4)
        with open(self.ui_config_path, 'w') as f:
            json.dump(self.ui_config, f, indent=4)

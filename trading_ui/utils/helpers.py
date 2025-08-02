from typing import Any, Dict, List
import json
from datetime import datetime

def load_json_file(file_path: str) -> Dict:
    """Load JSON file and return as dictionary"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_file(data: Dict, file_path: str):
    """Save dictionary to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def format_number(number: float, decimals: int = 2) -> str:
    """Format number with specified decimal places"""
    return f"{number:.{decimals}f}"

def get_timestamp() -> str:
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero"""
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return default

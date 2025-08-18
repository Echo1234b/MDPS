#!/usr/bin/env python3
"""
Script to build the complete MDPS directory structure according to the schema
"""

import os
import shutil

def create_directory_structure():
    """Create the complete MDPS directory structure"""
    
    # Base MDPS directory
    base_dir = "MDPS"
    
    # Define the complete directory structure
    structure = {
        "1. Data_Collection_and_Acquisition": {
            "1.1 data_connectivity_feed_integration": [
                "1.1.1",
                "1.1.2", 
                "1.1.3",
                "1.1.4",
                "1.1.5",
                "1.1.6",
                "1.1.7",
                "1.1.8",
                "1.1.9",
                "1.1.10",
                "1.1.11",
                "1.1.12"
            ],
            "1.2 pre_cleaning_preparation": [
                "1.2.1",
                "1.2.2"
            ],
            "1.3 data_validation_integrity_assurance": [
                "1.3.1",
                "1.3.2",
                "1.3.3",
                "1.3.4"
            ],
            "1.4 data_storage_profiling": [
                "1.4.1",
                "1.4.2",
                "1.4.3"
            ],
            "1.5 time_handling_candle_construction": [
                "1.5.1",
                "1.5.2",
                "1.5.3",
                "1.5.4"
            ],
            "1.6 pipeline_orchestration_monitoring": [
                "1.6.1",
                "1.6.2",
                "1.6.3"
            ],
            "1.11 integration_protocols": [
                "1.11.1",
                "1.11.2",
                "1.11.3"
            ]
        },
        "2. External Factors Integration": {
            "2.1 NewsAndEconomicEvents": [
                "2.1.1",
                "2.1.2",
                "2.1.3",
                "2.1.4",
                "2.1.5"
            ],
            "2.2 SocialAndCryptoSentiment": [
                "2.2.1",
                "2.2.2",
                "2.2.3",
                "2.2.4",
                "2.2.5"
            ],
            "2.3 MarketMicrostructureAndCorrelations": [
                "2.3.1",
                "2.3.2",
                "2.3.3"
            ],
            "2.4 BlockchainAndOnChainAnalytics": [
                "2.4.1",
                "2.4.2",
                "2.4.3",
                "2.4.4"
            ],
            "2.5 TimeWeightedEventImpactModel": [
                "2.5.1",
                "2.5.2"
            ],
            "2.10 integration_protocols": [
                "2.10.1",
                "2.10.2",
                "2.10.3"
            ]
        },
        "3. Data Cleaning & Signal Processing": {
            "3.1": [],
            "3.2": [],
            "3.3": [],
            "3.4": [],
            "3.5": [],
            "3.6": [],
            "3.7": [],
            "3.8": [],
            "3.9": [],
            "3.10": [],
            "3.11": [],
            "3.12": [],
            "3.13": [],
            "3.14": [],
            "3.15 integration_protocols": [
                "3.15.1",
                "3.15.2",
                "3.15.3"
            ],
            "3.16": [],
            "3.17": []
        },
        "4. Preprocessing & Feature Engineering": {
            "4.1 indicators": [
                "4.1.1",
                "4.1.2",
                "4.1.3",
                "4.1.4",
                "4.1.5",
                "4.1.6",
                "4.1.7",
                "4.1.8",
                "4.1.9",
                "4.1.10"
            ],
            "4.2 encoders": [
                "4.2.1",
                "4.2.2",
                "4.2.3",
                "4.2.4",
                "4.2.5",
                "4.2.6",
                "4.2.7",
                "4.2.8"
            ],
            "4.3 multi_scale": [
                "4.3.1",
                "4.3.2",
                "4.3.3",
                "4.3.4",
                "4.3.5",
                "4.3.6",
                "4.3.7",
                "4.3.8",
                "4.3.9"
            ],
            "4.4 pattern_recognition": [
                "4.4.1",
                "4.4.2",
                "4.4.3",
                "4.4.4",
                "4.4.5",
                "4.4.6"
            ],
            "4.5 feature_processing": [
                "4.5.1",
                "4.5.2",
                "4.5.3",
                "4.5.4",
                "4.5.5",
                "4.5.6"
            ],
            "4.6 sequence_modeling": [
                "4.6.1",
                "4.6.2",
                "4.6.3"
            ],
            "4.7 versioning": [
                "4.7.1",
                "4.7.2",
                "4.7.3"
            ],
            "4.8": [],
            "4.9": [],
            "4.10": [],
            "4.11": [],
            "4.12": [],
            "4.13": [],
            "4.14 integration_protocols": [
                "4.14.1",
                "4.14.2",
                "4.14.3"
            ],
            "4.15": []
        },
        "5. Market_Context_Structural_Analysis": {
            "5.1 key_zones_levels": [
                "5.1.1",
                "5.1.2",
                "5.1.3",
                "5.1.4"
            ],
            "5.2 liquidity_volume_structure": [
                "5.2.1",
                "5.2.2",
                "5.2.3",
                "5.2.4"
            ],
            "5.3 real_time_market_context": [
                "5.3.1",
                "5.3.2"
            ],
            "5.4 trend_structure_market_regime": [
                "5.4.1",
                "5.4.2",
                "5.4.3",
                "5.4.4",
                "5.4.5",
                "5.4.6"
            ],
            "5.5": [],
            "5.6": [],
            "5.7": [],
            "5.8": [],
            "5.9": [],
            "5.10 integration_protocols": [
                "5.10.1",
                "5.10.2",
                "5.10.3"
            ],
            "5.11": [],
            "5.12": []
        },
        "6. Advanced Chart Analysis Tools": {
            "6.1 advanced_indicators": [
                "6.1.1",
                "6.1.2",
                "6.1.3"
            ],
            "6.2 chart_pattern_detection": [
                "6.2.1",
                "6.2.2",
                "6.2.3",
                "6.2.4",
                "6.2.5"
            ],
            "6.3 elliott_wave_tools": [
                "6.3.1",
                "6.3.2",
                "6.3.3"
            ],
            "6.4 fibonacci_geometric_tools": [
                "6.4.1",
                "6.4.2",
                "6.4.3"
            ],
            "6.5 harmonic_pattern_tools": [
                "6.5.1",
                "6.5.2",
                "6.5.3"
            ],
            "6.6 pattern_signal_fusion": [
                "6.6.1",
                "6.6.2",
                "6.6.3"
            ],
            "6.7 price_action_annotators": [
                "6.7.1",
                "6.7.2",
                "6.7.3"
            ],
            "6.8 support_resistance_tools": [
                "6.8.1",
                "6.8.2",
                "6.8.3",
                "6.8.4",
                "6.8.5"
            ],
            "6.9": [],
            "6.10": [],
            "6.11": [],
            "6.12": [],
            "6.13": [],
            "6.14 integration_protocols": [
                "6.14.1",
                "6.14.2",
                "6.14.3"
            ],
            "6.15": [],
            "6.16": []
        },
        "7. Labeling & Target Engineering": {
            "7.1 label_quality_assessment": [
                "7.1.1",
                "7.1.2",
                "7.1.3"
            ],
            "7.2 label_transformers": [
                "7.2.1",
                "7.2.2",
                "7.2.3"
            ],
            "7.3 target_generators": [
                "7.3.1",
                "7.3.2",
                "7.3.3",
                "7.3.4"
            ],
            "7.4": [],
            "7.5": [],
            "7.6": [],
            "7.7": [],
            "7.8": [],
            "7.9 integration_protocols": [
                "7.9.1",
                "7.9.2",
                "7.9.3"
            ],
            "7.10": [],
            "7.11": []
        },
        "8. Prediction Engine (MLDL Models)": {
            "8.1 cnn_models": [
                "8.1.1",
                "8.1.2",
                "8.1.3"
            ],
            "8.2 ensemble_models": [
                "8.2.1",
                "8.2.2",
                "8.2.3",
                "8.2.4"
            ],
            "8.3 model_management": [
                "8.3.1",
                "8.3.2",
                "8.3.3"
            ],
            "8.4 reinforcement_learning": [
                "8.4.1",
                "8.4.2",
                "8.4.3",
                "8.4.4"
            ],
            "8.5 sequence_models": [
                "8.5.1",
                "8.5.2",
                "8.5.3",
                "8.5.4",
                "8.5.5",
                "8.5.6"
            ],
            "8.6 traditional_ml": [
                "8.6.1",
                "8.6.2",
                "8.6.3",
                "8.6.4",
                "8.6.5"
            ],
            "8.7 training_utils": [
                "8.7.1",
                "8.7.2",
                "8.7.3",
                "8.7.4"
            ],
            "8.8 transformer_models": [
                "8.8.1",
                "8.8.2"
            ],
            "8.9": [],
            "8.10": [],
            "8.11": [],
            "8.12": [],
            "8.13": [],
            "8.14 integration_protocols": [
                "8.14.1",
                "8.14.2",
                "8.14.3",
                "8.14.4",
                "8.14.5",
                "8.14.6"
            ],
            "8.15": [],
            "8.16": [],
            "8.17": [],
            "8.18": [],
            "8.19": [],
            "8.20": [],
            "8.21": [],
            "8.22": [],
            "8.23": []
        },
        "9. Strategy & Decision Layer": {
            "9.1 risk_management": [
                "9.1.1",
                "9.1.2",
                "9.1.3",
                "9.1.4"
            ],
            "9.2 signal_validation": [
                "9.2.1",
                "9.2.2",
                "9.2.3",
                "9.2.4"
            ],
            "9.3 simulation_analysis": [
                "9.3.1",
                "9.3.2",
                "9.3.3",
                "9.3.4",
                "9.3.5",
                "9.3.6",
                "9.3.7",
                "9.3.8"
            ],
            "9.4 strategy_selection": [
                "9.4.1",
                "9.4.2",
                "9.4.3",
                "9.4.4"
            ],
            "9.5 timing_execution": [
                "9.5.1",
                "9.5.2"
            ],
            "9.6": [],
            "9.7": [],
            "9.8": [],
            "9.9": [],
            "9.10": [],
            "9.11": [],
            "9.12": [],
            "9.13 integration_protocols": [
                "9.13.1",
                "9.13.2",
                "9.13.3"
            ],
            "9.14": [],
            "9.15": [],
            "9.16": [],
            "9.17": [],
            "9.18": [],
            "9.19": [],
            "9.20": [],
            "9.21": []
        },
        "10. trading_ui": {
            "10.1 config": [
                "10.1.1",
                "10.1.2",
                "10.1.3 localization",
                "10.1.4"
            ],
            "10.2 core": [
                "10.2.1",
                "10.2.2",
                "10.2.3",
                "10.2.4",
                "10.2.5",
                "10.2.6",
                "10.2.7",
                "10.2.8",
                "10.2.9",
                "10.2.10",
                "10.2.11"
            ],
            "10.3 data": [
                "10.3.1",
                "10.3.2",
                "10.3.3 models",
                "10.3.4",
                "10.3.5",
                "10.3.6"
            ],
            "10.4 services": [
                "10.4.1",
                "10.4.2",
                "10.4.3",
                "10.4.4",
                "10.4.5",
                "10.4.6",
                "10.4.7",
                "10.4.8",
                "10.4.9",
                "10.4.10",
                "10.4.11",
                "10.4.12",
                "10.4.13",
                "10.4.14",
                "10.4.15",
                "10.4.16"
            ],
            "10.5 tests": [
                "10.5.1",
                "10.5.2",
                "10.5.3",
                "10.5.4"
            ],
            "10.6 ui": [
                "10.6.1",
                "10.6.2 resources",
                "10.6.3 utils",
                "10.6.4 views",
                "10.6.5 widgets",
                "10.6.6 ui_components",
                "10.6.7 navigation_system"
            ],
            "10.7 utils": [
                "10.7.1",
                "10.7.2",
                "10.7.3",
                "10.7.4",
                "10.7.5",
                "10.7.6",
                "10.7.7",
                "10.7.8",
                "10.7.9",
                "10.7.10",
                "10.7.11",
                "10.7.12",
                "10.7.13",
                "10.7.14",
                "10.7.15",
                "10.7.16"
            ],
            "10.8": [],
            "10.9": [],
            "10.10": [],
            "10.11 Data_Visualization_Engine": {
                "10.11.1 data_insights_visualization": [
                    "10.11.1.1",
                    "10.11.1.2",
                    "10.11.1.3",
                    "10.11.1.4",
                    "10.11.1.5",
                    "10.11.1.6",
                    "10.11.1.7",
                    "10.11.1.8",
                    "10.11.1.9",
                    "10.11.1.10",
                    "10.11.1.11",
                    "10.11.1.12"
                ],
                "10.11.2 visualization_components": [
                    "10.11.2.1",
                    "10.11.2.2",
                    "10.11.2.3",
                    "10.11.2.4",
                    "10.11.2.5"
                ],
                "10.11.3 dashboard_framework": [
                    "10.11.3.1",
                    "10.11.3.2",
                    "10.11.3.3",
                    "10.11.3.4"
                ],
                "10.11.4 performance_optimization": [
                    "10.11.4.1",
                    "10.11.4.2",
                    "10.11.4.3"
                ],
                "10.11.5 integration_interface": [
                    "10.11.5.1",
                    "10.11.5.2",
                    "10.11.5.3"
                ]
            }
        },
        "11. System_Orchestration_Control": {
            "11.1": [],
            "11.2": [],
            "11.3": [],
            "11.4": [],
            "11.5": [],
            "11.6": [],
            "11.7": [],
            "11.8": [],
            "11.9": [],
            "11.10": [],
            "11.11 integration_protocols": [
                "11.11.1",
                "11.11.2",
                "11.11.3"
            ],
            "11.12": [],
            "11.13": []
        },
        "12. Enhanced_UI_Framework": {
            "12.1 main_application_window": [
                "12.1.1",
                "12.1.2",
                "12.1.3",
                "12.1.4"
            ],
            "12.2 system_control_panel": [
                "12.2.1",
                "12.2.2",
                "12.2.3",
                "12.2.4"
            ],
            "12.3 advanced_charting_system": [
                "12.3.1",
                "12.3.2",
                "12.3.3",
                "12.3.4",
                "12.3.5"
            ],
            "12.4 trading_interface": [
                "12.4.1",
                "12.4.2",
                "12.4.3",
                "12.4.4",
                "12.4.5"
            ],
            "12.5 configuration_management": [
                "12.5.1",
                "12.5.2",
                "12.5.3",
                "12.5.4"
            ],
            "12.6": [],
            "12.7": [],
            "12.8": [],
            "12.9": [],
            "12.10": [],
            "12.11": [],
            "12.12": []
        },
        "13. System_Integration_Layer": {
            "13.1": [],
            "13.2": [],
            "13.3": [],
            "13.4": [],
            "13.5": [],
            "13.6": [],
            "13.7": [],
            "13.8 integration_protocols": [
                "13.8.1",
                "13.8.2",
                "13.8.3"
            ],
            "13.9": [],
            "13.10": []
        },
        "14. System_Configuration": {
            "14.1": [],
            "14.2": [],
            "14.3": [],
            "14.4": [],
            "14.5": [],
            "14.6": [],
            "14.7": [],
            "14.8 integration_protocols": [
                "14.8.1",
                "14.8.2",
                "14.8.3"
            ],
            "14.9": [],
            "14.10": []
        }
    }
    
    def create_dirs(base_path, struct):
        """Recursively create directories"""
        for key, value in struct.items():
            if isinstance(value, list):
                # Create directory and subdirectories
                dir_path = os.path.join(base_path, key)
                os.makedirs(dir_path, exist_ok=True)
                for subdir in value:
                    subdir_path = os.path.join(dir_path, subdir)
                    os.makedirs(subdir_path, exist_ok=True)
            elif isinstance(value, dict):
                # Create directory and recurse
                dir_path = os.path.join(base_path, key)
                os.makedirs(dir_path, exist_ok=True)
                create_dirs(dir_path, value)
            else:
                # Create single directory
                dir_path = os.path.join(base_path, key)
                os.makedirs(dir_path, exist_ok=True)
    
    # Create the base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create the structure
    create_dirs(base_dir, structure)
    
    print(f"Created MDPS directory structure in {base_dir}/")

if __name__ == "__main__":
    create_directory_structure()
"""
Custom exceptions for the Strategy & Decision Layer.
"""

class StrategyDecisionError(Exception):
    """Base exception for strategy decision layer"""
    pass

class SignalValidationError(StrategyDecisionError):
    """Exception raised for signal validation errors"""
    def __init__(self, message: str, signal_details: dict = None):
        super().__init__(message)
        self.signal_details = signal_details or {}

class RiskManagementError(StrategyDecisionError):
    """Exception raised for risk management errors"""
    def __init__(self, message: str, risk_parameters: dict = None):
        super().__init__(message)
        self.risk_parameters = risk_parameters or {}

class StrategySelectionError(StrategyDecisionError):
    """Exception raised for strategy selection errors"""
    def __init__(self, message: str, available_strategies: list = None):
        super().__init__(message)
        self.available_strategies = available_strategies or []

class ExecutionError(StrategyDecisionError):
    """Exception raised for execution errors"""
    def __init__(self, message: str, execution_details: dict = None):
        super().__init__(message)
        self.execution_details = execution_details or {}

class SimulationError(StrategyDecisionError):
    """Exception raised for simulation errors"""
    def __init__(self, message: str, simulation_parameters: dict = None):
        super().__init__(message)
        self.simulation_parameters = simulation_parameters or {}

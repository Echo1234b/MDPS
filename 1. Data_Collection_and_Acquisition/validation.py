"""
Validation Module for Data Collection and Acquisition Section

Validates final processed data, applies validation rules, performs integrity checks,
and generates validation status reports with error/warning details.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib
from abc import ABC, abstractmethod

class ValidationLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationStatus(Enum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ValidationIssue:
    """Individual validation issue"""
    rule_name: str
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    expected: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ValidationResult:
    """Validation result for a dataset"""
    status: ValidationStatus
    issues: List[ValidationIssue]
    passed_rules: int
    failed_rules: int
    warning_rules: int
    total_rules: int
    execution_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class ValidationRule(ABC):
    """Abstract base class for validation rules"""
    
    def __init__(self, name: str, level: ValidationLevel = ValidationLevel.ERROR):
        self.name = name
        self.level = level
    
    @abstractmethod
    def validate(self, data: Any, context: Dict[str, Any] = None) -> List[ValidationIssue]:
        """Validate data and return issues"""
        pass

class DataTypeValidationRule(ValidationRule):
    """Validates data types"""
    
    def __init__(self, expected_types: Dict[str, type], 
                 level: ValidationLevel = ValidationLevel.ERROR):
        super().__init__("data_type_validation", level)
        self.expected_types = expected_types
    
    def validate(self, data: Any, context: Dict[str, Any] = None) -> List[ValidationIssue]:
        issues = []
        
        if isinstance(data, pd.DataFrame):
            for column, expected_type in self.expected_types.items():
                if column in data.columns:
                    actual_type = data[column].dtype
                    if not self._is_compatible_type(actual_type, expected_type):
                        issues.append(ValidationIssue(
                            rule_name=self.name,
                            level=self.level,
                            message=f"Column '{column}' has type {actual_type}, expected {expected_type}",
                            field=column,
                            value=str(actual_type),
                            expected=str(expected_type)
                        ))
        
        return issues
    
    def _is_compatible_type(self, actual, expected) -> bool:
        """Check if types are compatible"""
        type_mapping = {
            float: [np.float32, np.float64, 'float32', 'float64'],
            int: [np.int32, np.int64, 'int32', 'int64'],
            str: ['object', 'string'],
            bool: ['bool']
        }
        
        if expected in type_mapping:
            return str(actual) in [str(t) for t in type_mapping[expected]]
        
        return str(actual) == str(expected)

class RangeValidationRule(ValidationRule):
    """Validates numeric ranges"""
    
    def __init__(self, field_ranges: Dict[str, Tuple[float, float]], 
                 level: ValidationLevel = ValidationLevel.ERROR):
        super().__init__("range_validation", level)
        self.field_ranges = field_ranges
    
    def validate(self, data: Any, context: Dict[str, Any] = None) -> List[ValidationIssue]:
        issues = []
        
        if isinstance(data, pd.DataFrame):
            for field, (min_val, max_val) in self.field_ranges.items():
                if field in data.columns:
                    out_of_range = data[(data[field] < min_val) | (data[field] > max_val)]
                    if not out_of_range.empty:
                        issues.append(ValidationIssue(
                            rule_name=self.name,
                            level=self.level,
                            message=f"Field '{field}' has {len(out_of_range)} values outside range [{min_val}, {max_val}]",
                            field=field,
                            value=f"{len(out_of_range)} violations",
                            expected=f"[{min_val}, {max_val}]"
                        ))
        
        return issues

class NullValidationRule(ValidationRule):
    """Validates null/missing values"""
    
    def __init__(self, required_fields: List[str], 
                 max_null_percentage: float = 0.0,
                 level: ValidationLevel = ValidationLevel.ERROR):
        super().__init__("null_validation", level)
        self.required_fields = required_fields
        self.max_null_percentage = max_null_percentage
    
    def validate(self, data: Any, context: Dict[str, Any] = None) -> List[ValidationIssue]:
        issues = []
        
        if isinstance(data, pd.DataFrame):
            for field in self.required_fields:
                if field in data.columns:
                    null_count = data[field].isnull().sum()
                    null_percentage = null_count / len(data) * 100
                    
                    if null_percentage > self.max_null_percentage:
                        issues.append(ValidationIssue(
                            rule_name=self.name,
                            level=self.level,
                            message=f"Field '{field}' has {null_percentage:.2f}% null values, maximum allowed is {self.max_null_percentage}%",
                            field=field,
                            value=f"{null_percentage:.2f}%",
                            expected=f"<= {self.max_null_percentage}%"
                        ))
        
        return issues

class DuplicateValidationRule(ValidationRule):
    """Validates duplicate records"""
    
    def __init__(self, key_fields: List[str], 
                 level: ValidationLevel = ValidationLevel.WARNING):
        super().__init__("duplicate_validation", level)
        self.key_fields = key_fields
    
    def validate(self, data: Any, context: Dict[str, Any] = None) -> List[ValidationIssue]:
        issues = []
        
        if isinstance(data, pd.DataFrame):
            if all(field in data.columns for field in self.key_fields):
                duplicates = data.duplicated(subset=self.key_fields, keep=False)
                duplicate_count = duplicates.sum()
                
                if duplicate_count > 0:
                    issues.append(ValidationIssue(
                        rule_name=self.name,
                        level=self.level,
                        message=f"Found {duplicate_count} duplicate records based on key fields {self.key_fields}",
                        field=str(self.key_fields),
                        value=str(duplicate_count),
                        expected="0"
                    ))
        
        return issues

class TimeSeriesValidationRule(ValidationRule):
    """Validates time series data"""
    
    def __init__(self, timestamp_field: str, 
                 expected_frequency: Optional[str] = None,
                 allow_gaps: bool = True,
                 max_gap_duration: Optional[timedelta] = None,
                 level: ValidationLevel = ValidationLevel.WARNING):
        super().__init__("timeseries_validation", level)
        self.timestamp_field = timestamp_field
        self.expected_frequency = expected_frequency
        self.allow_gaps = allow_gaps
        self.max_gap_duration = max_gap_duration
    
    def validate(self, data: Any, context: Dict[str, Any] = None) -> List[ValidationIssue]:
        issues = []
        
        if isinstance(data, pd.DataFrame) and self.timestamp_field in data.columns:
            # Check if timestamps are sorted
            if not data[self.timestamp_field].is_monotonic_increasing:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    level=self.level,
                    message=f"Timestamps in '{self.timestamp_field}' are not sorted in ascending order",
                    field=self.timestamp_field
                ))
            
            # Check for gaps if not allowed
            if not self.allow_gaps and len(data) > 1:
                time_diffs = data[self.timestamp_field].diff().dropna()
                
                if self.expected_frequency:
                    expected_diff = pd.Timedelta(self.expected_frequency)
                    gaps = time_diffs[time_diffs > expected_diff * 1.5]  # Allow 50% tolerance
                    
                    if not gaps.empty:
                        issues.append(ValidationIssue(
                            rule_name=self.name,
                            level=self.level,
                            message=f"Found {len(gaps)} time gaps larger than expected frequency {self.expected_frequency}",
                            field=self.timestamp_field,
                            value=f"{len(gaps)} gaps",
                            expected=f"No gaps > {expected_diff * 1.5}"
                        ))
                
                # Check maximum gap duration
                if self.max_gap_duration:
                    large_gaps = time_diffs[time_diffs > self.max_gap_duration]
                    
                    if not large_gaps.empty:
                        issues.append(ValidationIssue(
                            rule_name=self.name,
                            level=self.level,
                            message=f"Found {len(large_gaps)} time gaps larger than maximum allowed duration",
                            field=self.timestamp_field,
                            value=f"Max gap: {large_gaps.max()}",
                            expected=f"<= {self.max_gap_duration}"
                        ))
        
        return issues

class BusinessLogicValidationRule(ValidationRule):
    """Validates business logic constraints"""
    
    def __init__(self, validation_function: Callable[[Any], List[ValidationIssue]],
                 name: str, level: ValidationLevel = ValidationLevel.ERROR):
        super().__init__(name, level)
        self.validation_function = validation_function
    
    def validate(self, data: Any, context: Dict[str, Any] = None) -> List[ValidationIssue]:
        try:
            return self.validation_function(data)
        except Exception as e:
            return [ValidationIssue(
                rule_name=self.name,
                level=ValidationLevel.ERROR,
                message=f"Business logic validation failed: {str(e)}"
            )]

class DataCollectionValidator:
    """
    Comprehensive data validator for the Data Collection and Acquisition section
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules: List[ValidationRule] = []
        self.validation_history: List[ValidationResult] = []
        
        # Add default validation rules
        self._add_default_rules()
    
    def _add_default_rules(self):
        """Add default validation rules for financial data"""
        
        # OHLCV data validation
        ohlcv_types = {
            'open': float,
            'high': float,
            'low': float,
            'close': float,
            'volume': float
        }
        self.add_rule(DataTypeValidationRule(ohlcv_types, ValidationLevel.ERROR))
        
        # Price ranges (assuming reasonable ranges for most assets)
        price_ranges = {
            'open': (0.0, 1000000.0),
            'high': (0.0, 1000000.0),
            'low': (0.0, 1000000.0),
            'close': (0.0, 1000000.0),
            'volume': (0.0, float('inf'))
        }
        self.add_rule(RangeValidationRule(price_ranges, ValidationLevel.ERROR))
        
        # Required fields validation
        self.add_rule(NullValidationRule(['open', 'high', 'low', 'close'], 0.0, ValidationLevel.ERROR))
        self.add_rule(NullValidationRule(['volume'], 5.0, ValidationLevel.WARNING))  # Allow some missing volume
        
        # Time series validation
        self.add_rule(TimeSeriesValidationRule('timestamp', allow_gaps=True, 
                                              max_gap_duration=timedelta(hours=24),
                                              level=ValidationLevel.WARNING))
        
        # Business logic: High >= Low, High >= Open/Close, Low <= Open/Close
        def ohlc_logic_validation(data):
            issues = []
            if isinstance(data, pd.DataFrame):
                required_cols = ['open', 'high', 'low', 'close']
                if all(col in data.columns for col in required_cols):
                    # High should be >= all other prices
                    high_violations = data[(data['high'] < data['open']) | 
                                          (data['high'] < data['low']) | 
                                          (data['high'] < data['close'])]
                    if not high_violations.empty:
                        issues.append(ValidationIssue(
                            rule_name="ohlc_logic",
                            level=ValidationLevel.ERROR,
                            message=f"Found {len(high_violations)} records where High < other prices",
                            value=str(len(high_violations)),
                            expected="0"
                        ))
                    
                    # Low should be <= all other prices
                    low_violations = data[(data['low'] > data['open']) | 
                                         (data['low'] > data['high']) | 
                                         (data['low'] > data['close'])]
                    if not low_violations.empty:
                        issues.append(ValidationIssue(
                            rule_name="ohlc_logic",
                            level=ValidationLevel.ERROR,
                            message=f"Found {len(low_violations)} records where Low > other prices",
                            value=str(len(low_violations)),
                            expected="0"
                        ))
            
            return issues
        
        self.add_rule(BusinessLogicValidationRule(ohlc_logic_validation, "ohlc_logic", ValidationLevel.ERROR))
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule"""
        self.rules.append(rule)
        self.logger.debug(f"Added validation rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a validation rule by name"""
        original_count = len(self.rules)
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        removed = len(self.rules) < original_count
        
        if removed:
            self.logger.debug(f"Removed validation rule: {rule_name}")
        
        return removed
    
    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """
        Validate data against all rules
        
        Args:
            data: Data to validate
            context: Additional context for validation
            
        Returns:
            ValidationResult: Validation results
        """
        start_time = datetime.now()
        all_issues = []
        passed_rules = 0
        failed_rules = 0
        warning_rules = 0
        
        try:
            for rule in self.rules:
                try:
                    issues = rule.validate(data, context)
                    
                    if issues:
                        all_issues.extend(issues)
                        
                        # Count by severity
                        has_error = any(issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL] 
                                      for issue in issues)
                        has_warning = any(issue.level == ValidationLevel.WARNING 
                                        for issue in issues)
                        
                        if has_error:
                            failed_rules += 1
                        elif has_warning:
                            warning_rules += 1
                        else:
                            passed_rules += 1
                    else:
                        passed_rules += 1
                        
                except Exception as e:
                    self.logger.error(f"Error in validation rule {rule.name}: {str(e)}")
                    all_issues.append(ValidationIssue(
                        rule_name=rule.name,
                        level=ValidationLevel.CRITICAL,
                        message=f"Validation rule execution failed: {str(e)}"
                    ))
                    failed_rules += 1
            
            # Determine overall status
            if any(issue.level == ValidationLevel.CRITICAL for issue in all_issues):
                status = ValidationStatus.FAILED
            elif failed_rules > 0:
                status = ValidationStatus.FAILED
            elif warning_rules > 0:
                status = ValidationStatus.WARNING
            else:
                status = ValidationStatus.PASSED
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = ValidationResult(
                status=status,
                issues=all_issues,
                passed_rules=passed_rules,
                failed_rules=failed_rules,
                warning_rules=warning_rules,
                total_rules=len(self.rules),
                execution_time=execution_time,
                timestamp=datetime.now(),
                metadata={
                    'data_type': str(type(data)),
                    'data_shape': getattr(data, 'shape', None),
                    'context': context or {}
                }
            )
            
            # Store in history
            self.validation_history.append(result)
            
            # Log results
            self.logger.info(f"Validation completed: {status.value}, "
                           f"{failed_rules} failed, {warning_rules} warnings, "
                           f"{passed_rules} passed ({execution_time:.3f}s)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Validation process failed: {str(e)}")
            return ValidationResult(
                status=ValidationStatus.FAILED,
                issues=[ValidationIssue(
                    rule_name="validation_process",
                    level=ValidationLevel.CRITICAL,
                    message=f"Validation process failed: {str(e)}"
                )],
                passed_rules=0,
                failed_rules=1,
                warning_rules=0,
                total_rules=1,
                execution_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now()
            )
    
    def validate_symbol_data(self, symbol: str, data: pd.DataFrame, 
                           data_type: str = "ohlcv") -> ValidationResult:
        """
        Validate symbol-specific data
        
        Args:
            symbol: Trading symbol
            data: Data to validate
            data_type: Type of data (ohlcv, tick, etc.)
            
        Returns:
            ValidationResult: Validation results
        """
        context = {
            'symbol': symbol,
            'data_type': data_type,
            'record_count': len(data) if hasattr(data, '__len__') else None
        }
        
        return self.validate(data, context)
    
    def get_validation_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Get validation summary statistics
        
        Args:
            last_n: Number of recent validations to include
            
        Returns:
            Dict: Summary statistics
        """
        if not self.validation_history:
            return {'total_validations': 0}
        
        recent_results = self.validation_history[-last_n:] if last_n else self.validation_history
        
        total_validations = len(recent_results)
        passed_validations = sum(1 for r in recent_results if r.status == ValidationStatus.PASSED)
        warning_validations = sum(1 for r in recent_results if r.status == ValidationStatus.WARNING)
        failed_validations = sum(1 for r in recent_results if r.status == ValidationStatus.FAILED)
        
        avg_execution_time = np.mean([r.execution_time for r in recent_results])
        
        # Issue statistics
        all_issues = [issue for result in recent_results for issue in result.issues]
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue.rule_name] = issue_counts.get(issue.rule_name, 0) + 1
        
        return {
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'warning_validations': warning_validations,
            'failed_validations': failed_validations,
            'success_rate': passed_validations / total_validations if total_validations > 0 else 0,
            'avg_execution_time': avg_execution_time,
            'total_issues': len(all_issues),
            'issue_counts_by_rule': issue_counts,
            'active_rules': len(self.rules)
        }
    
    def export_validation_report(self, result: ValidationResult, 
                                format: str = "dict") -> Union[Dict, str]:
        """
        Export validation report in specified format
        
        Args:
            result: Validation result to export
            format: Export format (dict, json, html)
            
        Returns:
            Formatted report
        """
        if format == "dict":
            return {
                'status': result.status.value,
                'summary': {
                    'passed_rules': result.passed_rules,
                    'failed_rules': result.failed_rules,
                    'warning_rules': result.warning_rules,
                    'total_rules': result.total_rules,
                    'execution_time': result.execution_time,
                    'timestamp': result.timestamp.isoformat()
                },
                'issues': [
                    {
                        'rule_name': issue.rule_name,
                        'level': issue.level.value,
                        'message': issue.message,
                        'field': issue.field,
                        'value': str(issue.value) if issue.value is not None else None,
                        'expected': str(issue.expected) if issue.expected is not None else None,
                        'timestamp': issue.timestamp.isoformat()
                    }
                    for issue in result.issues
                ],
                'metadata': result.metadata
            }
        
        elif format == "json":
            import json
            return json.dumps(self.export_validation_report(result, "dict"), indent=2)
        
        elif format == "html":
            return self._generate_html_report(result)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _generate_html_report(self, result: ValidationResult) -> str:
        """Generate HTML validation report"""
        status_color = {
            ValidationStatus.PASSED: "green",
            ValidationStatus.WARNING: "orange", 
            ValidationStatus.FAILED: "red",
            ValidationStatus.SKIPPED: "gray"
        }
        
        level_color = {
            ValidationLevel.INFO: "blue",
            ValidationLevel.WARNING: "orange",
            ValidationLevel.ERROR: "red",
            ValidationLevel.CRITICAL: "darkred"
        }
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {status_color[result.status]}; color: white; padding: 10px; }}
                .summary {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; }}
                .issue {{ margin: 10px 0; padding: 10px; border-left: 4px solid; }}
                .info {{ border-left-color: {level_color[ValidationLevel.INFO]}; }}
                .warning {{ border-left-color: {level_color[ValidationLevel.WARNING]}; }}
                .error {{ border-left-color: {level_color[ValidationLevel.ERROR]}; }}
                .critical {{ border-left-color: {level_color[ValidationLevel.CRITICAL]}; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Validation Report</h1>
                <p>Status: {result.status.value.upper()}</p>
                <p>Timestamp: {result.timestamp}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <ul>
                    <li>Total Rules: {result.total_rules}</li>
                    <li>Passed: {result.passed_rules}</li>
                    <li>Failed: {result.failed_rules}</li>
                    <li>Warnings: {result.warning_rules}</li>
                    <li>Execution Time: {result.execution_time:.3f} seconds</li>
                    <li>Total Issues: {len(result.issues)}</li>
                </ul>
            </div>
            
            <div class="issues">
                <h2>Issues ({len(result.issues)})</h2>
        """
        
        for issue in result.issues:
            html += f"""
                <div class="issue {issue.level.value}">
                    <h3>{issue.rule_name} - {issue.level.value.upper()}</h3>
                    <p><strong>Message:</strong> {issue.message}</p>
            """
            
            if issue.field:
                html += f"<p><strong>Field:</strong> {issue.field}</p>"
            if issue.value is not None:
                html += f"<p><strong>Value:</strong> {issue.value}</p>"
            if issue.expected is not None:
                html += f"<p><strong>Expected:</strong> {issue.expected}</p>"
            
            html += f"<p><strong>Timestamp:</strong> {issue.timestamp}</p></div>"
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def clear_history(self):
        """Clear validation history"""
        self.validation_history.clear()
        self.logger.info("Validation history cleared")
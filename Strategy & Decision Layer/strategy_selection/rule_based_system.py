"""
Rule-Based System implementation.
"""

from typing import Dict, Any, List, Callable
from ..exceptions import StrategySelectionError

class RuleBasedSystem:
    def __init__(self):
        self.rules = {}
        self.rule_conditions = {}
    
    def add_rule(self, name: str, condition: Callable[[Dict[str, Any]], bool], action: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a new rule to the system
        Args:
            name: Rule name
            condition: Function that evaluates if rule should trigger
            action: Function to execute when rule triggers
        """
        self.rules[name] = {
            'condition': condition,
            'action': action
        }
    
    def apply_rules(self, signal: Dict[str, Any], market_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply all rules to signal and context
        Args:
            signal: Trading signal
            market_context: Market context data
        Returns:
           
        Returns:
            list: List of triggered rule actions
        """
        triggered_actions = []
        combined_context = {**signal, **market_context}
        
        try:
            for rule_name, rule in self.rules.items():
                if rule['condition'](combined_context):
                    action_result = rule['action'](combined_context)
                    triggered_actions.append({
                        'rule': rule_name,
                        'action': action_result
                    })
            
            return triggered_actions
            
        except Exception as e:
            raise StrategySelectionError(f"Rule application failed: {str(e)}")
    
    def update_rules(self, new_rules: Dict[str, Any]) -> None:
        """
        Update trading rules
        Args:
            new_rules: Dictionary of new rules
        """
        try:
            for rule_name, rule_config in new_rules.items():
                if rule_name in self.rules:
                    # Update existing rule
                    self.rules[rule_name].update(rule_config)
                else:
                    # Add new rule
                    self.add_rule(
                        rule_name,
                        rule_config['condition'],
                        rule_config['action']
                    )
        except Exception as e:
            raise StrategySelectionError(f"Rule update failed: {str(e)}")
    
    def remove_rule(self, rule_name: str) -> None:
        """
        Remove a rule from the system
        Args:
            rule_name: Name of rule to remove
        """
        if rule_name in self.rules:
            del self.rules[rule_name]
    
    def get_active_rules(self) -> List[str]:
        """
        Get list of active rule names
        Returns:
            list: List of active rule names
        """
        return list(self.rules.keys())

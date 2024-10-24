"""
Rule-based engine providing alternatives to ML-based features.
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

@dataclass
class LoadProfile:
    """Predefined load profiles for different day types."""
    weekday_pattern: List[float]
    weekend_pattern: List[float]
    holiday_pattern: List[float]

class RuleBasedPredictor:
    """Rule-based predictions for VPP operations."""
    
    def __init__(self):
        self._initialize_load_profiles()
        self._initialize_price_rules()
        self._initialize_maintenance_rules()
        
    def _initialize_load_profiles(self) -> None:
        """Initialize standard load profiles."""
        # 24-hour patterns (percentage of peak load)
        self.load_profiles = LoadProfile(
            weekday_pattern=[  # Typical weekday pattern
                0.6, 0.55, 0.5, 0.5, 0.55, 0.7,    # 00:00 - 06:00
                0.85, 0.95, 1.0, 0.95, 0.9, 0.9,   # 06:00 - 12:00
                0.85, 0.85, 0.85, 0.85, 0.9, 0.95, # 12:00 - 18:00
                1.0, 0.95, 0.9, 0.8, 0.7, 0.65     # 18:00 - 24:00
            ],
            weekend_pattern=[  # Typical weekend pattern
                0.65, 0.6, 0.55, 0.5, 0.5, 0.55,   # 00:00 - 06:00
                0.6, 0.7, 0.8, 0.85, 0.9, 0.95,    # 06:00 - 12:00
                1.0, 1.0, 0.95, 0.95, 0.9, 0.9,    # 12:00 - 18:00
                0.85, 0.8, 0.75, 0.7, 0.7, 0.65    # 18:00 - 24:00
            ],
            holiday_pattern=[  # Holiday pattern
                0.65, 0.6, 0.55, 0.5, 0.5, 0.55,   # 00:00 - 06:00
                0.6, 0.65, 0.7, 0.75, 0.8, 0.85,   # 06:00 - 12:00
                0.9, 0.95, 1.0, 0.95, 0.9, 0.85,   # 12:00 - 18:00
                0.8, 0.75, 0.7, 0.7, 0.65, 0.65    # 18:00 - 24:00
            ]
        )
        
    def _initialize_price_rules(self) -> None:
        """Initialize price prediction rules."""
        self.price_rules = {
            'base_price': 50.0,  # $/MWh
            'peak_multiplier': 1.5,
            'off_peak_multiplier': 0.7,
            'shortage_multiplier': 2.0,
            'surplus_multiplier': 0.5,
            'renewable_discount': 0.1
        }
        
    def _initialize_maintenance_rules(self) -> None:
        """Initialize maintenance prediction rules."""
        self.maintenance_rules = {
            'operating_hours_threshold': 2000,
            'efficiency_threshold': 0.85,
            'temperature_threshold': 80,
            'vibration_threshold': 0.5
        }

    def predict_load(self, timestamp: datetime, base_load: float,
                    conditions: Dict) -> float:
        """Predict load using rule-based approach."""
        # Get appropriate daily pattern
        if timestamp.weekday() >= 5:  # Weekend
            pattern = self.load_profiles.weekend_pattern
        else:  # Weekday
            pattern = self.load_profiles.weekday_pattern
            
        # Get base multiplier from pattern
        hour = timestamp.hour
        base_multiplier = pattern[hour]
        
        # Apply weather adjustments
        temperature = conditions.get('temperature', 20)
        if temperature > 28:  # Hot day
            base_multiplier *= 1.2  # Increased AC usage
        elif temperature < 5:  # Cold day
            base_multiplier *= 1.15  # Increased heating
            
        return base_load * base_multiplier

    def predict_price(self, demand: float, supply: float,
                     renewable_percentage: float) -> float:
        """Predict energy price using rule-based approach."""
        base_price = self.price_rules['base_price']
        
        # Supply-demand balance
        if supply < demand:
            price_multiplier = self.price_rules['shortage_multiplier']
        elif supply > demand * 1.2:  # 20% oversupply
            price_multiplier = self.price_rules['surplus_multiplier']
        else:
            price_multiplier = 1.0
            
        # Time of day adjustment
        hour = datetime.now().hour
        if 9 <= hour <= 20:  # Peak hours
            price_multiplier *= self.price_rules['peak_multiplier']
        else:  # Off-peak hours
            price_multiplier *= self.price_rules['off_peak_multiplier']
            
        # Renewable energy discount
        renewable_discount = renewable_percentage * self.price_rules['renewable_discount']
        
        return base_price * price_multiplier * (1 - renewable_discount)

    def check_maintenance_needed(self, equipment_status: Dict) -> bool:
        """Check if maintenance is needed using rule-based approach."""
        rules = self.maintenance_rules
        
        if equipment_status['operating_hours'] > rules['operating_hours_threshold']:
            return True
            
        if equipment_status['efficiency'] < rules['efficiency_threshold']:
            return True
            
        if equipment_status['temperature'] > rules['temperature_threshold']:
            return True
            
        if equipment_status['vibration_level'] > rules['vibration_threshold']:
            return True
            
        return False

    def detect_anomalies(self, current_values: Dict, thresholds: Dict) -> List[str]:
        """Detect anomalies using rule-based thresholds."""
        anomalies = []
        
        # Check frequency deviation
        if abs(current_values['grid_frequency'] - 50.0) > thresholds['frequency_deviation']:
            anomalies.append('frequency_deviation')
            
        # Check voltage levels
        if abs(current_values['voltage'] - 1.0) > thresholds['voltage_deviation']:
            anomalies.append('voltage_deviation')
            
        # Check power factor
        if current_values['power_factor'] < thresholds['power_factor_min']:
            anomalies.append('poor_power_factor')
            
        # Check efficiency
        if current_values['efficiency'] < thresholds['efficiency_min']:
            anomalies.append('low_efficiency')
            
        return anomalies

class RuleEngine:
    """Main rule engine coordinating all rule-based predictions."""
    
    def __init__(self):
        self.predictor = RuleBasedPredictor()
        self.thresholds = {
            'frequency_deviation': 0.5,  # Hz
            'voltage_deviation': 0.1,    # p.u.
            'power_factor_min': 0.85,
            'efficiency_min': 0.8
        }
        
    def get_predictions(self, current_data: Dict, conditions: Dict) -> Dict:
        """Get predictions from all rule-based methods."""
        timestamp = datetime.now()
        base_load = current_data.get('base_load', 100.0)
        
        predictions = {
            'load_forecast': self.predictor.predict_load(
                timestamp, base_load, conditions
            ),
            'price_forecast': self.predictor.predict_price(
                current_data['demand'],
                current_data['total_generation'],
                current_data['renewable_percentage']
            ),
            'maintenance_needed': self.predictor.check_maintenance_needed(
                current_data['equipment_status']
            ),
            'anomalies': self.predictor.detect_anomalies(
                current_data, self.thresholds
            )
        }
        
        return predictions

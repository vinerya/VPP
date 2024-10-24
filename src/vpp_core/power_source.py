"""
Core power source management for Virtual Power Plant.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np

@dataclass
class PowerSourceStatus:
    """Status information for a power source."""
    available_capacity: float
    current_output: float
    efficiency: float
    operational: bool
    maintenance_required: bool

class PowerSource(ABC):
    """Abstract base class for all power sources in the VPP."""
    
    def __init__(self, name: str, capacity: float):
        self.name = name
        self.capacity = capacity
        self._status = PowerSourceStatus(
            available_capacity=capacity,
            current_output=0.0,
            efficiency=1.0,
            operational=True,
            maintenance_required=False
        )
        
    @abstractmethod
    def calculate_available_power(self, conditions: Dict) -> float:
        """Calculate available power based on current conditions."""
        pass
    
    @abstractmethod
    def adjust_output(self, target_output: float) -> float:
        """Adjust power output to target level."""
        pass
    
    def get_status(self) -> PowerSourceStatus:
        """Get current status of the power source."""
        return self._status

class RenewableSource(PowerSource):
    """Base class for renewable energy sources."""
    
    def __init__(self, name: str, capacity: float, weather_dependent: bool = True):
        super().__init__(name, capacity)
        self.weather_dependent = weather_dependent
        
    def calculate_available_power(self, conditions: Dict) -> float:
        """Calculate available renewable power based on weather conditions."""
        if not self.weather_dependent:
            return self.capacity
        return self._calculate_weather_impact(conditions) * self.capacity
    
    @abstractmethod
    def _calculate_weather_impact(self, conditions: Dict) -> float:
        """Calculate weather impact factor (0-1) on power generation."""
        pass

class DispatchableSource(PowerSource):
    """Base class for dispatchable power sources."""
    
    def __init__(self, name: str, capacity: float, ramp_rate: float):
        super().__init__(name, capacity)
        self.ramp_rate = ramp_rate  # MW per minute
        
    def adjust_output(self, target_output: float) -> float:
        """Adjust output considering ramp rate constraints."""
        max_change = self.ramp_rate * 60  # Convert to MW per hour
        current = self._status.current_output
        target = np.clip(target_output, 0, self.capacity)
        
        if abs(target - current) <= max_change:
            new_output = target
        else:
            new_output = current + np.sign(target - current) * max_change
            
        self._status.current_output = new_output
        return new_output

class StorageSystem(PowerSource):
    """Base class for energy storage systems."""
    
    def __init__(self, name: str, capacity: float, max_charge_rate: float, 
                 max_discharge_rate: float, efficiency: float = 0.9):
        super().__init__(name, capacity)
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.efficiency = efficiency
        self.stored_energy = 0.0
        
    def charge(self, amount: float) -> float:
        """Charge the storage system."""
        max_possible = min(
            self.max_charge_rate,
            (self.capacity - self.stored_energy) / self.efficiency
        )
        actual_charge = min(amount, max_possible)
        self.stored_energy += actual_charge * self.efficiency
        return actual_charge
    
    def discharge(self, amount: float) -> float:
        """Discharge from the storage system."""
        max_possible = min(
            self.max_discharge_rate,
            self.stored_energy * self.efficiency
        )
        actual_discharge = min(amount, max_possible)
        self.stored_energy -= actual_discharge / self.efficiency
        return actual_discharge

"""
Grid management and optimization for Virtual Power Plant.
"""
from typing import Dict, List, Optional
import numpy as np
from .power_source import PowerSource

class GridManager:
    """Manages grid operations and optimization for the VPP."""
    
    def __init__(self):
        self.power_sources: Dict[str, PowerSource] = {}
        self.total_capacity = 0.0
        self.current_demand = 0.0
        self.grid_frequency = 50.0  # Hz
        self.voltage_levels: Dict[str, float] = {}
        
    def add_power_source(self, source: PowerSource) -> None:
        """Add a power source to the grid."""
        self.power_sources[source.name] = source
        self.total_capacity += source.capacity
        
    def remove_power_source(self, source_name: str) -> None:
        """Remove a power source from the grid."""
        if source_name in self.power_sources:
            self.total_capacity -= self.power_sources[source_name].capacity
            del self.power_sources[source_name]
            
    def optimize_dispatch(self, demand: float, conditions: Dict) -> Dict[str, float]:
        """Optimize power dispatch across all sources."""
        dispatch_plan = {}
        available_power = {}
        
        # Calculate available power from each source
        for name, source in self.power_sources.items():
            available = source.calculate_available_power(conditions)
            available_power[name] = available
            
        # Simple merit order dispatch
        remaining_demand = demand
        sorted_sources = sorted(
            available_power.items(),
            key=lambda x: self._get_merit_order_priority(x[0])
        )
        
        for name, available in sorted_sources:
            if remaining_demand <= 0:
                dispatch_plan[name] = 0
            else:
                dispatch = min(available, remaining_demand)
                dispatch_plan[name] = dispatch
                remaining_demand -= dispatch
                
        return dispatch_plan
    
    def _get_merit_order_priority(self, source_name: str) -> int:
        """Get priority for merit order dispatch (lower is better)."""
        # Example priority order: renewables first, then storage, then dispatchable
        source = self.power_sources[source_name]
        if isinstance(source, RenewableSource):
            return 0
        elif isinstance(source, StorageSystem):
            return 1
        else:
            return 2
            
    def balance_grid(self, dispatch_plan: Dict[str, float]) -> bool:
        """Execute dispatch plan and balance the grid."""
        total_output = 0.0
        success = True
        
        for name, target in dispatch_plan.items():
            source = self.power_sources[name]
            actual_output = source.adjust_output(target)
            total_output += actual_output
            
            if abs(actual_output - target) > 0.01:  # 1% tolerance
                success = False
                
        self._update_grid_metrics(total_output)
        return success
    
    def _update_grid_metrics(self, total_output: float) -> None:
        """Update grid metrics based on current state."""
        # Simplified frequency calculation
        nominal_frequency = 50.0
        self.grid_frequency = nominal_frequency + (total_output - self.current_demand) * 0.1
        self.grid_frequency = np.clip(self.grid_frequency, 49.5, 50.5)

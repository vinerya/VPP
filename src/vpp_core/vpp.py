"""
Main Virtual Power Plant (VPP) controller.
"""
from typing import Dict, List, Optional
from datetime import datetime
import logging
from .power_source import PowerSource, StorageSystem
from .grid_manager import GridManager

logger = logging.getLogger(__name__)

class VirtualPowerPlant:
    """Main VPP controller class managing all components."""
    
    def __init__(self):
        self.grid_manager = GridManager()
        self.current_conditions: Dict = {}
        self.demand_forecast: List[float] = []
        self._initialize_logging()
        
    def _initialize_logging(self):
        """Initialize logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def add_power_source(self, source: PowerSource) -> None:
        """Add a power source to the VPP."""
        self.grid_manager.add_power_source(source)
        logger.info(f"Added power source: {source.name} with capacity {source.capacity}MW")
        
    def update_conditions(self, conditions: Dict) -> None:
        """Update current operating conditions."""
        self.current_conditions = conditions
        logger.info(f"Updated conditions: {conditions}")
        
    def optimize_and_dispatch(self, current_demand: float) -> Dict[str, float]:
        """Optimize and dispatch power sources based on current demand."""
        logger.info(f"Optimizing dispatch for demand: {current_demand}MW")
        
        # Get optimal dispatch plan
        dispatch_plan = self.grid_manager.optimize_dispatch(
            current_demand,
            self.current_conditions
        )
        
        # Execute dispatch plan
        success = self.grid_manager.balance_grid(dispatch_plan)
        
        if success:
            logger.info("Successfully executed dispatch plan")
        else:
            logger.warning("Some dispatch targets could not be met exactly")
            
        return dispatch_plan
    
    def get_system_status(self) -> Dict:
        """Get current status of the entire VPP system."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'total_capacity': self.grid_manager.total_capacity,
            'grid_frequency': self.grid_manager.grid_frequency,
            'power_sources': {}
        }
        
        for name, source in self.grid_manager.power_sources.items():
            source_status = source.get_status()
            status['power_sources'][name] = {
                'available_capacity': source_status.available_capacity,
                'current_output': source_status.current_output,
                'operational': source_status.operational
            }
            
        return status
    
    def handle_grid_event(self, event_type: str, event_data: Dict) -> None:
        """Handle grid events (e.g., frequency deviations, voltage issues)."""
        logger.info(f"Handling grid event: {event_type}")
        
        if event_type == 'frequency_deviation':
            self._handle_frequency_deviation(event_data)
        elif event_type == 'voltage_issue':
            self._handle_voltage_issue(event_data)
        elif event_type == 'demand_spike':
            self._handle_demand_spike(event_data)
            
    def _handle_frequency_deviation(self, event_data: Dict) -> None:
        """Handle frequency deviation events."""
        deviation = event_data.get('deviation', 0.0)
        if abs(deviation) > 0.5:  # Hz
            logger.warning(f"Significant frequency deviation: {deviation}Hz")
            # Implement frequency response
            
    def _handle_voltage_issue(self, event_data: Dict) -> None:
        """Handle voltage-related issues."""
        pass  # Implement voltage control logic
        
    def _handle_demand_spike(self, event_data: Dict) -> None:
        """Handle sudden demand spikes."""
        pass  # Implement demand response logic

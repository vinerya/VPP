"""
Advanced simulation capabilities for Virtual Power Plant demonstrations.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm

@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    duration_hours: int = 24
    time_step_minutes: int = 5
    base_load: float = 100.0  # MW
    load_volatility: float = 0.15
    weather_volatility: float = 0.2
    market_volatility: float = 0.1
    fault_probability: float = 0.01

class WeatherSimulator:
    """Simulates realistic weather patterns."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self._initialize_patterns()
        
    def _initialize_patterns(self):
        """Initialize daily patterns for weather variables."""
        hours = np.arange(24)
        
        # Solar irradiance pattern (bell curve centered at noon)
        self.solar_pattern = norm.pdf(hours, loc=12, scale=3)
        self.solar_pattern = self.solar_pattern / np.max(self.solar_pattern)
        
        # Wind speed pattern (higher at night)
        self.wind_pattern = 0.6 + 0.4 * np.cos(hours * 2 * np.pi / 24)
        
        # Temperature pattern
        self.temp_pattern = 20 + 5 * np.sin(hours * 2 * np.pi / 24 - np.pi/2)
        
    def get_conditions(self, timestamp: datetime) -> Dict[str, float]:
        """Get weather conditions for a specific timestamp."""
        hour = timestamp.hour + timestamp.minute / 60.0
        
        # Add randomness to base patterns
        noise = np.random.normal(0, self.config.weather_volatility)
        
        solar = np.interp(hour, np.arange(24), self.solar_pattern)
        solar = np.clip(solar + noise * 0.1, 0, 1)
        
        wind = np.interp(hour, np.arange(24), self.wind_pattern)
        wind = np.clip(wind + noise * 0.2, 0, 1)
        
        temp = np.interp(hour, np.arange(24), self.temp_pattern)
        temp = temp + noise * 2
        
        return {
            "solar_irradiance": solar * 1000,  # W/m²
            "wind_speed": wind * 15,           # m/s
            "temperature": temp,               # °C
            "cloud_cover": np.clip(1 - solar + noise * 0.1, 0, 1)
        }

class DemandSimulator:
    """Simulates realistic power demand patterns."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self._initialize_patterns()
        
    def _initialize_patterns(self):
        """Initialize demand patterns for different day types."""
        hours = np.arange(24)
        
        # Weekday pattern (two peaks: morning and evening)
        morning_peak = norm.pdf(hours, loc=9, scale=2)
        evening_peak = norm.pdf(hours, loc=19, scale=2.5)
        self.weekday_pattern = morning_peak + evening_peak
        self.weekday_pattern = self.weekday_pattern / np.max(self.weekday_pattern)
        
        # Weekend pattern (single afternoon peak)
        self.weekend_pattern = norm.pdf(hours, loc=14, scale=4)
        self.weekend_pattern = self.weekend_pattern / np.max(self.weekend_pattern)
        
    def get_demand(self, timestamp: datetime) -> float:
        """Get demand for a specific timestamp."""
        hour = timestamp.hour + timestamp.minute / 60.0
        is_weekend = timestamp.weekday() >= 5
        
        # Select appropriate pattern
        pattern = self.weekend_pattern if is_weekend else self.weekday_pattern
        base_demand = np.interp(hour, np.arange(24), pattern)
        
        # Add random variations
        noise = np.random.normal(0, self.config.load_volatility)
        demand = self.config.base_load * base_demand * (1 + noise)
        
        return max(0, demand)

class MarketSimulator:
    """Simulates energy market dynamics."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.base_price = 50.0  # $/MWh
        self.price_history: List[float] = []
        
    def get_price(self, demand: float, total_generation: float) -> float:
        """Calculate market price based on supply-demand balance."""
        supply_ratio = total_generation / max(demand, 0.1)
        price_factor = np.exp(-0.5 * (supply_ratio - 1))
        
        # Add random variations
        noise = np.random.normal(0, self.config.market_volatility)
        price = self.base_price * price_factor * (1 + noise)
        
        self.price_history.append(price)
        return max(0, price)

class GridDisturbanceSimulator:
    """Simulates grid disturbances and faults."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.active_faults: Dict[str, Dict] = {}
        
    def get_disturbances(self, timestamp: datetime) -> List[Dict]:
        """Generate potential grid disturbances."""
        disturbances = []
        
        # Random fault generation
        if np.random.random() < self.config.fault_probability:
            fault = self._generate_fault(timestamp)
            self.active_faults[fault['id']] = fault
            disturbances.append(fault)
        
        # Update existing faults
        self._update_active_faults(timestamp)
        
        return disturbances
    
    def _generate_fault(self, timestamp: datetime) -> Dict:
        """Generate a random fault event."""
        fault_types = ['line_fault', 'transformer_fault', 'frequency_deviation']
        fault_type = np.random.choice(fault_types)
        
        fault = {
            'id': f"fault_{timestamp.timestamp()}",
            'type': fault_type,
            'start_time': timestamp,
            'duration': timedelta(minutes=np.random.randint(5, 60)),
            'severity': np.random.uniform(0.1, 1.0)
        }
        
        return fault
    
    def _update_active_faults(self, timestamp: datetime) -> None:
        """Update status of active faults."""
        expired_faults = []
        for fault_id, fault in self.active_faults.items():
            if timestamp - fault['start_time'] > fault['duration']:
                expired_faults.append(fault_id)
        
        for fault_id in expired_faults:
            del self.active_faults[fault_id]

class VPPSimulator:
    """Main simulation controller for VPP demonstrations."""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.weather_sim = WeatherSimulator(self.config)
        self.demand_sim = DemandSimulator(self.config)
        self.market_sim = MarketSimulator(self.config)
        self.grid_sim = GridDisturbanceSimulator(self.config)
        self.simulation_data: List[Dict] = []
        
    def run_simulation(self, vpp) -> pd.DataFrame:
        """Run a complete VPP simulation."""
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        time_step = timedelta(minutes=self.config.time_step_minutes)
        
        for step in range(int(self.config.duration_hours * 60 / self.config.time_step_minutes)):
            current_time = start_time + step * time_step
            
            # Generate simulation conditions
            conditions = self.weather_sim.get_conditions(current_time)
            demand = self.demand_sim.get_demand(current_time)
            disturbances = self.grid_sim.get_disturbances(current_time)
            
            # Update VPP with simulated conditions
            vpp.update_conditions(conditions)
            
            # Handle any grid disturbances
            for disturbance in disturbances:
                vpp.handle_grid_event(disturbance['type'], disturbance)
            
            # Optimize and dispatch
            dispatch_plan = vpp.optimize_and_dispatch(demand)
            
            # Calculate market price
            total_generation = sum(dispatch_plan.values())
            price = self.market_sim.get_price(demand, total_generation)
            
            # Record simulation step data
            step_data = {
                'timestamp': current_time,
                'conditions': conditions,
                'demand': demand,
                'dispatch': dispatch_plan,
                'price': price,
                'disturbances': disturbances
            }
            self.simulation_data.append(step_data)
        
        # Convert simulation data to DataFrame for analysis
        return pd.DataFrame(self.simulation_data)

    def get_simulation_summary(self) -> Dict:
        """Get summary statistics of the simulation."""
        if not self.simulation_data:
            return {}
            
        df = pd.DataFrame(self.simulation_data)
        
        return {
            'duration_hours': self.config.duration_hours,
            'time_steps': len(self.simulation_data),
            'avg_demand': df['demand'].mean(),
            'peak_demand': df['demand'].max(),
            'avg_price': df['price'].mean(),
            'total_disturbances': sum(len(d) for d in df['disturbances']),
            'simulation_complete': True
        }

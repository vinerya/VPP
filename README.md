# VPP Core - Professional Virtual Power Plant Management System

A professional-grade Python library for managing Virtual Power Plants (VPP), featuring both rule-based and ML-based approaches for intelligent operations.

## Core Features

### Power Management
- Abstract base classes for different power sources
- Support for renewable, dispatchable, and storage systems
- Real-time power source status monitoring

### Grid Management
- Automated grid balancing
- Merit order dispatch optimization
- Frequency and voltage management
- Real-time grid metrics monitoring

### Advanced Simulation
- Weather pattern simulation
- Demand pattern generation
- Market price dynamics
- Grid disturbance scenarios
- Real-time system response

### Prediction and Optimization

#### Rule-Based Approach
The package provides rule-based methods for immediate deployment without requiring historical data:

- Load Prediction: Uses predefined daily patterns and weather adjustments
- Price Prediction: Based on supply-demand balance and time-of-day factors
- Maintenance Prediction: Uses equipment status thresholds
- Anomaly Detection: Threshold-based monitoring of key metrics

#### Machine Learning Approach
For enhanced accuracy with historical data:

- Load Prediction: RandomForest model trained on historical patterns
- Price Prediction: ML model considering multiple market factors
- Maintenance Prediction: Predictive maintenance using equipment data
- Anomaly Detection: IsolationForest for detecting unusual patterns

## Installation

```bash
pip install vpp-core
```

## Quick Start

### Using Rule-Based Approach

```python
from vpp_core import VirtualPowerPlant
from vpp_core.rule_engine import RuleEngine

# Initialize VPP and rule engine
vpp = VirtualPowerPlant()
rule_engine = RuleEngine()

# Get predictions
current_data = vpp.get_system_status()
conditions = {
    'temperature': 25,
    'cloud_cover': 0.3
}

predictions = rule_engine.get_predictions(current_data, conditions)
```

### Using ML-Based Approach

```python
from vpp_core import VirtualPowerPlant
from vpp_core.ml_engine import MLEngine

# Initialize VPP and ML engine
vpp = VirtualPowerPlant()
ml_engine = MLEngine()

# Train models with historical data
historical_data = pd.read_csv('historical_operations.csv')
ml_engine.train_all_models(historical_data)

# Get predictions
current_data = vpp.get_system_status()
future_conditions = pd.DataFrame(...)  # Future conditions data
predictions = ml_engine.get_predictions(current_data, future_conditions)
```

## Transitioning from Rule-Based to ML-Based

1. Start with Rule-Based Approach:
   - Implement initial VPP operations using rule_engine
   - Collect operational data during rule-based operation
   - Monitor system performance and data quality

2. Data Collection Phase:
   - Store all operational data including:
     - Power generation and demand
     - Weather conditions
     - Equipment status
     - Market prices
     - System events
   - Ensure data quality and proper labeling

3. ML Model Training:
   - Once sufficient data is collected (typically 3-6 months):
     ```python
     from vpp_core.ml_engine import MLEngine
     
     # Initialize ML engine
     ml_engine = MLEngine()
     
     # Train models with collected data
     historical_data = collect_historical_data()
     ml_engine.train_all_models(historical_data)
     
     # Save trained models
     ml_engine.save_models('models/')
     ```

4. Parallel Operation:
   - Run both rule-based and ML-based predictions
   - Compare performance metrics
   - Gradually transition to ML-based approach
   
5. Continuous Improvement:
   - Regularly retrain models with new data
   - Monitor prediction accuracy
   - Adjust model parameters as needed

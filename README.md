# Advanced Virtual Power Plant Simulation

This project simulates an advanced virtual power plant (VPP) that manages various power sources, including renewable energy, while considering weather conditions, energy market dynamics, and demand response mechanisms. It uses sophisticated algorithms and professional-grade libraries for power system modeling, forecasting, and optimization.

## Features

- Simulates multiple power sources: Solar Farm, Wind Farm, Battery Storage, and Biomass Plant
- Uses PandaPower for power system modeling and analysis
- Implements weather forecasting using scikit-learn's Random Forest algorithm
- Utilizes NumPy for efficient numerical computations
- Implements advanced optimization for grid balancing using SciPy
- Uses Plotly for interactive visualizations
- Simulates realistic market dynamics

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SciPy
- PandaPower
- Plotly

## Installation

1. Clone this repository or download the source code.
2. Install the required packages:

```
pip install numpy pandas matplotlib scikit-learn scipy pandapower plotly
```

## Usage

To run the simulation:

1. Navigate to the project directory in your terminal.
2. Run the following command:

```
python main.py
```

The simulation will run for 72 hours by default. You can modify the simulation duration in the `main()` function of `main.py`.

## Output

The simulation provides two types of output:

1. Console logs: Detailed information about each time step, including weather conditions, power generation, demand, grid balance, and market price.

2. Visualization: An HTML file named `simulation_results.html` will be generated in the project directory. This file contains interactive plots of:
   - Demand vs Generation over time
   - Grid Balance over time
   - Market Price over time

Open the HTML file in a web browser to view the interactive visualizations.

## Customization

You can customize the simulation by modifying the following in `main.py`:

- Add or remove power sources in the `main()` function
- Adjust the capacity and availability of power sources
- Modify the weather forecasting model in the `WeatherForecaster` class
- Change the demand simulation algorithm in the `VirtualPowerPlant.simulate_demand()` method
- Adjust the energy market model in the `EnergyMarket` class
- Modify the optimization algorithm in the `VirtualPowerPlant.optimize_dispatch()` method

## Contributing

Contributions to improve the simulation or add new features are welcome. Please feel free to submit a pull request or open an issue to discuss potential changes.

## License

This project is open-source and available under the MIT License.
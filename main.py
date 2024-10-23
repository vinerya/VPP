import random
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.timeseries import DFData
from pandapower.control import ConstControl
from pandapower.timeseries.run_time_series import run_timeseries
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import linprog
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherForecaster:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.history = []
        self.is_fitted = False

    def update(self, temperature, wind_speed, cloud_cover):
        self.history.append([temperature, wind_speed, cloud_cover])
        if len(self.history) > 24:  # Train on last 24 hours
            X = self.history[-24:-1]
            y = self.history[-23:]
            self.model.fit(X, y)
            self.is_fitted = True

    def forecast(self):
        if not self.is_fitted:
            return self.history[-1] if self.history else [20, 5, 0.5]  # Default values if not fitted
        return self.model.predict([self.history[-1]])[0]

class PowerSource:
    def __init__(self, name, capacity, availability):
        self.name = name
        self.capacity = capacity
        self.availability = availability

    def generate(self, weather):
        return self.capacity * self.availability * self._weather_factor(weather)

    def _weather_factor(self, weather):
        return 1.0

class SolarFarm(PowerSource):
    def _weather_factor(self, weather):
        return 1 - (0.7 * weather[2])  # cloud_cover

class WindFarm(PowerSource):
    def _weather_factor(self, weather):
        wind_speed = weather[1]
        if wind_speed < 3:
            return 0.1
        elif wind_speed > 25:
            return 0.5
        else:
            return (wind_speed - 3) / 22

class BatteryStorage(PowerSource):
    def __init__(self, name, capacity, availability):
        super().__init__(name, capacity, availability)
        self.charge = capacity * 0.5

    def generate(self, weather):
        output = min(self.charge, self.capacity * 0.2)
        self.charge -= output
        return output

    def store(self, amount):
        space_left = self.capacity - self.charge
        stored = min(amount, space_left)
        self.charge += stored
        return stored

class BiomassPlant(PowerSource):
    pass

class EnergyMarket:
    def __init__(self):
        self.price = 50

    def update_price(self, supply, demand):
        ratio = supply / max(demand, 0.1)  # Avoid division by zero
        self.price *= np.clip(ratio, 0.95, 1.05)
        self.price = np.clip(self.price, 10, 200)

class VirtualPowerPlant:
    def __init__(self):
        self.power_sources = []
        self.total_capacity = 0
        self.current_demand = 0
        self.weather_forecaster = WeatherForecaster()
        self.market = EnergyMarket()
        self.battery_storage = None
        self.history = {'time': [], 'demand': [], 'generation': [], 'price': [], 'balance': []}
        self.net = self._create_pandapower_network()

    def _create_pandapower_network(self):
        net = pp.create_empty_network()
        
        # Create buses
        b1 = pp.create_bus(net, vn_kv=110.)
        b2 = pp.create_bus(net, vn_kv=20.)
        
        # Create transformer
        pp.create_transformer(net, hv_bus=b1, lv_bus=b2, std_type="40 MVA 110/20 kV")
        
        # Create load
        pp.create_load(net, bus=b2, p_mw=0, q_mvar=0)
        
        # Create generator (to represent all power sources)
        pp.create_gen(net, bus=b2, p_mw=0, vm_pu=1.0)
        
        # Create external grid connection (slack bus)
        pp.create_ext_grid(net, bus=b1, vm_pu=1.0, va_degree=0)
        
        return net

    def add_power_source(self, power_source):
        self.power_sources.append(power_source)
        self.total_capacity += power_source.capacity
        if isinstance(power_source, BatteryStorage):
            self.battery_storage = power_source

    def simulate_demand(self, time):
        base_demand = self.total_capacity * (0.6 + 0.4 * np.sin(time.hour * np.pi / 12))
        self.current_demand = base_demand * np.random.uniform(0.9, 1.1)

    def balance_grid(self, weather):
        generation = [source.generate(weather) for source in self.power_sources if not isinstance(source, BatteryStorage)]
        total_generation = sum(generation)
        balance = total_generation - self.current_demand

        if balance > 0 and self.battery_storage:
            stored = self.battery_storage.store(balance)
            balance -= stored
        elif balance < 0 and self.battery_storage:
            additional_power = self.battery_storage.generate(weather)
            balance += additional_power

        return balance, total_generation

    def optimize_dispatch(self, weather):
        c = [-source.generate(weather) for source in self.power_sources]
        A = [[1] * len(self.power_sources)]
        b = [self.current_demand]
        x_bounds = [(0, source.capacity) for source in self.power_sources]
        
        res = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method='highs')
        return res.x if res.success else [0] * len(self.power_sources)

    def run_simulation(self, hours):
        start_time = datetime.now()
        time_index = pd.date_range(start=start_time, periods=hours, freq='H')
        demand_series = pd.Series(index=time_index, dtype=float)
        generation_series = pd.Series(index=time_index, dtype=float)

        # Initialize weather with random values
        weather = [random.uniform(10, 30), random.uniform(0, 20), random.uniform(0, 1)]
        self.weather_forecaster.update(*weather)

        for hour in range(hours):
            current_time = start_time + timedelta(hours=hour)
            weather = self.weather_forecaster.forecast()
            self.weather_forecaster.update(*weather)
            self.simulate_demand(current_time)
            
            optimal_dispatch = self.optimize_dispatch(weather)
            total_generation = sum(optimal_dispatch)
            
            balance = total_generation - self.current_demand
            self.market.update_price(total_generation, self.current_demand)

            demand_series[current_time] = self.current_demand
            generation_series[current_time] = total_generation

            self.history['time'].append(current_time)
            self.history['demand'].append(self.current_demand)
            self.history['generation'].append(total_generation)
            self.history['price'].append(self.market.price)
            self.history['balance'].append(balance)

            logger.info(f"Time: {current_time}")
            logger.info(f"Weather - Temp: {weather[0]:.1f}°C, Wind: {weather[1]:.1f} m/s, Cloud: {weather[2]:.2f}")
            logger.info(f"Demand: {self.current_demand:.2f} MW")
            logger.info(f"Generation: {total_generation:.2f} MW")
            logger.info(f"Balance: {balance:.2f} MW")
            logger.info(f"Market Price: ${self.market.price:.2f}/MWh")

        # Prepare data for power flow simulation
        profiles = pd.DataFrame({"load": demand_series, "gen": generation_series})
        ds = DFData(profiles)

        # Create controllers
        ConstControl(self.net, element='load', variable='p_mw', element_index=0, data_source=ds, profile_name="load")
        ConstControl(self.net, element='gen', variable='p_mw', element_index=0, data_source=ds, profile_name="gen")

        # Run time series simulation
        run_timeseries(self.net, time_steps=range(len(time_index)))

        logger.info("Power flow simulation completed successfully.")

    def visualize_results(self):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=('Demand vs Generation', 'Grid Balance', 'Market Price'))

        fig.add_trace(go.Scatter(x=self.history['time'], y=self.history['demand'], name='Demand'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.history['time'], y=self.history['generation'], name='Generation'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.history['time'], y=self.history['balance'], name='Balance'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.history['time'], y=self.history['price'], name='Price'), row=3, col=1)

        fig.update_layout(height=900, width=1200, title_text="Virtual Power Plant Simulation Results")
        fig.write_html("simulation_results.html")
        logger.info("Simulation results visualization saved as 'simulation_results.html'")

def main():
    vpp = VirtualPowerPlant()

    vpp.add_power_source(SolarFarm("Solar Farm", 100, 0.7))
    vpp.add_power_source(WindFarm("Wind Farm", 150, 0.6))
    vpp.add_power_source(BatteryStorage("Battery Storage", 50, 0.9))
    vpp.add_power_source(BiomassPlant("Biomass Plant", 75, 0.8))

    vpp.run_simulation(72)
    vpp.visualize_results()

if __name__ == "__main__":
    main()
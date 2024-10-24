"""
Machine Learning Engine for VPP predictive capabilities.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import joblib
import logging

logger = logging.getLogger(__name__)

class LoadPredictor:
    """Predicts future load patterns using machine learning."""
    
    def __init__(self, forecast_horizon: int = 24):
        self.horizon = forecast_horizon
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for load prediction."""
        features = []
        for idx in range(len(data)):
            row = data.iloc[idx]
            timestamp = row.name if isinstance(row.name, datetime) else pd.to_datetime(row.name)
            
            feature_vector = [
                timestamp.hour,
                timestamp.weekday(),
                timestamp.month,
                int(timestamp.weekday() >= 5),  # is_weekend
                row['temperature'] if 'temperature' in row else 20.0,
                row['cloud_cover'] if 'cloud_cover' in row else 0.5,
            ]
            features.append(feature_vector)
            
        return np.array(features)
        
    def train(self, historical_data: pd.DataFrame) -> None:
        """Train the load prediction model."""
        X = self.prepare_features(historical_data)
        y = historical_data['demand'].values
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        logger.info("Load prediction model trained successfully")
        
    def predict(self, future_conditions: pd.DataFrame) -> np.ndarray:
        """Predict future load values."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        X = self.prepare_features(future_conditions)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class AnomalyDetector:
    """Detects anomalies in VPP operations."""
    
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection."""
        features = [
            'demand',
            'grid_frequency',
            'total_generation',
            'price'
        ]
        return data[features].values
        
    def train(self, historical_data: pd.DataFrame) -> None:
        """Train the anomaly detection model."""
        X = self.prepare_features(historical_data)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
        logger.info("Anomaly detection model trained successfully")
        
    def detect_anomalies(self, data: pd.DataFrame) -> List[bool]:
        """Detect anomalies in operational data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before detecting anomalies")
            
        X = self.prepare_features(data)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return [pred == -1 for pred in predictions]

class MaintenancePredictor:
    """Predicts maintenance requirements for power sources."""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for maintenance prediction."""
        features = [
            'operating_hours',
            'efficiency',
            'output_variance',
            'temperature',
            'vibration_level'
        ]
        return data[features].values
        
    def train(self, historical_data: pd.DataFrame) -> None:
        """Train the maintenance prediction model."""
        X = self.prepare_features(historical_data)
        y = historical_data['maintenance_needed'].values
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        logger.info("Maintenance prediction model trained successfully")
        
    def predict_maintenance(self, operational_data: pd.DataFrame) -> np.ndarray:
        """Predict maintenance requirements."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        X = self.prepare_features(operational_data)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]  # Probability of maintenance needed

class PricePredictor:
    """Predicts energy market prices using machine learning."""
    
    def __init__(self, forecast_horizon: int = 24):
        self.horizon = forecast_horizon
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for price prediction."""
        features = []
        for idx in range(len(data)):
            row = data.iloc[idx]
            timestamp = row.name if isinstance(row.name, datetime) else pd.to_datetime(row.name)
            
            feature_vector = [
                timestamp.hour,
                timestamp.weekday(),
                row['demand'],
                row['total_generation'],
                row['renewable_percentage'],
                row['grid_frequency']
            ]
            features.append(feature_vector)
            
        return np.array(features)
        
    def train(self, historical_data: pd.DataFrame) -> None:
        """Train the price prediction model."""
        X = self.prepare_features(historical_data)
        y = historical_data['price'].values
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        logger.info("Price prediction model trained successfully")
        
    def predict_prices(self, future_conditions: pd.DataFrame) -> np.ndarray:
        """Predict future energy prices."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        X = self.prepare_features(future_conditions)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class MLEngine:
    """Main ML engine coordinating all predictive capabilities."""
    
    def __init__(self):
        self.load_predictor = LoadPredictor()
        self.anomaly_detector = AnomalyDetector()
        self.maintenance_predictor = MaintenancePredictor()
        self.price_predictor = PricePredictor()
        
    def train_all_models(self, historical_data: pd.DataFrame) -> None:
        """Train all ML models with historical data."""
        logger.info("Starting training of all ML models")
        
        try:
            self.load_predictor.train(historical_data)
            self.anomaly_detector.train(historical_data)
            self.maintenance_predictor.train(historical_data)
            self.price_predictor.train(historical_data)
            logger.info("All ML models trained successfully")
        except Exception as e:
            logger.error(f"Error training ML models: {str(e)}")
            raise
            
    def save_models(self, directory: str) -> None:
        """Save all trained models to disk."""
        models = {
            'load_predictor': self.load_predictor,
            'anomaly_detector': self.anomaly_detector,
            'maintenance_predictor': self.maintenance_predictor,
            'price_predictor': self.price_predictor
        }
        
        for name, model in models.items():
            joblib.dump(model, f"{directory}/{name}.joblib")
            
    def load_models(self, directory: str) -> None:
        """Load all models from disk."""
        models = {
            'load_predictor': self.load_predictor,
            'anomaly_detector': self.anomaly_detector,
            'maintenance_predictor': self.maintenance_predictor,
            'price_predictor': self.price_predictor
        }
        
        for name, model in models.items():
            loaded_model = joblib.load(f"{directory}/{name}.joblib")
            setattr(self, name, loaded_model)
            
    def get_predictions(self, current_data: pd.DataFrame, 
                       future_conditions: pd.DataFrame) -> Dict:
        """Get predictions from all models."""
        predictions = {
            'load_forecast': self.load_predictor.predict(future_conditions),
            'anomalies': self.anomaly_detector.detect_anomalies(current_data),
            'maintenance_required': self.maintenance_predictor.predict_maintenance(current_data),
            'price_forecast': self.price_predictor.predict_prices(future_conditions)
        }
        
        return predictions

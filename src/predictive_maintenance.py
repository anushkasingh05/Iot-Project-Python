import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random

class PredictiveMaintenance:
    """Predictive maintenance system using machine learning"""
    
    def __init__(self):
        self.models_path = "models"
        self.ensure_models_directory()
        self.equipment_types = ['HVAC_Unit_1', 'HVAC_Unit_2', 'HVAC_Unit_3', 
                               'Lighting_System', 'Security_System', 'Energy_System']
        self.models = {}
        self.scalers = {}
        self.load_or_train_models()
    
    def ensure_models_directory(self):
        """Ensure models directory exists"""
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
    
    def load_or_train_models(self):
        """Load existing models or train new ones"""
        for equipment in self.equipment_types:
            model_file = os.path.join(self.models_path, f"{equipment}_model.pkl")
            scaler_file = os.path.join(self.models_path, f"{equipment}_scaler.pkl")
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                # Load existing model
                self.models[equipment] = joblib.load(model_file)
                self.scalers[equipment] = joblib.load(scaler_file)
            else:
                # Train new model
                self.train_model_for_equipment(equipment)
    
    def generate_training_data(self, equipment: str, days: int = 365) -> pd.DataFrame:
        """Generate realistic training data for equipment"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate daily data points
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = []
        for date in dates:
            # Base values for different equipment types
            if 'HVAC' in equipment:
                base_temp = 22 + random.uniform(-5, 5)
                base_humidity = 45 + random.uniform(-15, 15)
                base_vibration = random.uniform(0.1, 0.5)
                base_pressure = random.uniform(100, 150)
                base_energy = random.uniform(30, 70)
                
                # Simulate wear and tear over time
                days_elapsed = (date - start_date).days
                wear_factor = min(days_elapsed / 365, 1.0)  # Normalize to 0-1
                
                # Increase failure probability with wear
                failure_prob = min(0.1 + wear_factor * 0.3, 0.8)
                
                # Add some random failures
                if random.random() < 0.05:  # 5% random failure rate
                    failure_prob = random.uniform(0.7, 1.0)
                
            elif 'Lighting' in equipment:
                base_temp = 25 + random.uniform(-3, 3)
                base_humidity = 40 + random.uniform(-10, 10)
                base_vibration = random.uniform(0.05, 0.2)
                base_pressure = random.uniform(110, 120)
                base_energy = random.uniform(20, 50)
                
                days_elapsed = (date - start_date).days
                wear_factor = min(days_elapsed / 365, 1.0)
                failure_prob = min(0.05 + wear_factor * 0.2, 0.6)
                
                if random.random() < 0.03:  # 3% random failure rate
                    failure_prob = random.uniform(0.6, 1.0)
                    
            elif 'Security' in equipment:
                base_temp = 23 + random.uniform(-2, 2)
                base_humidity = 35 + random.uniform(-5, 5)
                base_vibration = random.uniform(0.01, 0.1)
                base_pressure = random.uniform(100, 110)
                base_energy = random.uniform(10, 30)
                
                days_elapsed = (date - start_date).days
                wear_factor = min(days_elapsed / 365, 1.0)
                failure_prob = min(0.02 + wear_factor * 0.15, 0.4)
                
                if random.random() < 0.02:  # 2% random failure rate
                    failure_prob = random.uniform(0.4, 1.0)
                    
            else:  # Energy system
                base_temp = 24 + random.uniform(-3, 3)
                base_humidity = 42 + random.uniform(-8, 8)
                base_vibration = random.uniform(0.1, 0.3)
                base_pressure = random.uniform(105, 135)
                base_energy = random.uniform(40, 80)
                
                days_elapsed = (date - start_date).days
                wear_factor = min(days_elapsed / 365, 1.0)
                failure_prob = min(0.08 + wear_factor * 0.25, 0.7)
                
                if random.random() < 0.04:  # 4% random failure rate
                    failure_prob = random.uniform(0.6, 1.0)
            
            # Add noise to base values
            temp = base_temp + random.uniform(-2, 2)
            humidity = base_humidity + random.uniform(-5, 5)
            vibration = base_vibration + random.uniform(-0.05, 0.05)
            pressure = base_pressure + random.uniform(-5, 5)
            energy_consumption = base_energy + random.uniform(-3, 3)
            
            # Determine if failure occurred
            failure = 1 if random.random() < failure_prob else 0
            
            # Add seasonal effects
            season_factor = np.sin(2 * np.pi * date.dayofyear / 365)
            temp += season_factor * 3
            humidity += season_factor * 10
            
            data.append({
                'date': date,
                'temperature': temp,
                'humidity': humidity,
                'vibration': vibration,
                'pressure': pressure,
                'energy_consumption': energy_consumption,
                'days_since_maintenance': random.randint(0, 90),
                'operating_hours': random.randint(8, 24),
                'failure': failure,
                'failure_probability': failure_prob
            })
        
        return pd.DataFrame(data)
    
    def train_model_for_equipment(self, equipment: str):
        """Train ML model for specific equipment"""
        try:
            # Generate training data
            training_data = self.generate_training_data(equipment)
            
            # Prepare features
            feature_columns = ['temperature', 'humidity', 'vibration', 'pressure', 
                             'energy_consumption', 'days_since_maintenance', 'operating_hours']
            
            X = training_data[feature_columns]
            y = training_data['failure']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            
            print(f"Model trained for {equipment}:")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1-Score: {f1:.3f}")
            
            # Save model and scaler
            model_file = os.path.join(self.models_path, f"{equipment}_model.pkl")
            scaler_file = os.path.join(self.models_path, f"{equipment}_scaler.pkl")
            
            joblib.dump(model, model_file)
            joblib.dump(scaler, scaler_file)
            
            # Store in memory
            self.models[equipment] = model
            self.scalers[equipment] = scaler
            
        except Exception as e:
            print(f"Error training model for {equipment}: {e}")
    
    def predict_failure(self, equipment: str, sensor_data: Dict) -> Dict:
        """Predict failure probability for equipment"""
        try:
            if equipment not in self.models:
                return {'error': f'No model available for {equipment}'}
            
            # Prepare features
            features = [
                sensor_data.get('temperature', 25),
                sensor_data.get('humidity', 45),
                sensor_data.get('vibration', 0.2),
                sensor_data.get('pressure', 120),
                sensor_data.get('energy_consumption', 50),
                sensor_data.get('days_since_maintenance', 30),
                sensor_data.get('operating_hours', 16)
            ]
            
            # Scale features
            features_scaled = self.scalers[equipment].transform([features])
            
            # Make prediction
            failure_prob = self.models[equipment].predict_proba(features_scaled)[0][1]
            
            # Determine risk level
            if failure_prob < 0.2:
                risk_level = 'Low'
            elif failure_prob < 0.5:
                risk_level = 'Medium'
            else:
                risk_level = 'High'
            
            return {
                'equipment': equipment,
                'failure_probability': failure_prob,
                'risk_level': risk_level,
                'recommendation': self._get_recommendation(failure_prob, risk_level),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Prediction error: {str(e)}'}
    
    def _get_recommendation(self, failure_prob: float, risk_level: str) -> str:
        """Get maintenance recommendation based on failure probability"""
        if risk_level == 'Low':
            return "Continue monitoring. Schedule routine maintenance within 30 days."
        elif risk_level == 'Medium':
            return "Schedule preventive maintenance within 7 days. Monitor closely."
        else:
            return "Immediate maintenance required. Consider emergency shutdown if necessary."
    
    def get_predictions(self) -> Optional[pd.DataFrame]:
        """Get failure predictions for all equipment"""
        try:
            predictions = []
            
            for equipment in self.equipment_types:
                # Generate current sensor data
                sensor_data = self._generate_current_sensor_data(equipment)
                
                # Get prediction
                prediction = self.predict_failure(equipment, sensor_data)
                
                if 'error' not in prediction:
                    predictions.append(prediction)
            
            if predictions:
                return pd.DataFrame(predictions)
            return None
            
        except Exception as e:
            print(f"Error getting predictions: {e}")
            return None
    
    def _generate_current_sensor_data(self, equipment: str) -> Dict:
        """Generate current sensor data for equipment"""
        if 'HVAC' in equipment:
            return {
                'temperature': random.uniform(20, 28),
                'humidity': random.uniform(35, 55),
                'vibration': random.uniform(0.1, 0.6),
                'pressure': random.uniform(100, 160),
                'energy_consumption': random.uniform(30, 80),
                'days_since_maintenance': random.randint(0, 120),
                'operating_hours': random.randint(8, 24)
            }
        elif 'Lighting' in equipment:
            return {
                'temperature': random.uniform(22, 28),
                'humidity': random.uniform(35, 50),
                'vibration': random.uniform(0.05, 0.25),
                'pressure': random.uniform(110, 125),
                'energy_consumption': random.uniform(20, 60),
                'days_since_maintenance': random.randint(0, 90),
                'operating_hours': random.randint(10, 18)
            }
        elif 'Security' in equipment:
            return {
                'temperature': random.uniform(20, 26),
                'humidity': random.uniform(30, 45),
                'vibration': random.uniform(0.01, 0.15),
                'pressure': random.uniform(100, 115),
                'energy_consumption': random.uniform(10, 40),
                'days_since_maintenance': random.randint(0, 60),
                'operating_hours': random.randint(20, 24)
            }
        else:  # Energy system
            return {
                'temperature': random.uniform(21, 27),
                'humidity': random.uniform(35, 50),
                'vibration': random.uniform(0.1, 0.4),
                'pressure': random.uniform(105, 145),
                'energy_consumption': random.uniform(40, 90),
                'days_since_maintenance': random.randint(0, 100),
                'operating_hours': random.randint(12, 24)
            }
    
    def get_maintenance_schedule(self) -> Optional[pd.DataFrame]:
        """Get optimized maintenance schedule"""
        try:
            predictions = self.get_predictions()
            
            if predictions is None:
                return None
            
            schedule = []
            current_date = datetime.now()
            
            for _, prediction in predictions.iterrows():
                equipment = prediction['equipment']
                risk_level = prediction['risk_level']
                failure_prob = prediction['failure_probability']
                
                # Determine maintenance priority and timing
                if risk_level == 'High':
                    priority = 'Critical'
                    days_until_maintenance = 1
                elif risk_level == 'Medium':
                    priority = 'High'
                    days_until_maintenance = 7
                else:
                    priority = 'Normal'
                    days_until_maintenance = 30
                
                maintenance_date = current_date + timedelta(days=days_until_maintenance)
                
                schedule.append({
                    'equipment': equipment,
                    'priority': priority,
                    'maintenance_date': maintenance_date.strftime('%Y-%m-%d'),
                    'risk_level': risk_level,
                    'failure_probability': failure_prob,
                    'estimated_cost': self._estimate_maintenance_cost(equipment, risk_level)
                })
            
            return pd.DataFrame(schedule)
            
        except Exception as e:
            print(f"Error getting maintenance schedule: {e}")
            return None
    
    def _estimate_maintenance_cost(self, equipment: str, risk_level: str) -> float:
        """Estimate maintenance cost based on equipment and risk level"""
        base_costs = {
            'HVAC_Unit_1': 500,
            'HVAC_Unit_2': 500,
            'HVAC_Unit_3': 500,
            'Lighting_System': 200,
            'Security_System': 300,
            'Energy_System': 400
        }
        
        base_cost = base_costs.get(equipment, 300)
        
        if risk_level == 'High':
            multiplier = 2.0
        elif risk_level == 'Medium':
            multiplier = 1.5
        else:
            multiplier = 1.0
        
        return base_cost * multiplier
    
    def get_cost_analysis(self) -> Optional[pd.DataFrame]:
        """Get maintenance cost analysis"""
        try:
            # Generate sample cost data
            months = pd.date_range(start=datetime.now() - timedelta(days=180), 
                                 end=datetime.now(), freq='ME')
            
            cost_data = []
            for month in months:
                # Generate monthly costs
                preventive_cost = random.uniform(2000, 4000)
                corrective_cost = random.uniform(1000, 3000)
                emergency_cost = random.uniform(500, 2000)
                total_cost = preventive_cost + corrective_cost + emergency_cost
                
                cost_data.append({
                    'month': month.strftime('%Y-%m'),
                    'preventive': preventive_cost,
                    'corrective': corrective_cost,
                    'emergency': emergency_cost,
                    'total_cost': total_cost
                })
            
            # Add category breakdown for pie chart
            categories = ['Preventive', 'Corrective', 'Emergency']
            costs = [sum(d['preventive'] for d in cost_data),
                    sum(d['corrective'] for d in cost_data),
                    sum(d['emergency'] for d in cost_data)]
            
            pie_data = pd.DataFrame({
                'category': categories,
                'cost': costs
            })
            
            return pie_data
            
        except Exception as e:
            print(f"Error getting cost analysis: {e}")
            return None
    
    def retrain_models(self):
        """Retrain all models with new data"""
        try:
            for equipment in self.equipment_types:
                print(f"Retraining model for {equipment}...")
                self.train_model_for_equipment(equipment)
            print("All models retrained successfully!")
        except Exception as e:
            print(f"Error retraining models: {e}")
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for all models"""
        try:
            performance = {}
            
            for equipment in self.equipment_types:
                if equipment in self.models:
                    # Generate test data
                    test_data = self.generate_training_data(equipment, days=30)
                    
                    # Prepare features
                    feature_columns = ['temperature', 'humidity', 'vibration', 'pressure', 
                                     'energy_consumption', 'days_since_maintenance', 'operating_hours']
                    
                    X_test = test_data[feature_columns]
                    y_test = test_data['failure']
                    
                    # Scale features
                    X_test_scaled = self.scalers[equipment].transform(X_test)
                    
                    # Make predictions
                    y_pred = self.models[equipment].predict(X_test_scaled)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                    
                    performance[equipment] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    }
            
            return performance
            
        except Exception as e:
            print(f"Error getting model performance: {e}")
            return {}


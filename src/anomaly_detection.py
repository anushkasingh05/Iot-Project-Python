import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

class AnomalyDetection:
    """Real-time anomaly detection for IoT sensor data"""
    
    def __init__(self):
        self.models_path = "models/anomaly"
        self.ensure_models_directory()
        self.systems = ['HVAC', 'Lighting', 'Security', 'Energy', 'Environmental']
        self.models = {}
        self.scalers = {}
        self.sensitivity = 0.5
        self.threshold = 0.7
        self.load_or_train_models()
    
    def ensure_models_directory(self):
        """Ensure models directory exists"""
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
    
    def load_or_train_models(self):
        """Load existing models or train new ones"""
        for system in self.systems:
            model_file = os.path.join(self.models_path, f"{system}_anomaly_model.pkl")
            scaler_file = os.path.join(self.models_path, f"{system}_anomaly_scaler.pkl")
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                # Load existing model
                self.models[system] = joblib.load(model_file)
                self.scalers[system] = joblib.load(scaler_file)
            else:
                # Train new model
                self.train_model_for_system(system)
    
    def generate_training_data(self, system: str, days: int = 30) -> pd.DataFrame:
        """Generate training data for anomaly detection"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate hourly data points
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
        
        data = []
        for timestamp in timestamps:
            # Generate normal data with some anomalies
            if system == 'HVAC':
                # Normal HVAC data
                temp = 22 + random.uniform(-3, 3)
                humidity = 45 + random.uniform(-10, 10)
                pressure = 120 + random.uniform(-10, 10)
                energy = 50 + random.uniform(-10, 10)
                
                # Add some anomalies (5% of data)
                if random.random() < 0.05:
                    temp += random.uniform(10, 20)  # High temperature anomaly
                    energy += random.uniform(20, 40)  # High energy consumption
                
            elif system == 'Lighting':
                # Normal lighting data
                usage = 40 + random.uniform(-20, 20)
                efficiency = 85 + random.uniform(-10, 10)
                power = 30 + random.uniform(-5, 5)
                duration = 8 + random.uniform(-2, 2)
                
                # Add some anomalies
                if random.random() < 0.05:
                    usage += random.uniform(30, 50)  # High usage anomaly
                    efficiency -= random.uniform(20, 40)  # Low efficiency
                
            elif system == 'Security':
                # Normal security data
                alerts = 0
                access_attempts = random.randint(50, 150)
                camera_status = 1
                system_health = 95 + random.uniform(-5, 5)
                
                # Add some anomalies
                if random.random() < 0.05:
                    alerts = random.randint(1, 5)  # Security alert anomaly
                    access_attempts += random.randint(100, 300)  # High access attempts
                
            elif system == 'Energy':
                # Normal energy data
                consumption = 100 + random.uniform(-20, 20)
                voltage = 230 + random.uniform(-10, 10)
                current = 50 + random.uniform(-10, 10)
                power_factor = 0.95 + random.uniform(-0.05, 0.05)
                
                # Add some anomalies
                if random.random() < 0.05:
                    consumption += random.uniform(50, 100)  # High consumption anomaly
                    voltage += random.uniform(20, 40)  # Voltage spike
                
            else:  # Environmental
                # Normal environmental data
                co2 = 400 + random.uniform(-50, 50)
                voc = 100 + random.uniform(-20, 20)
                pm25 = 10 + random.uniform(-5, 5)
                air_quality = 85 + random.uniform(-10, 10)
                
                # Add some anomalies
                if random.random() < 0.05:
                    co2 += random.uniform(200, 400)  # High CO2 anomaly
                    pm25 += random.uniform(20, 40)  # High particulate matter
            
            # Create feature vector based on system
            if system == 'HVAC':
                features = [temp, humidity, pressure, energy]
            elif system == 'Lighting':
                features = [usage, efficiency, power, duration]
            elif system == 'Security':
                features = [alerts, access_attempts, camera_status, system_health]
            elif system == 'Energy':
                features = [consumption, voltage, current, power_factor]
            else:  # Environmental
                features = [co2, voc, pm25, air_quality]
            
            data.append({
                'timestamp': timestamp,
                'system': system,
                'features': features
            })
        
        return pd.DataFrame(data)
    
    def train_model_for_system(self, system: str):
        """Train anomaly detection model for specific system"""
        try:
            # Generate training data
            training_data = self.generate_training_data(system)
            
            # Extract features
            feature_vectors = np.array([row['features'] for _, row in training_data.iterrows()])
            
            # Scale features
            scaler = StandardScaler()
            # Create feature names for proper scaling
            feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
            features_df = pd.DataFrame(feature_vectors, columns=feature_names)
            features_scaled = scaler.fit_transform(features_df)
            
            # Train isolation forest
            model = IsolationForest(
                contamination=0.1,  # 10% of data expected to be anomalous
                random_state=42,
                n_estimators=100
            )
            model.fit(features_scaled)
            
            # Save model and scaler
            model_file = os.path.join(self.models_path, f"{system}_anomaly_model.pkl")
            scaler_file = os.path.join(self.models_path, f"{system}_anomaly_scaler.pkl")
            
            joblib.dump(model, model_file)
            joblib.dump(scaler, scaler_file)
            
            # Store in memory
            self.models[system] = model
            self.scalers[system] = scaler
            
            print(f"Anomaly detection model trained for {system}")
            
        except Exception as e:
            print(f"Error training anomaly model for {system}: {e}")
    
    def detect_anomaly(self, system: str, sensor_data: Dict) -> Dict:
        """Detect anomalies in sensor data"""
        try:
            if system not in self.models:
                return {'error': f'No model available for {system}'}
            
            # Prepare features based on system
            if system == 'HVAC':
                features = [
                    sensor_data.get('temperature', 22),
                    sensor_data.get('humidity', 45),
                    sensor_data.get('pressure', 120),
                    sensor_data.get('energy_consumption', 50)
                ]
            elif system == 'Lighting':
                features = [
                    sensor_data.get('usage', 40),
                    sensor_data.get('efficiency', 85),
                    sensor_data.get('power', 30),
                    sensor_data.get('duration', 8)
                ]
            elif system == 'Security':
                features = [
                    sensor_data.get('alerts', 0),
                    sensor_data.get('access_attempts', 100),
                    sensor_data.get('camera_status', 1),
                    sensor_data.get('system_health', 95)
                ]
            elif system == 'Energy':
                features = [
                    sensor_data.get('consumption', 100),
                    sensor_data.get('voltage', 230),
                    sensor_data.get('current', 50),
                    sensor_data.get('power_factor', 0.95)
                ]
            else:  # Environmental
                features = [
                    sensor_data.get('co2', 400),
                    sensor_data.get('voc', 100),
                    sensor_data.get('pm25', 10),
                    sensor_data.get('air_quality', 85)
                ]
            
            # Scale features with proper feature names
            feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
            features_df = pd.DataFrame([features], columns=feature_names)
            features_scaled = self.scalers[system].transform(features_df)
            
            # Predict anomaly
            prediction = self.models[system].predict(features_scaled)[0]
            anomaly_score = self.models[system].score_samples(features_scaled)[0]
            
            # Determine if anomaly based on threshold and sensitivity
            is_anomaly = prediction == -1 or anomaly_score < self.threshold
            
            # Determine severity
            if is_anomaly:
                if anomaly_score < -0.5:
                    severity = 'High'
                elif anomaly_score < -0.2:
                    severity = 'Medium'
                else:
                    severity = 'Low'
            else:
                severity = 'None'
            
            return {
                'system': system,
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score,
                'severity': severity,
                'timestamp': datetime.now().isoformat(),
                'sensor_data': sensor_data
            }
            
        except Exception as e:
            return {'error': f'Anomaly detection error: {str(e)}'}
    
    def get_current_anomalies(self) -> Optional[pd.DataFrame]:
        """Get current anomalies for all systems"""
        try:
            anomalies = []
            
            for system in self.systems:
                # Generate current sensor data
                sensor_data = self._generate_current_sensor_data(system)
                
                # Detect anomalies
                anomaly_result = self.detect_anomaly(system, sensor_data)
                
                if 'error' not in anomaly_result and anomaly_result['is_anomaly']:
                    anomalies.append({
                        'system': system,
                        'severity': anomaly_result['severity'],
                        'anomaly_score': anomaly_result['anomaly_score'],
                        'timestamp': anomaly_result['timestamp'],
                        'description': self._get_anomaly_description(system, anomaly_result),
                        'impact': self._get_anomaly_impact(anomaly_result['severity']),
                        'count': 1
                    })
            
            if anomalies:
                return pd.DataFrame(anomalies)
            return None
            
        except Exception as e:
            print(f"Error getting current anomalies: {e}")
            return None
    
    def _generate_current_sensor_data(self, system: str) -> Dict:
        """Generate current sensor data for system"""
        if system == 'HVAC':
            return {
                'temperature': random.uniform(18, 28),
                'humidity': random.uniform(30, 60),
                'pressure': random.uniform(100, 140),
                'energy_consumption': random.uniform(30, 80)
            }
        elif system == 'Lighting':
            return {
                'usage': random.uniform(20, 80),
                'efficiency': random.uniform(70, 95),
                'power': random.uniform(20, 50),
                'duration': random.uniform(6, 12)
            }
        elif system == 'Security':
            return {
                'alerts': random.choices([0, 1], weights=[0.95, 0.05])[0],
                'access_attempts': random.randint(50, 200),
                'camera_status': 1,
                'system_health': random.uniform(85, 100)
            }
        elif system == 'Energy':
            return {
                'consumption': random.uniform(80, 150),
                'voltage': random.uniform(220, 240),
                'current': random.uniform(40, 70),
                'power_factor': random.uniform(0.9, 1.0)
            }
        else:  # Environmental
            return {
                'co2': random.uniform(350, 500),
                'voc': random.uniform(80, 150),
                'pm25': random.uniform(5, 20),
                'air_quality': random.uniform(75, 95)
            }
    
    def _get_anomaly_description(self, system: str, anomaly_result: Dict) -> str:
        """Get description for detected anomaly"""
        descriptions = {
            'HVAC': {
                'High': 'Critical HVAC system malfunction detected',
                'Medium': 'HVAC performance degradation observed',
                'Low': 'Minor HVAC system irregularity'
            },
            'Lighting': {
                'High': 'Lighting system failure or excessive usage',
                'Medium': 'Lighting efficiency below normal levels',
                'Low': 'Minor lighting system irregularity'
            },
            'Security': {
                'High': 'Security breach or system failure detected',
                'Medium': 'Unusual security activity or system degradation',
                'Low': 'Minor security system irregularity'
            },
            'Energy': {
                'High': 'Critical energy consumption anomaly',
                'Medium': 'Energy usage above normal levels',
                'Low': 'Minor energy consumption irregularity'
            },
            'Environmental': {
                'High': 'Critical environmental condition detected',
                'Medium': 'Environmental parameters outside normal range',
                'Low': 'Minor environmental irregularity'
            }
        }
        
        return descriptions.get(system, {}).get(anomaly_result['severity'], 'Unknown anomaly')
    
    def _get_anomaly_impact(self, severity: str) -> str:
        """Get impact description for anomaly severity"""
        impacts = {
            'High': 'Critical - Immediate action required',
            'Medium': 'Significant - Monitor closely',
            'Low': 'Minor - Routine attention needed'
        }
        return impacts.get(severity, 'Unknown impact')
    
    def get_anomaly_timeline(self) -> Optional[pd.DataFrame]:
        """Get anomaly timeline data"""
        try:
            # Generate timeline data for the past 24 hours
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            timeline_data = []
            current_time = start_time
            
            while current_time <= end_time:
                # Generate some random anomalies
                if random.random() < 0.1:  # 10% chance of anomaly per hour
                    system = random.choice(self.systems)
                    severity = random.choice(['Low', 'Medium', 'High'])
                    
                    timeline_data.append({
                        'timestamp': current_time,
                        'system': system,
                        'severity': severity,
                        'impact': random.uniform(0.1, 1.0)
                    })
                
                current_time += timedelta(hours=1)
            
            if timeline_data:
                return pd.DataFrame(timeline_data)
            return None
            
        except Exception as e:
            print(f"Error getting anomaly timeline: {e}")
            return None
    
    def update_settings(self, sensitivity: float, threshold: float):
        """Update anomaly detection settings"""
        self.sensitivity = sensitivity
        self.threshold = threshold
        print(f"Anomaly detection settings updated: sensitivity={sensitivity}, threshold={threshold}")
    
    def get_anomaly_statistics(self) -> Dict:
        """Get anomaly detection statistics"""
        try:
            # Generate some sample statistics
            total_anomalies = random.randint(5, 20)
            high_severity = random.randint(1, 5)
            medium_severity = random.randint(2, 8)
            low_severity = total_anomalies - high_severity - medium_severity
            
            stats = {
                'total_anomalies_24h': total_anomalies,
                'high_severity': high_severity,
                'medium_severity': medium_severity,
                'low_severity': low_severity,
                'detection_rate': random.uniform(0.85, 0.98),
                'false_positive_rate': random.uniform(0.02, 0.08),
                'average_response_time': random.uniform(30, 120),  # seconds
                'last_updated': datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting anomaly statistics: {e}")
            return {}
    
    def retrain_models(self):
        """Retrain all anomaly detection models"""
        try:
            for system in self.systems:
                print(f"Retraining anomaly detection model for {system}...")
                self.train_model_for_system(system)
            print("All anomaly detection models retrained successfully!")
        except Exception as e:
            print(f"Error retraining anomaly models: {e}")
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for anomaly detection models"""
        try:
            performance = {}
            
            for system in self.systems:
                if system in self.models:
                    # Generate test data
                    test_data = self.generate_training_data(system, days=7)
                    
                    # Extract features
                    feature_vectors = np.array([row['features'] for _, row in test_data.iterrows()])
                    
                    # Scale features
                    features_scaled = self.scalers[system].transform(feature_vectors)
                    
                    # Get anomaly scores
                    scores = self.models[system].score_samples(features_scaled)
                    
                    # Calculate performance metrics
                    avg_score = np.mean(scores)
                    std_score = np.std(scores)
                    min_score = np.min(scores)
                    max_score = np.max(scores)
                    
                    performance[system] = {
                        'average_score': avg_score,
                        'std_score': std_score,
                        'min_score': min_score,
                        'max_score': max_score,
                        'anomaly_threshold': self.threshold
                    }
            
            return performance
            
        except Exception as e:
            print(f"Error getting model performance: {e}")
            return {}

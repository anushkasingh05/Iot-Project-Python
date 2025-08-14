import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Optional, Dict, List
import random

class SensorDataIngestion:
    """Handles real-time IoT sensor data ingestion and processing"""
    
    def __init__(self):
        self.sensor_data_path = "data/sensor_data"
        self.sensor_types = ['temperature', 'humidity', 'hvac', 'lighting', 'security', 'energy']
        self.locations = ['Floor_1', 'Floor_2', 'Floor_3', 'Basement', 'Roof']
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Ensure data directory exists"""
        if not os.path.exists(self.sensor_data_path):
            os.makedirs(self.sensor_data_path)
    
    def generate_sensor_data(self, sensor_type: str, location: str, hours: int = 24) -> pd.DataFrame:
        """Generate realistic sensor data for a given type and location"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Generate timestamps
        timestamps = pd.date_range(start=start_time, end=end_time, freq='5min')
        
        # Generate values based on sensor type
        if sensor_type == 'temperature':
            base_temp = 22 + random.uniform(-2, 2)  # Base temperature around 22°C
            values = [base_temp + random.uniform(-3, 3) + 5 * np.sin(2 * np.pi * i / 288) for i in range(len(timestamps))]
            values = [max(15, min(30, v)) for v in values]  # Keep within reasonable range
            
        elif sensor_type == 'humidity':
            base_humidity = 45 + random.uniform(-5, 5)  # Base humidity around 45%
            values = [base_humidity + random.uniform(-10, 10) for _ in range(len(timestamps))]
            values = [max(20, min(80, v)) for v in values]  # Keep within reasonable range
            
        elif sensor_type == 'hvac':
            # HVAC efficiency percentage
            values = [random.uniform(75, 95) for _ in range(len(timestamps))]
            
        elif sensor_type == 'lighting':
            # Lighting usage percentage
            values = [random.uniform(20, 80) for _ in range(len(timestamps))]
            
        elif sensor_type == 'security':
            # Security system status (0=normal, 1=alert)
            values = [random.choices([0, 1], weights=[0.95, 0.05])[0] for _ in range(len(timestamps))]
            
        elif sensor_type == 'energy':
            # Energy consumption in kWh
            base_consumption = 50 + random.uniform(-10, 10)
            values = [base_consumption + random.uniform(-5, 5) for _ in range(len(timestamps))]
            values = [max(0, v) for v in values]
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'sensor_type': sensor_type,
            'location': location,
            'value': values,
            'status': self._generate_status(values, sensor_type)
        })
        
        return df
    
    def _generate_status(self, values: List[float], sensor_type: str) -> List[str]:
        """Generate status based on sensor values"""
        statuses = []
        
        for value in values:
            if sensor_type == 'temperature':
                if value < 18 or value > 26:
                    statuses.append('warning')
                else:
                    statuses.append('normal')
                    
            elif sensor_type == 'humidity':
                if value < 30 or value > 60:
                    statuses.append('warning')
                else:
                    statuses.append('normal')
                    
            elif sensor_type == 'hvac':
                if value < 80:
                    statuses.append('maintenance_needed')
                else:
                    statuses.append('optimal')
                    
            elif sensor_type == 'lighting':
                if value > 70:
                    statuses.append('high_usage')
                else:
                    statuses.append('normal')
                    
            elif sensor_type == 'security':
                if value == 1:
                    statuses.append('alert')
                else:
                    statuses.append('secure')
                    
            elif sensor_type == 'energy':
                if value > 60:
                    statuses.append('high_consumption')
                else:
                    statuses.append('normal')
        
        return statuses
    
    def get_current_data(self) -> Optional[pd.DataFrame]:
        """Get current sensor data for all sensors"""
        try:
            all_data = []
            
            for sensor_type in self.sensor_types:
                for location in self.locations:
                    # Generate recent data for each sensor
                    sensor_data = self.generate_sensor_data(sensor_type, location, hours=6)
                    all_data.append(sensor_data)
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                return combined_data.sort_values('timestamp')
            
            return None
            
        except Exception as e:
            print(f"Error getting current data: {e}")
            return None
    
    def get_system_health(self) -> Optional[pd.DataFrame]:
        """Get system health overview data"""
        try:
            systems = ['HVAC', 'Lighting', 'Security', 'Energy', 'Environmental']
            health_values = []
            
            for system in systems:
                # Generate health score based on sensor data
                if system == 'HVAC':
                    health = random.uniform(85, 95)
                elif system == 'Lighting':
                    health = random.uniform(80, 90)
                elif system == 'Security':
                    health = random.uniform(95, 100)
                elif system == 'Energy':
                    health = random.uniform(75, 85)
                elif system == 'Environmental':
                    health = random.uniform(80, 90)
                
                health_values.append({
                    'system': system,
                    'value': health
                })
            
            return pd.DataFrame(health_values)
            
        except Exception as e:
            print(f"Error getting system health: {e}")
            return None
    
    def save_sensor_data(self, data: pd.DataFrame, sensor_type: str, location: str):
        """Save sensor data to file"""
        try:
            filename = f"{sensor_type}_{location}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.sensor_data_path, filename)
            data.to_csv(filepath, index=False)
        except Exception as e:
            print(f"Error saving sensor data: {e}")
    
    def get_historical_data(self, sensor_type: str, location: str, days: int = 7) -> Optional[pd.DataFrame]:
        """Get historical sensor data"""
        try:
            # Generate historical data
            hours = days * 24
            historical_data = self.generate_sensor_data(sensor_type, location, hours)
            return historical_data
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return None
    
    def get_sensor_metadata(self) -> Dict:
        """Get metadata about all sensors"""
        metadata = {
            'total_sensors': len(self.sensor_types) * len(self.locations),
            'sensor_types': self.sensor_types,
            'locations': self.locations,
            'last_update': datetime.now().isoformat(),
            'data_frequency': '5 minutes',
            'data_retention': '30 days'
        }
        return metadata
    
    def process_real_time_stream(self, sensor_data: Dict) -> Dict:
        """Process incoming real-time sensor data"""
        try:
            # Add processing timestamp
            sensor_data['processed_at'] = datetime.now().isoformat()
            
            # Validate data
            if self._validate_sensor_data(sensor_data):
                # Apply any necessary transformations
                processed_data = self._transform_sensor_data(sensor_data)
                return processed_data
            else:
                return {'error': 'Invalid sensor data'}
                
        except Exception as e:
            return {'error': f'Processing error: {str(e)}'}
    
    def _validate_sensor_data(self, data: Dict) -> bool:
        """Validate incoming sensor data"""
        required_fields = ['sensor_id', 'sensor_type', 'location', 'value', 'timestamp']
        return all(field in data for field in required_fields)
    
    def _transform_sensor_data(self, data: Dict) -> Dict:
        """Transform sensor data as needed"""
        # Add any necessary transformations here
        data['value_normalized'] = self._normalize_value(data['value'], data['sensor_type'])
        return data
    
    def _normalize_value(self, value: float, sensor_type: str) -> float:
        """Normalize sensor values to 0-1 range"""
        if sensor_type == 'temperature':
            return (value - 15) / (30 - 15)  # Normalize 15-30°C to 0-1
        elif sensor_type == 'humidity':
            return value / 100  # Normalize 0-100% to 0-1
        elif sensor_type == 'hvac':
            return value / 100  # Normalize 0-100% to 0-1
        elif sensor_type == 'lighting':
            return value / 100  # Normalize 0-100% to 0-1
        elif sensor_type == 'security':
            return value  # Already 0 or 1
        elif sensor_type == 'energy':
            return min(value / 100, 1.0)  # Normalize with max at 100 kWh
        else:
            return value

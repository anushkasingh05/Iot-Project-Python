import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

class EnergyOptimization:
    """AI-driven energy optimization and efficiency recommendations"""
    
    def __init__(self):
        self.models_path = "models/energy"
        self.ensure_models_directory()
        self.systems = ['HVAC', 'Lighting', 'Security', 'Energy', 'Environmental']
        self.models = {}
        self.scalers = {}
        self.load_or_train_models()
    
    def ensure_models_directory(self):
        """Ensure models directory exists"""
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
    
    def load_or_train_models(self):
        """Load existing models or train new ones"""
        for system in self.systems:
            model_file = os.path.join(self.models_path, f"{system}_energy_model.pkl")
            scaler_file = os.path.join(self.models_path, f"{system}_energy_scaler.pkl")
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                # Load existing model
                self.models[system] = joblib.load(model_file)
                self.scalers[system] = joblib.load(scaler_file)
            else:
                # Train new model
                self.train_model_for_system(system)
    
    def generate_training_data(self, system: str, days: int = 90) -> pd.DataFrame:
        """Generate training data for energy optimization"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate daily data points
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = []
        for date in dates:
            # Generate realistic energy data with seasonal patterns
            day_of_year = date.dayofyear
            season_factor = np.sin(2 * np.pi * day_of_year / 365)
            
            if system == 'HVAC':
                # HVAC energy consumption patterns
                base_consumption = 80 + season_factor * 40  # Higher in summer/winter
                efficiency = 85 + random.uniform(-10, 10)
                temperature = 22 + season_factor * 8
                humidity = 45 + random.uniform(-15, 15)
                occupancy = random.uniform(0.3, 0.9)
                
                # Calculate optimized consumption
                optimized_consumption = base_consumption * (0.7 + 0.3 * (1 - efficiency/100))
                
            elif system == 'Lighting':
                # Lighting energy consumption patterns
                base_consumption = 60 + random.uniform(-10, 10)
                efficiency = 90 + random.uniform(-5, 5)
                daylight_hours = 8 + season_factor * 4  # More daylight in summer
                occupancy = random.uniform(0.4, 0.8)
                natural_light = max(0, 1 - abs(season_factor))  # More natural light in summer
                
                # Calculate optimized consumption
                optimized_consumption = base_consumption * (0.6 + 0.4 * (1 - natural_light))
                
            elif system == 'Security':
                # Security energy consumption (more stable)
                base_consumption = 30 + random.uniform(-5, 5)
                efficiency = 95 + random.uniform(-3, 3)
                system_health = 90 + random.uniform(-10, 10)
                uptime = random.uniform(0.98, 1.0)
                
                # Calculate optimized consumption
                optimized_consumption = base_consumption * (0.8 + 0.2 * (1 - efficiency/100))
                
            elif system == 'Energy':
                # Overall energy system
                base_consumption = 200 + season_factor * 60
                efficiency = 80 + random.uniform(-10, 10)
                peak_demand = base_consumption * (1 + random.uniform(0.1, 0.3))
                load_factor = random.uniform(0.6, 0.9)
                
                # Calculate optimized consumption
                optimized_consumption = base_consumption * (0.75 + 0.25 * (1 - efficiency/100))
                
            else:  # Environmental
                # Environmental systems
                base_consumption = 40 + random.uniform(-10, 10)
                efficiency = 85 + random.uniform(-10, 10)
                air_quality = 85 + random.uniform(-10, 10)
                ventilation_rate = random.uniform(0.5, 1.0)
                
                # Calculate optimized consumption
                optimized_consumption = base_consumption * (0.7 + 0.3 * (1 - efficiency/100))
            
            # Add some optimization potential
            optimization_potential = random.uniform(0.05, 0.25)  # 5-25% potential savings
            
            data.append({
                'date': date,
                'system': system,
                'consumption': base_consumption,
                'efficiency': efficiency,
                'optimized_consumption': optimized_consumption,
                'optimization_potential': optimization_potential,
                'temperature': temperature if system == 'HVAC' else None,
                'humidity': humidity if system == 'HVAC' else None,
                'occupancy': occupancy if system in ['HVAC', 'Lighting'] else None,
                'daylight_hours': daylight_hours if system == 'Lighting' else None,
                'natural_light': natural_light if system == 'Lighting' else None,
                'system_health': system_health if system == 'Security' else None,
                'uptime': uptime if system == 'Security' else None,
                'peak_demand': peak_demand if system == 'Energy' else None,
                'load_factor': load_factor if system == 'Energy' else None,
                'air_quality': air_quality if system == 'Environmental' else None,
                'ventilation_rate': ventilation_rate if system == 'Environmental' else None
            })
        
        return pd.DataFrame(data)
    
    def train_model_for_system(self, system: str):
        """Train energy optimization model for specific system"""
        try:
            # Generate training data
            training_data = self.generate_training_data(system)
            
            # Prepare features based on system
            if system == 'HVAC':
                feature_columns = ['temperature', 'humidity', 'occupancy', 'efficiency']
            elif system == 'Lighting':
                feature_columns = ['daylight_hours', 'natural_light', 'occupancy', 'efficiency']
            elif system == 'Security':
                feature_columns = ['system_health', 'uptime', 'efficiency']
            elif system == 'Energy':
                feature_columns = ['peak_demand', 'load_factor', 'efficiency']
            else:  # Environmental
                feature_columns = ['air_quality', 'ventilation_rate', 'efficiency']
            
            # Prepare target (optimization potential)
            X = training_data[feature_columns].fillna(0)
            y = training_data['optimization_potential']
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            # Save model and scaler
            model_file = os.path.join(self.models_path, f"{system}_energy_model.pkl")
            scaler_file = os.path.join(self.models_path, f"{system}_energy_scaler.pkl")
            
            joblib.dump(model, model_file)
            joblib.dump(scaler, scaler_file)
            
            # Store in memory
            self.models[system] = model
            self.scalers[system] = scaler
            
            print(f"Energy optimization model trained for {system}")
            
        except Exception as e:
            print(f"Error training energy model for {system}: {e}")
    
    def predict_optimization_potential(self, system: str, sensor_data: Dict) -> Dict:
        """Predict energy optimization potential for system"""
        try:
            if system not in self.models:
                return {'error': f'No model available for {system}'}
            
            # Prepare features based on system
            if system == 'HVAC':
                features = [
                    sensor_data.get('temperature', 22),
                    sensor_data.get('humidity', 45),
                    sensor_data.get('occupancy', 0.6),
                    sensor_data.get('efficiency', 85)
                ]
            elif system == 'Lighting':
                features = [
                    sensor_data.get('daylight_hours', 10),
                    sensor_data.get('natural_light', 0.5),
                    sensor_data.get('occupancy', 0.6),
                    sensor_data.get('efficiency', 90)
                ]
            elif system == 'Security':
                features = [
                    sensor_data.get('system_health', 90),
                    sensor_data.get('uptime', 0.99),
                    sensor_data.get('efficiency', 95)
                ]
            elif system == 'Energy':
                features = [
                    sensor_data.get('peak_demand', 250),
                    sensor_data.get('load_factor', 0.75),
                    sensor_data.get('efficiency', 80)
                ]
            else:  # Environmental
                features = [
                    sensor_data.get('air_quality', 85),
                    sensor_data.get('ventilation_rate', 0.75),
                    sensor_data.get('efficiency', 85)
                ]
            
            # Scale features with proper feature names
            feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
            features_df = pd.DataFrame([features], columns=feature_names[:len(features)])
            features_scaled = self.scalers[system].transform(features_df)
            
            # Make prediction
            optimization_potential = self.models[system].predict(features_scaled)[0]
            
            # Calculate potential savings
            current_consumption = sensor_data.get('consumption', 100)
            potential_savings = current_consumption * optimization_potential
            
            return {
                'system': system,
                'optimization_potential': optimization_potential,
                'potential_savings_kwh': potential_savings,
                'potential_savings_percent': optimization_potential * 100,
                'recommendations': self._get_optimization_recommendations(system, optimization_potential),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Optimization prediction error: {str(e)}'}
    
    def _get_optimization_recommendations(self, system: str, potential: float) -> List[str]:
        """Get optimization recommendations based on system and potential"""
        recommendations = []
        
        if potential > 0.15:  # High potential
            if system == 'HVAC':
                recommendations.extend([
                    "Implement smart thermostat controls",
                    "Optimize temperature setpoints based on occupancy",
                    "Improve HVAC system maintenance schedule",
                    "Consider variable speed drives for motors"
                ])
            elif system == 'Lighting':
                recommendations.extend([
                    "Install motion sensors for automatic control",
                    "Implement daylight harvesting systems",
                    "Upgrade to LED fixtures with smart controls",
                    "Optimize lighting schedules based on occupancy"
                ])
            elif system == 'Security':
                recommendations.extend([
                    "Implement power management for security systems",
                    "Optimize camera positioning and settings",
                    "Use energy-efficient security equipment",
                    "Implement smart access control systems"
                ])
            elif system == 'Energy':
                recommendations.extend([
                    "Implement demand response programs",
                    "Optimize load balancing across systems",
                    "Install energy storage systems",
                    "Implement peak demand management"
                ])
            else:  # Environmental
                recommendations.extend([
                    "Optimize ventilation rates based on occupancy",
                    "Implement demand-controlled ventilation",
                    "Improve air quality monitoring",
                    "Optimize environmental control systems"
                ])
        elif potential > 0.08:  # Medium potential
            recommendations.extend([
                "Review and optimize system settings",
                "Implement regular maintenance schedules",
                "Monitor system performance metrics",
                "Consider energy-efficient upgrades"
            ])
        else:  # Low potential
            recommendations.extend([
                "Continue current optimization practices",
                "Monitor for future optimization opportunities",
                "Maintain current efficiency levels"
            ])
        
        return recommendations
    
    def get_energy_data(self) -> Optional[pd.DataFrame]:
        """Get current energy consumption data for all systems"""
        try:
            energy_data = []
            
            for system in self.systems:
                # Generate current sensor data
                sensor_data = self._generate_current_sensor_data(system)
                
                # Get optimization prediction
                optimization_result = self.predict_optimization_potential(system, sensor_data)
                
                if 'error' not in optimization_result:
                    energy_data.append({
                        'system': system,
                        'consumption': sensor_data.get('consumption', 100),
                        'efficiency': sensor_data.get('efficiency', 85),
                        'optimization_potential': optimization_result['optimization_potential'],
                        'potential_savings': optimization_result['potential_savings_kwh']
                    })
            
            if energy_data:
                return pd.DataFrame(energy_data)
            return None
            
        except Exception as e:
            print(f"Error getting energy data: {e}")
            return None
    
    def _generate_current_sensor_data(self, system: str) -> Dict:
        """Generate current sensor data for system"""
        if system == 'HVAC':
            return {
                'temperature': random.uniform(20, 26),
                'humidity': random.uniform(35, 55),
                'occupancy': random.uniform(0.4, 0.8),
                'efficiency': random.uniform(75, 95),
                'consumption': random.uniform(60, 120)
            }
        elif system == 'Lighting':
            return {
                'daylight_hours': random.uniform(8, 14),
                'natural_light': random.uniform(0.3, 0.8),
                'occupancy': random.uniform(0.3, 0.7),
                'efficiency': random.uniform(80, 95),
                'consumption': random.uniform(40, 80)
            }
        elif system == 'Security':
            return {
                'system_health': random.uniform(85, 100),
                'uptime': random.uniform(0.95, 1.0),
                'efficiency': random.uniform(90, 98),
                'consumption': random.uniform(20, 40)
            }
        elif system == 'Energy':
            return {
                'peak_demand': random.uniform(200, 300),
                'load_factor': random.uniform(0.6, 0.9),
                'efficiency': random.uniform(70, 90),
                'consumption': random.uniform(150, 250)
            }
        else:  # Environmental
            return {
                'air_quality': random.uniform(75, 95),
                'ventilation_rate': random.uniform(0.5, 1.0),
                'efficiency': random.uniform(75, 90),
                'consumption': random.uniform(30, 60)
            }
    
    def get_efficiency_trends(self) -> Optional[pd.DataFrame]:
        """Get efficiency trends over time"""
        try:
            # Generate trend data for the past 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            trend_data = []
            for date in dates:
                for system in self.systems:
                    # Generate efficiency data with some trend
                    base_efficiency = 85
                    trend_factor = (date - start_date).days / 30  # 0 to 1 over 30 days
                    
                    # Add some improvement trend
                    efficiency = base_efficiency + trend_factor * 5 + random.uniform(-3, 3)
                    efficiency = max(70, min(98, efficiency))  # Keep within reasonable range
                    
                    trend_data.append({
                        'date': date,
                        'system': system,
                        'efficiency': efficiency
                    })
            
            if trend_data:
                return pd.DataFrame(trend_data)
            return None
            
        except Exception as e:
            print(f"Error getting efficiency trends: {e}")
            return None
    
    def get_recommendations(self) -> List[Dict]:
        """Get AI-driven optimization recommendations"""
        try:
            recommendations = []
            
            # Get current energy data
            energy_data = self.get_energy_data()
            
            if energy_data is not None:
                for _, row in energy_data.iterrows():
                    system = row['system']
                    potential = row['optimization_potential']
                    savings = row['potential_savings']
                    
                    if potential > 0.1:  # Only show recommendations for significant potential
                        rec = {
                            'title': f"Optimize {system} System",
                            'impact': f"Potential {potential*100:.1f}% energy savings",
                            'implementation': self._get_implementation_steps(system),
                            'savings': f"Estimated {savings:.1f} kWh savings per day",
                            'priority': 'High' if potential > 0.15 else 'Medium',
                            'cost_estimate': self._estimate_implementation_cost(system, potential)
                        }
                        recommendations.append(rec)
            
            # Add some general recommendations
            general_recommendations = [
                {
                    'title': "Implement Building Energy Management System",
                    'impact': "Centralized control and optimization of all systems",
                    'implementation': "Install BMS software and integrate with existing systems",
                    'savings': "Estimated 10-15% overall energy savings",
                    'priority': 'High',
                    'cost_estimate': "$50,000 - $100,000"
                },
                {
                    'title': "Conduct Energy Audit",
                    'impact': "Identify additional optimization opportunities",
                    'implementation': "Hire energy consultant for comprehensive building audit",
                    'savings': "Variable based on findings",
                    'priority': 'Medium',
                    'cost_estimate': "$5,000 - $15,000"
                },
                {
                    'title': "Employee Energy Awareness Program",
                    'impact': "Behavioral changes leading to energy savings",
                    'implementation': "Training sessions and energy-saving campaigns",
                    'savings': "Estimated 5-10% behavioral savings",
                    'priority': 'Low',
                    'cost_estimate': "$2,000 - $5,000"
                }
            ]
            
            recommendations.extend(general_recommendations)
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []
    
    def _get_implementation_steps(self, system: str) -> str:
        """Get implementation steps for system optimization"""
        steps = {
            'HVAC': "1. Install smart thermostats\n2. Implement occupancy sensors\n3. Optimize temperature setpoints\n4. Schedule regular maintenance",
            'Lighting': "1. Install motion sensors\n2. Implement daylight harvesting\n3. Upgrade to LED fixtures\n4. Configure smart controls",
            'Security': "1. Implement power management\n2. Optimize camera settings\n3. Install energy-efficient equipment\n4. Configure smart access control",
            'Energy': "1. Implement demand response\n2. Install energy storage\n3. Optimize load balancing\n4. Configure peak management",
            'Environmental': "1. Install demand-controlled ventilation\n2. Optimize air quality monitoring\n3. Implement smart environmental controls\n4. Schedule regular maintenance"
        }
        return steps.get(system, "Contact energy consultant for specific recommendations")
    
    def _estimate_implementation_cost(self, system: str, potential: float) -> str:
        """Estimate implementation cost for optimization"""
        base_costs = {
            'HVAC': 15000,
            'Lighting': 25000,
            'Security': 10000,
            'Energy': 50000,
            'Environmental': 12000
        }
        
        base_cost = base_costs.get(system, 20000)
        
        # Adjust cost based on potential savings
        if potential > 0.15:
            cost_multiplier = 1.2  # Higher cost for high potential
        elif potential > 0.1:
            cost_multiplier = 1.0
        else:
            cost_multiplier = 0.8
        
        estimated_cost = base_cost * cost_multiplier
        
        return f"${estimated_cost:,.0f} - ${estimated_cost*1.3:,.0f}"
    
    def get_energy_statistics(self) -> Dict:
        """Get energy optimization statistics"""
        try:
            # Generate sample statistics
            total_consumption = random.uniform(800, 1200)  # kWh per day
            total_savings = total_consumption * random.uniform(0.1, 0.2)  # 10-20% savings
            efficiency_score = random.uniform(75, 90)
            
            stats = {
                'total_daily_consumption_kwh': total_consumption,
                'total_daily_savings_kwh': total_savings,
                'overall_efficiency_score': efficiency_score,
                'optimization_opportunities': random.randint(3, 8),
                'estimated_monthly_savings': total_savings * 30,
                'estimated_annual_savings': total_savings * 365,
                'last_updated': datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting energy statistics: {e}")
            return {}
    
    def retrain_models(self):
        """Retrain all energy optimization models"""
        try:
            for system in self.systems:
                print(f"Retraining energy optimization model for {system}...")
                self.train_model_for_system(system)
            print("All energy optimization models retrained successfully!")
        except Exception as e:
            print(f"Error retraining energy models: {e}")

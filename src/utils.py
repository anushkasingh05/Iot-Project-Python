import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import random
from typing import Dict, List

def generate_sample_data():
    """Generate sample sensor data files"""
    try:
        # Create data directories
        data_dirs = [
            "data/sensor_data",
            "data/manuals", 
            "data/building_specs"
        ]
        
        for dir_path in data_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # Generate sample sensor data
        generate_hvac_sensor_data()
        generate_lighting_sensor_data()
        generate_environmental_sensor_data()
        generate_security_sensor_data()
        generate_energy_sensor_data()
        
        print("Sample sensor data generated successfully!")
        
    except Exception as e:
        print(f"Error generating sample data: {e}")

def generate_hvac_sensor_data():
    """Generate sample HVAC sensor data"""
    try:
        # Generate 7 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')
        
        data = []
        for timestamp in timestamps:
            # Generate realistic HVAC data
            base_temp = 22 + 5 * np.sin(2 * np.pi * timestamp.hour / 24)  # Daily cycle
            base_humidity = 45 + 10 * np.sin(2 * np.pi * timestamp.hour / 24)
            
            data.append({
                'timestamp': timestamp,
                'temperature': base_temp + random.uniform(-2, 2),
                'humidity': base_humidity + random.uniform(-5, 5),
                'pressure': 120 + random.uniform(-10, 10),
                'airflow': random.uniform(80, 95),
                'efficiency': random.uniform(75, 95),
                'status': random.choice(['normal', 'maintenance_needed', 'optimal'])
            })
        
        df = pd.DataFrame(data)
        df.to_csv('data/sensor_data/hvac_sensors.csv', index=False)
        
    except Exception as e:
        print(f"Error generating HVAC data: {e}")

def generate_lighting_sensor_data():
    """Generate sample lighting sensor data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')
        
        data = []
        for timestamp in timestamps:
            # Lighting usage based on time of day
            hour = timestamp.hour
            if 6 <= hour <= 22:  # Business hours
                usage = random.uniform(60, 90)
            else:  # Off hours
                usage = random.uniform(10, 30)
            
            data.append({
                'timestamp': timestamp,
                'usage_percentage': usage,
                'efficiency': random.uniform(85, 98),
                'power_consumption': usage * random.uniform(0.8, 1.2),
                'occupancy': random.uniform(0.3, 0.9),
                'natural_light': random.uniform(0.1, 0.8),
                'status': random.choice(['normal', 'high_usage', 'optimal'])
            })
        
        df = pd.DataFrame(data)
        df.to_csv('data/sensor_data/lighting_sensors.csv', index=False)
        
    except Exception as e:
        print(f"Error generating lighting data: {e}")

def generate_environmental_sensor_data():
    """Generate sample environmental sensor data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')
        
        data = []
        for timestamp in timestamps:
            data.append({
                'timestamp': timestamp,
                'co2_level': random.uniform(350, 600),
                'voc_level': random.uniform(50, 200),
                'pm25': random.uniform(5, 25),
                'air_quality_index': random.uniform(70, 95),
                'ventilation_rate': random.uniform(0.5, 1.0),
                'status': random.choice(['normal', 'warning', 'optimal'])
            })
        
        df = pd.DataFrame(data)
        df.to_csv('data/sensor_data/environmental_sensors.csv', index=False)
        
    except Exception as e:
        print(f"Error generating environmental data: {e}")

def generate_security_sensor_data():
    """Generate sample security sensor data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')
        
        data = []
        for timestamp in timestamps:
            # Security alerts are rare
            alert = random.choices([0, 1], weights=[0.98, 0.02])[0]
            
            data.append({
                'timestamp': timestamp,
                'alerts': alert,
                'access_attempts': random.randint(50, 150),
                'camera_status': 1,  # Always on
                'system_health': random.uniform(90, 100),
                'uptime': random.uniform(0.98, 1.0),
                'status': 'alert' if alert else 'secure'
            })
        
        df = pd.DataFrame(data)
        df.to_csv('data/sensor_data/security_sensors.csv', index=False)
        
    except Exception as e:
        print(f"Error generating security data: {e}")

def generate_energy_sensor_data():
    """Generate sample energy sensor data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')
        
        data = []
        for timestamp in timestamps:
            # Energy consumption varies by time of day
            hour = timestamp.hour
            if 8 <= hour <= 18:  # Peak hours
                consumption = random.uniform(150, 250)
            else:  # Off-peak hours
                consumption = random.uniform(80, 150)
            
            data.append({
                'timestamp': timestamp,
                'consumption_kwh': consumption,
                'voltage': random.uniform(220, 240),
                'current': random.uniform(40, 80),
                'power_factor': random.uniform(0.9, 1.0),
                'peak_demand': consumption * random.uniform(1.1, 1.3),
                'status': 'high_consumption' if consumption > 200 else 'normal'
            })
        
        df = pd.DataFrame(data)
        df.to_csv('data/sensor_data/energy_sensors.csv', index=False)
        
    except Exception as e:
        print(f"Error generating energy data: {e}")

def create_sample_manuals():
    """Create sample maintenance manuals and documentation"""
    try:
        # Create sample manual files
        manuals = {
            'hvac_maintenance_manual.txt': '''
HVAC System Maintenance Manual
Version 2.1
Last Updated: 2024-01-15

1. DAILY CHECKS
   - Monitor temperature readings
   - Check humidity levels
   - Verify system status indicators
   - Review energy consumption

2. WEEKLY MAINTENANCE
   - Clean air filters
   - Check thermostat calibration
   - Inspect ductwork for leaks
   - Monitor refrigerant levels

3. MONTHLY MAINTENANCE
   - Clean evaporator and condenser coils
   - Lubricate motors and bearings
   - Check electrical connections
   - Test emergency shutdown procedures

4. QUARTERLY MAINTENANCE
   - Comprehensive system inspection
   - Performance testing and optimization
   - Update maintenance records
   - Review energy efficiency metrics

5. ANNUAL MAINTENANCE
   - Complete system overhaul
   - Replace worn components
   - Update control software
   - Energy audit and optimization

TROUBLESHOOTING GUIDE:
- High energy consumption: Check filters and coils
- Uneven cooling: Verify duct balance and thermostat
- Unusual noises: Inspect motors and bearings
- System failure: Check electrical connections and fuses

EMERGENCY PROCEDURES:
- System shutdown: Use emergency stop button
- Fire alarm: Follow building evacuation procedures
- Power failure: Activate backup systems
- Gas leak: Evacuate and contact emergency services
            ''',
            
            'lighting_system_manual.txt': '''
Lighting System Operation Manual
Version 1.5
Last Updated: 2024-01-10

1. SYSTEM OVERVIEW
   - LED fixtures with smart controls
   - Motion sensors for occupancy detection
   - Daylight harvesting capabilities
   - Emergency lighting backup

2. DAILY OPERATIONS
   - Check system status indicators
   - Monitor energy consumption
   - Verify sensor functionality
   - Review occupancy patterns

3. WEEKLY MAINTENANCE
   - Clean LED fixtures
   - Test motion sensors
   - Check emergency lighting
   - Update lighting schedules

4. MONTHLY MAINTENANCE
   - Calibrate daylight sensors
   - Inspect electrical connections
   - Test dimming functionality
   - Review energy efficiency

5. QUARTERLY MAINTENANCE
   - Comprehensive system testing
   - Update control software
   - Performance optimization
   - Energy audit

ENERGY SAVING TIPS:
- Use daylight harvesting when possible
- Implement occupancy-based control
- Optimize lighting schedules
- Regular maintenance for peak efficiency

TROUBLESHOOTING:
- Flickering lights: Check electrical connections
- Sensor not working: Clean and recalibrate
- High energy usage: Review schedules and settings
- Emergency light failure: Check battery backup
            ''',
            
            'security_system_manual.txt': '''
Security System Operation Manual
Version 3.0
Last Updated: 2024-01-20

1. SYSTEM COMPONENTS
   - IP cameras with night vision
   - Access control systems
   - Motion detectors
   - Fire and security alarms
   - Backup power systems

2. DAILY MONITORING
   - Check camera feeds
   - Review access logs
   - Monitor alarm status
   - Verify system health

3. WEEKLY MAINTENANCE
   - Clean camera lenses
   - Test access control
   - Check motion detectors
   - Verify backup systems

4. MONTHLY MAINTENANCE
   - Update security software
   - Test alarm systems
   - Review access permissions
   - Backup system data

5. QUARTERLY MAINTENANCE
   - Comprehensive system test
   - Update firmware
   - Review security protocols
   - Staff training updates

EMERGENCY PROCEDURES:
- Security breach: Contact security immediately
- Fire alarm: Follow evacuation procedures
- Power failure: Activate backup systems
- System failure: Use manual override

ACCESS CONTROL:
- Card readers at all entrances
- Biometric systems for restricted areas
- Visitor management system
- Audit trail for all access
            ''',
            
            'building_specifications.json': json.dumps({
                "building_name": "Smart Office Building",
                "total_floors": 5,
                "total_area_sqft": 50000,
                "construction_year": 2020,
                "systems": {
                    "hvac": {
                        "units": 3,
                        "type": "Variable Air Volume",
                        "capacity": "150 tons",
                        "efficiency": "SEER 16"
                    },
                    "lighting": {
                        "fixtures": 500,
                        "type": "LED with smart controls",
                        "efficiency": "120 lumens/watt",
                        "controls": "Motion sensors and daylight harvesting"
                    },
                    "security": {
                        "cameras": 25,
                        "access_points": 8,
                        "alarm_zones": 12,
                        "backup_power": "UPS and generator"
                    },
                    "electrical": {
                        "main_service": "400A",
                        "backup_generator": "200kW",
                        "subpanels": 15,
                        "emergency_power": "Life safety systems"
                    },
                    "plumbing": {
                        "fixtures": 50,
                        "water_conservation": "Low-flow fixtures",
                        "backflow_prevention": "All fixtures"
                    }
                },
                "maintenance_schedule": {
                    "daily": ["System monitoring", "Status checks"],
                    "weekly": ["Filter changes", "Sensor testing"],
                    "monthly": ["Performance testing", "Software updates"],
                    "quarterly": ["Comprehensive inspection", "Energy audit"],
                    "annually": ["System overhaul", "Major upgrades"]
                }
            }, indent=2)
        }
        
        # Write manual files
        for filename, content in manuals.items():
            filepath = os.path.join('data/manuals', filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print("Sample manuals created successfully!")
        
    except Exception as e:
        print(f"Error creating sample manuals: {e}")

def create_sample_building_specs():
    """Create sample building specification files"""
    try:
        # Create building specifications
        specs = {
            'floor_plans': {
                'floor_1': 'Office space with open plan layout',
                'floor_2': 'Conference rooms and meeting spaces',
                'floor_3': 'Executive offices and boardroom',
                'floor_4': 'IT department and server room',
                'floor_5': 'Penthouse and rooftop terrace'
            },
            'system_specifications': {
                'hvac': {
                    'manufacturer': 'Carrier',
                    'model': '48TC',
                    'capacity': '150 tons',
                    'efficiency': 'SEER 16',
                    'installation_date': '2020-03-15'
                },
                'lighting': {
                    'manufacturer': 'Philips',
                    'model': 'Smart LED',
                    'wattage': '15W per fixture',
                    'lifetime': '50000 hours',
                    'installation_date': '2020-03-20'
                },
                'security': {
                    'manufacturer': 'Honeywell',
                    'model': 'ProWatch',
                    'cameras': '25 IP cameras',
                    'storage': '30 days retention',
                    'installation_date': '2020-03-10'
                }
            }
        }
        
        # Save specifications
        filepath = os.path.join('data/building_specs', 'system_specifications.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(specs, f, indent=2)
        
        print("Sample building specifications created successfully!")
        
    except Exception as e:
        print(f"Error creating building specifications: {e}")

def get_system_status():
    """Get overall system status"""
    try:
        status = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': random.uniform(85, 95),
            'systems': {
                'hvac': {
                    'status': random.choice(['optimal', 'normal', 'maintenance_needed']),
                    'health': random.uniform(80, 95),
                    'last_maintenance': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
                },
                'lighting': {
                    'status': random.choice(['optimal', 'normal', 'high_usage']),
                    'health': random.uniform(85, 98),
                    'last_maintenance': (datetime.now() - timedelta(days=random.randint(1, 45))).isoformat()
                },
                'security': {
                    'status': random.choice(['secure', 'normal', 'alert']),
                    'health': random.uniform(90, 100),
                    'last_maintenance': (datetime.now() - timedelta(days=random.randint(1, 60))).isoformat()
                },
                'energy': {
                    'status': random.choice(['optimal', 'normal', 'high_consumption']),
                    'health': random.uniform(75, 90),
                    'last_maintenance': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
                },
                'environmental': {
                    'status': random.choice(['optimal', 'normal', 'warning']),
                    'health': random.uniform(80, 95),
                    'last_maintenance': (datetime.now() - timedelta(days=random.randint(1, 40))).isoformat()
                }
            },
            'alerts': {
                'active': random.randint(0, 3),
                'critical': random.randint(0, 1),
                'maintenance_due': random.randint(1, 5)
            },
            'energy_metrics': {
                'current_consumption': random.uniform(800, 1200),
                'daily_average': random.uniform(900, 1100),
                'efficiency_score': random.uniform(75, 90),
                'cost_today': random.uniform(200, 400)
            }
        }
        
        return status
        
    except Exception as e:
        print(f"Error getting system status: {e}")
        return {}

def generate_historical_data(days: int = 30):
    """Generate historical data for analysis"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate daily data points
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        historical_data = []
        for date in dates:
            # Generate realistic daily metrics
            day_of_week = date.weekday()
            is_weekend = day_of_week >= 5
            
            # Adjust metrics based on day type
            if is_weekend:
                occupancy = random.uniform(0.1, 0.3)
                energy_consumption = random.uniform(400, 600)
            else:
                occupancy = random.uniform(0.6, 0.9)
                energy_consumption = random.uniform(800, 1200)
            
            # Add seasonal effects
            day_of_year = date.dayofyear
            season_factor = np.sin(2 * np.pi * day_of_year / 365)
            
            # HVAC consumption varies with season
            hvac_consumption = energy_consumption * (0.4 + 0.2 * abs(season_factor))
            
            historical_data.append({
                'date': date,
                'occupancy_rate': occupancy,
                'total_energy_kwh': energy_consumption,
                'hvac_energy_kwh': hvac_consumption,
                'lighting_energy_kwh': energy_consumption * 0.25,
                'security_energy_kwh': energy_consumption * 0.05,
                'environmental_energy_kwh': energy_consumption * 0.1,
                'other_energy_kwh': energy_consumption * 0.2,
                'temperature_avg': 22 + season_factor * 8,
                'humidity_avg': 45 + random.uniform(-10, 10),
                'system_health_avg': random.uniform(85, 95),
                'maintenance_events': random.randint(0, 2),
                'alerts_generated': random.randint(0, 5)
            })
        
        df = pd.DataFrame(historical_data)
        df.to_csv('data/historical_data.csv', index=False)
        
        print(f"Historical data generated for {days} days")
        return df
        
    except Exception as e:
        print(f"Error generating historical data: {e}")
        return None

def calculate_metrics(data: pd.DataFrame) -> Dict:
    """Calculate key performance metrics from data"""
    try:
        metrics = {
            'total_records': len(data),
            'date_range': {
                'start': data['date'].min().strftime('%Y-%m-%d'),
                'end': data['date'].max().strftime('%Y-%m-%d')
            },
            'energy_metrics': {
                'total_consumption': data['total_energy_kwh'].sum(),
                'average_daily': data['total_energy_kwh'].mean(),
                'peak_consumption': data['total_energy_kwh'].max(),
                'lowest_consumption': data['total_energy_kwh'].min()
            },
            'occupancy_metrics': {
                'average_occupancy': data['occupancy_rate'].mean(),
                'peak_occupancy': data['occupancy_rate'].max(),
                'weekend_average': data[data['date'].dt.weekday >= 5]['occupancy_rate'].mean(),
                'weekday_average': data[data['date'].dt.weekday < 5]['occupancy_rate'].mean()
            },
            'system_health': {
                'average_health': data['system_health_avg'].mean(),
                'health_trend': 'improving' if data['system_health_avg'].iloc[-1] > data['system_health_avg'].iloc[0] else 'declining'
            },
            'maintenance_metrics': {
                'total_events': data['maintenance_events'].sum(),
                'average_alerts': data['alerts_generated'].mean(),
                'maintenance_frequency': data['maintenance_events'].sum() / len(data) * 30  # per month
            }
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {}

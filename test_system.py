#!/usr/bin/env python3
"""
Test script for IoT Smart Building RAG System
"""

import sys
import os
import traceback

def test_imports():
    """Test if all modules can be imported"""
    print("🔍 Testing module imports...")
    
    modules = [
        'src.data_ingestion',
        'src.rag_engine', 
        'src.predictive_maintenance',
        'src.anomaly_detection',
        'src.energy_optimization',
        'src.utils'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            return False
    
    return True

def test_data_ingestion():
    """Test data ingestion module"""
    print("\n📊 Testing data ingestion...")
    try:
        from src.data_ingestion import SensorDataIngestion
        
        sensor_ingestion = SensorDataIngestion()
        
        # Test getting current data
        data = sensor_ingestion.get_current_data()
        if data is not None and len(data) > 0:
            print(f"✅ Data ingestion: {len(data)} records generated")
        else:
            print("❌ Data ingestion: No data generated")
            return False
        
        # Test system health
        health = sensor_ingestion.get_system_health()
        if health is not None and len(health) > 0:
            print(f"✅ System health: {len(health)} systems monitored")
        else:
            print("❌ System health: No health data")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data ingestion error: {e}")
        return False

def test_rag_engine():
    """Test RAG engine"""
    print("\n🔍 Testing RAG engine...")
    try:
        from src.rag_engine import RAGEngine
        
        rag_engine = RAGEngine()
        
        # Test search
        results = rag_engine.search("HVAC maintenance")
        if results and len(results) > 0:
            print(f"✅ RAG search: {len(results)} results found")
        else:
            print("❌ RAG search: No results")
            return False
        
        # Test statistics
        stats = rag_engine.get_statistics()
        if stats:
            print(f"✅ RAG statistics: {stats.get('total_documents', 0)} documents")
        else:
            print("❌ RAG statistics: No stats")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ RAG engine error: {e}")
        return False

def test_predictive_maintenance():
    """Test predictive maintenance"""
    print("\n🔮 Testing predictive maintenance...")
    try:
        from src.predictive_maintenance import PredictiveMaintenance
        
        pm = PredictiveMaintenance()
        
        # Test predictions
        predictions = pm.get_predictions()
        if predictions is not None and len(predictions) > 0:
            print(f"✅ Predictions: {len(predictions)} equipment predictions")
        else:
            print("❌ Predictions: No predictions generated")
            return False
        
        # Test maintenance schedule
        schedule = pm.get_maintenance_schedule()
        if schedule is not None and len(schedule) > 0:
            print(f"✅ Maintenance schedule: {len(schedule)} scheduled items")
        else:
            print("❌ Maintenance schedule: No schedule generated")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Predictive maintenance error: {e}")
        return False

def test_anomaly_detection():
    """Test anomaly detection"""
    print("\n🚨 Testing anomaly detection...")
    try:
        from src.anomaly_detection import AnomalyDetection
        
        ad = AnomalyDetection()
        
        # Test anomaly detection
        anomalies = ad.get_current_anomalies()
        if anomalies is not None:
            print(f"✅ Anomaly detection: {len(anomalies) if not anomalies.empty else 0} anomalies")
        else:
            print("❌ Anomaly detection: No anomaly data")
            return False
        
        # Test statistics
        stats = ad.get_anomaly_statistics()
        if stats:
            print(f"✅ Anomaly statistics: {stats.get('total_anomalies_24h', 0)} anomalies in 24h")
        else:
            print("❌ Anomaly statistics: No stats")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Anomaly detection error: {e}")
        return False

def test_energy_optimization():
    """Test energy optimization"""
    print("\n⚡ Testing energy optimization...")
    try:
        from src.energy_optimization import EnergyOptimization
        
        eo = EnergyOptimization()
        
        # Test energy data
        energy_data = eo.get_energy_data()
        if energy_data is not None and len(energy_data) > 0:
            print(f"✅ Energy data: {len(energy_data)} systems analyzed")
        else:
            print("❌ Energy data: No energy data")
            return False
        
        # Test recommendations
        recommendations = eo.get_recommendations()
        if recommendations and len(recommendations) > 0:
            print(f"✅ Energy recommendations: {len(recommendations)} recommendations")
        else:
            print("❌ Energy recommendations: No recommendations")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Energy optimization error: {e}")
        return False

def test_utils():
    """Test utility functions"""
    print("\n🛠️ Testing utility functions...")
    try:
        from src.utils import get_system_status, generate_historical_data
        
        # Test system status
        status = get_system_status()
        if status:
            print(f"✅ System status: Overall health {status.get('overall_health', 0):.1f}%")
        else:
            print("❌ System status: No status data")
            return False
        
        # Test historical data generation
        hist_data = generate_historical_data(days=7)
        if hist_data is not None and len(hist_data) > 0:
            print(f"✅ Historical data: {len(hist_data)} days of data")
        else:
            print("❌ Historical data: No historical data")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Utility functions error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 IoT Smart Building RAG System - System Test")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Data Ingestion", test_data_ingestion),
        ("RAG Engine", test_rag_engine),
        ("Predictive Maintenance", test_predictive_maintenance),
        ("Anomaly Detection", test_anomaly_detection),
        ("Energy Optimization", test_energy_optimization),
        ("Utility Functions", test_utils)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to run.")
        print("\nTo start the application, run:")
        print("python run.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

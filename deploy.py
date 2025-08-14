#!/usr/bin/env python3
"""
IoT Smart Building RAG System Deployment Script
This script sets up and runs the complete IoT RAG system.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "data/sensor_data",
        "data/manuals",
        "data/building_specs",
        "models",
        "models/anomaly",
        "models/energy",
        "models/predictive",
        "chroma_db",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

def setup_environment():
    """Set up environment configuration"""
    env_file = ".env"
    env_example = "env_example.txt"
    
    if not os.path.exists(env_file):
        if os.path.exists(env_example):
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file with your OpenAI API key")
        else:
            print("⚠️  No .env file found. Please create one manually.")
    else:
        print("✅ .env file already exists")

def generate_sample_data():
    """Generate sample data for the system"""
    print("📊 Generating sample data...")
    try:
        from src.utils import generate_sample_data, create_sample_manuals
        generate_sample_data()
        create_sample_manuals()
        print("✅ Sample data generated successfully")
    except Exception as e:
        print(f"❌ Error generating sample data: {e}")

def initialize_models():
    """Initialize ML models"""
    print("🤖 Initializing ML models...")
    try:
        from src.predictive_maintenance import PredictiveMaintenance
        from src.anomaly_detection import AnomalyDetection
        from src.energy_optimization import EnergyOptimization
        
        # Initialize models (this will train them if they don't exist)
        print("  - Initializing predictive maintenance models...")
        PredictiveMaintenance()
        
        print("  - Initializing anomaly detection models...")
        AnomalyDetection()
        
        print("  - Initializing energy optimization models...")
        EnergyOptimization()
        
        print("✅ ML models initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing models: {e}")

def initialize_rag():
    """Initialize RAG system"""
    print("🔍 Initializing RAG system...")
    try:
        from src.rag_engine import RAGEngine
        rag_engine = RAGEngine()
        print("✅ RAG system initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")

def run_system_checks():
    """Run system health checks"""
    print("🔍 Running system health checks...")
    
    # Check if all required files exist
    required_files = [
        "app.py",
        "requirements.txt",
        "src/__init__.py",
        "src/data_ingestion.py",
        "src/rag_engine.py",
        "src/predictive_maintenance.py",
        "src/anomaly_detection.py",
        "src/energy_optimization.py",
        "src/utils.py"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
    
    # Check if directories exist
    required_dirs = ["data", "models", "chroma_db"]
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory: {dir_path}")
        else:
            print(f"❌ Missing directory: {dir_path}")

def start_application():
    """Start the Streamlit application"""
    print("🚀 Starting IoT Smart Building RAG System...")
    print("📱 The application will be available at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

def main():
    """Main deployment function"""
    print("🏢 IoT Smart Building RAG System Deployment")
    print("=" * 50)
    
    # Step 1: Check Python version
    check_python_version()
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Install dependencies
    install_dependencies()
    
    # Step 4: Setup environment
    setup_environment()
    
    # Step 5: Generate sample data
    generate_sample_data()
    
    # Step 6: Initialize models
    initialize_models()
    
    # Step 7: Initialize RAG
    initialize_rag()
    
    # Step 8: Run system checks
    run_system_checks()
    
    print("\n🎉 Deployment completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your OpenAI API key")
    print("2. Run: python deploy.py --start")
    print("3. Open http://localhost:8501 in your browser")
    
    # Check if --start flag is provided
    if "--start" in sys.argv:
        start_application()

if __name__ == "__main__":
    main()

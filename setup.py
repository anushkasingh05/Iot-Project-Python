#!/usr/bin/env python3
"""
Setup script for IoT Smart Building RAG System
"""

import os
import sys
import subprocess
import shutil

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = [
        "data/sensor_data",
        "data/manuals", 
        "data/building_specs",
        "models/anomaly",
        "models/energy",
        "models/predictive",
        "chroma_db",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")

def setup_environment():
    """Setup environment file"""
    print("⚙️ Setting up environment...")
    if not os.path.exists('.env'):
        if os.path.exists('env_example.txt'):
            shutil.copy('env_example.txt', '.env')
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file with your OpenAI API key")
        else:
            print("⚠️  No env_example.txt found")
    else:
        print("✅ .env file already exists")

def generate_sample_data():
    """Generate sample data"""
    print("📊 Generating sample data...")
    try:
        from src.utils import generate_sample_data, create_sample_manuals
        generate_sample_data()
        create_sample_manuals()
        print("✅ Sample data generated")
        return True
    except Exception as e:
        print(f"❌ Error generating sample data: {e}")
        return False

def main():
    """Main setup function"""
    print("🏢 IoT Smart Building RAG System Setup")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup environment
    setup_environment()
    
    # Generate sample data
    if not generate_sample_data():
        print("⚠️  Sample data generation failed, but setup can continue")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your OpenAI API key")
    print("2. Run: python test_system.py")
    print("3. Run: python run.py")
    print("4. Open http://localhost:8501 in your browser")

if __name__ == "__main__":
    main()

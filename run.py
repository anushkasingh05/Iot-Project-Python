#!/usr/bin/env python3
"""
Simple run script for IoT Smart Building RAG System
"""

import os
import sys
import subprocess

def main():
    print("🏢 IoT Smart Building RAG System")
    print("Starting application...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  No .env file found. Creating from template...")
        if os.path.exists('env_example.txt'):
            with open('env_example.txt', 'r') as f:
                content = f.read()
            with open('.env', 'w') as f:
                f.write(content)
            print("✅ Created .env file. Please edit it with your OpenAI API key.")
    
    # Start Streamlit app
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

"""
Script to run the Streamlit dashboard.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def download_spacy_model():
    """Download spaCy English model."""
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Warning: Could not download spaCy model: {e}")
        print("The application will work but with reduced preprocessing capabilities.")

def run_streamlit():
    """Run the Streamlit application."""
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Streamlit application stopped.")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

if __name__ == "__main__":
    print("🚀 Setting up Document Clustering & Topic Modeling Dashboard...")
    print("=" * 60)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        sys.exit(1)
    
    # Install requirements
    print("📦 Installing requirements...")
    if not install_requirements():
        sys.exit(1)
    
    # Download spaCy model
    print("🔤 Downloading spaCy model...")
    download_spacy_model()
    
    # Run Streamlit
    print("\n🌟 Starting Streamlit dashboard...")
    print("📱 The dashboard will open in your browser automatically.")
    print("🔗 If it doesn't open, go to: http://localhost:8501")
    print("=" * 60)
    
    run_streamlit()
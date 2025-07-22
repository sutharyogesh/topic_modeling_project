"""
Script to run the Flask web application.
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

def run_flask():
    """Run the Flask application."""
    try:
        os.environ['FLASK_APP'] = 'flask_app.py'
        os.environ['FLASK_ENV'] = 'development'
        subprocess.run([sys.executable, "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"])
    except KeyboardInterrupt:
        print("\n👋 Flask application stopped.")
    except Exception as e:
        print(f"❌ Error running Flask: {e}")

if __name__ == "__main__":
    print("🚀 Setting up Document Clustering & Topic Modeling Web App...")
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
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static/plots", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    # Run Flask
    print("\n🌟 Starting Flask web application...")
    print("🔗 Open your browser and go to: http://localhost:5000")
    print("=" * 60)
    
    run_flask()
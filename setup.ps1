# Remove existing venv if exists
if (Test-Path venv) {
    Remove-Item -Recurse -Force venv
}

# Create new venv with Python 3.12
C:\Users\Fyru\AppData\Local\Programs\Python\Python312\python.exe -m venv venv

# Activate the venv
. .\venv\Scripts\Activate.ps1

# Print Python version to confirm
python --version

# Install requirements
pip install -r requirements.txt

# Run the Streamlit app
python -m streamlit run main.py
@echo off

rem Remove existing venv if exists
if exist venv (
    rmdir /s /q venv
)

rem Create new venv with Python 3.12
C:\Users\Fyru\AppData\Local\Programs\Python\Python312\python.exe -m venv venv

rem Activate the venv
call venv\Scripts\activate.bat

rem Print Python version to confirm
python --version

rem Install requirements
pip install -r requirements.txt

rem Run the Streamlit app
python -m streamlit run main.py
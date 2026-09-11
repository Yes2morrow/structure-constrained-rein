@echo off
cd /d "%~dp0"
echo Starting Streamlit workbench at http://localhost:8501
echo Press Ctrl+C in this window to stop.
set "PROJECT_PYTHON=%~dp0.venv312\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" set "PROJECT_PYTHON=%~dp0..\.venv312\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
    echo Python environment not found. See README.md for setup.
    pause
    exit /b 1
)
"%PROJECT_PYTHON%" -m streamlit run main.py --server.headless true --browser.gatherUsageStats false --logger.level error
pause

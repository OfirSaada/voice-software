@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting transcription... (loading takes 30-60 seconds, please wait)
echo.
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv\Scripts\python.exe not found in %cd%
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 transcribe_v3.py
echo.
echo Finished. Press any key to close.
pause >nul

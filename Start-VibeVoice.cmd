@echo off
setlocal
title VibeVoice
cd /d "%~dp0"
echo Starting VibeVoice...
echo Loading the app. Your browser will open when it is ready.
echo The first start downloads the voice model, about 5.4 GB.
echo Keep this window open while using VibeVoice.
echo.
".venv\Scripts\python.exe" -u -X utf8 "demo\gradio_demo.py" --model_path vibevoice/VibeVoice-1.5B --port 7861 %*
if errorlevel 1 (
    echo.
    echo VibeVoice could not start. The error details are shown above.
    pause
    exit /b 1
)

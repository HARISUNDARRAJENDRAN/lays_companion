@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" -Execute -VoiceGpu iris
if errorlevel 1 pause

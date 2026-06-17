@echo off
REM Double-clickable wrapper for run.ps1 (bypasses the default .ps1 execution policy).
REM Pass-through args, e.g.:  run.bat -NoRefresh
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
pause

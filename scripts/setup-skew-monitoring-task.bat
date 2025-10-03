@echo off
REM Quick setup for nightly skew monitoring task
REM Creates a Windows Task Scheduler entry

echo ============================================================
echo Trading Platform - Skew Monitoring Task Setup
echo ============================================================
echo.

REM Get the project directory (parent of scripts folder)
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set PYTHON_SCRIPT=%PROJECT_DIR%\scripts\nightly_skew_monitor.py

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found in PATH
    echo Please install Python or add it to your PATH
    pause
    exit /b 1
)

echo Creating scheduled task...
echo.

REM Create the scheduled task
schtasks /Create /TN "TradingPlatform-SkewMonitoring" ^
    /TR "python \"%PYTHON_SCRIPT%\"" ^
    /SC DAILY ^
    /ST 02:00 ^
    /F ^
    /RL HIGHEST ^
    /RU "%USERNAME%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS: Scheduled task created!
    echo ============================================================
    echo.
    echo Task Name: TradingPlatform-SkewMonitoring
    echo Schedule: Daily at 2:00 AM
    echo Script: %PYTHON_SCRIPT%
    echo.
    echo To run manually:
    echo   schtasks /Run /TN "TradingPlatform-SkewMonitoring"
    echo.
    echo To view task:
    echo   schtasks /Query /TN "TradingPlatform-SkewMonitoring"
    echo.
    echo To delete task:
    echo   schtasks /Delete /TN "TradingPlatform-SkewMonitoring" /F
    echo ============================================================
) else (
    echo.
    echo ERROR: Failed to create scheduled task
    echo Please run this script as Administrator
)

echo.
pause

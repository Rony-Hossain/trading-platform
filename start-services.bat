@echo off
echo Starting Trading Platform Services...
echo =====================================

echo.
echo Checking Docker Desktop status...
docker ps >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Desktop is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo ✅ Docker Desktop is running
echo.

echo Starting all services with Docker Compose...
docker-compose up -d --build

echo.
echo Waiting for services to start (30 seconds)...
timeout /t 30 /nobreak >nul

echo.
echo Checking service status...
docker-compose ps

echo.
echo Services should be starting up. 
echo You can monitor logs with: docker-compose logs -f [service-name]
echo.
echo Testing services in 60 seconds...
timeout /t 60 /nobreak >nul

echo.
echo Running comprehensive service tests...
python test-all-services.py

pause
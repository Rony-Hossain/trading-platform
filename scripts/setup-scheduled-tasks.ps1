# Windows Task Scheduler Setup for Trading Platform
# Creates scheduled tasks for nightly monitoring jobs

param(
    [string]$PythonPath = "python",
    [string]$ProjectRoot = $PSScriptRoot + "\..",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

# Task definitions
$tasks = @(
    @{
        Name = "TradingPlatform-SkewMonitoring"
        Description = "Nightly skew monitoring between offline and online features"
        Script = "scripts\nightly_skew_monitor.py"
        Schedule = "Daily"
        StartTime = "02:00"
    }
)

function Remove-ScheduledTask {
    param($TaskName)

    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task) {
        Write-Host "Removing scheduled task: $TaskName"
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "✓ Task removed successfully"
    } else {
        Write-Host "Task $TaskName not found"
    }
}

function Create-ScheduledTask {
    param($TaskConfig)

    $taskName = $TaskConfig.Name
    $description = $TaskConfig.Description
    $scriptPath = Join-Path $ProjectRoot $TaskConfig.Script
    $startTime = $TaskConfig.StartTime

    Write-Host "`n============================================================"
    Write-Host "Creating scheduled task: $taskName"
    Write-Host "============================================================"

    # Remove existing task if present
    $existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($existingTask) {
        Write-Host "Removing existing task..."
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    }

    # Create action
    $action = New-ScheduledTaskAction `
        -Execute $PythonPath `
        -Argument "$scriptPath" `
        -WorkingDirectory $ProjectRoot

    # Create trigger (daily at specified time)
    $trigger = New-ScheduledTaskTrigger -Daily -At $startTime

    # Create settings
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 2)

    # Create principal (run whether user is logged on or not)
    $principal = New-ScheduledTaskPrincipal `
        -UserId $env:USERNAME `
        -LogonType S4U `
        -RunLevel Highest

    # Register the task
    Register-ScheduledTask `
        -TaskName $taskName `
        -Description $description `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Force | Out-Null

    Write-Host "✓ Task created successfully"
    Write-Host "  Schedule: Daily at $startTime"
    Write-Host "  Script: $scriptPath"
}

# Main execution
Write-Host "============================================================"
Write-Host "Trading Platform - Scheduled Tasks Setup"
Write-Host "============================================================"
Write-Host "Project Root: $ProjectRoot"
Write-Host "Python Path: $PythonPath"
Write-Host ""

if ($Remove) {
    Write-Host "Removing all scheduled tasks..."
    foreach ($task in $tasks) {
        Remove-ScheduledTask -TaskName $task.Name
    }
} else {
    # Verify Python is available
    try {
        $pythonVersion = & $PythonPath --version 2>&1
        Write-Host "Python version: $pythonVersion"
    } catch {
        Write-Host "ERROR: Python not found at '$PythonPath'" -ForegroundColor Red
        Write-Host "Please specify the correct Python path using -PythonPath parameter"
        exit 1
    }

    # Create all tasks
    foreach ($task in $tasks) {
        Create-ScheduledTask -TaskConfig $task
    }

    Write-Host "`n============================================================"
    Write-Host "SUMMARY"
    Write-Host "============================================================"
    Write-Host "$($tasks.Count) scheduled task(s) created successfully"
    Write-Host ""
    Write-Host "To view tasks:"
    Write-Host "  Get-ScheduledTask | Where-Object {`$_.TaskName -like 'TradingPlatform-*'}"
    Write-Host ""
    Write-Host "To run a task manually:"
    Write-Host "  Start-ScheduledTask -TaskName 'TradingPlatform-SkewMonitoring'"
    Write-Host ""
    Write-Host "To remove all tasks:"
    Write-Host "  .\scripts\setup-scheduled-tasks.ps1 -Remove"
    Write-Host "============================================================"
}

<#
.SYNOPSIS
    Capture the measure_km.ino serial log from Windows, for WSL2 users.

.DESCRIPTION
    WSL2 does not expose Windows USB serial ports, so /dev/ttyUSB0 does not
    exist there and identify_km.py --port cannot be used.  This script reads
    the board from the Windows side and writes the log straight into the WSL
    filesystem, so the analysis can run in WSL as usual.

    Opening the port asserts DTR, which resets an Arduino Uno.  The capture
    therefore starts from the sketch's own banner, and no manual reset is
    needed.

.PARAMETER Port
    Windows COM port of the board, e.g. COM10.  List them with:
        Get-CimInstance Win32_SerialPort | Select-Object DeviceID,Description

.PARAMETER Out
    Where to write the log.  A UNC path into WSL works, e.g.
        \\wsl.localhost\Ubuntu-22.04\home\me\dev\pg\data\km-measurement.csv

.EXAMPLE
    From WSL, letting wslpath build the Windows paths:

    powershell.exe -NoProfile -ExecutionPolicy Bypass `
        -File "$(wslpath -w scripts/motor-calibration/km-identification/capture_windows.ps1)" `
        -Port COM10 -Out "$(wslpath -w data/km-measurement.csv)"
#>

param(
    [string]$Port = "COM10",
    [int]$Baud = 115200,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$TimeoutSec = 180
)

$ErrorActionPreference = "Stop"

Write-Host "port    : $Port @ $Baud"
Write-Host "output  : $Out"

$serial = New-Object System.IO.Ports.SerialPort $Port, $Baud, 'None', 8, 'One'
$serial.ReadTimeout = 5000

# Asserting DTR resets the Uno, so the sketch restarts and we catch its
# "# BEGIN" marker instead of joining a run already in progress.
$serial.DtrEnable = $true

try {
    $serial.Open()
} catch {
    Write-Host ""
    Write-Host "could not open $Port : $($_.Exception.Message)"
    Write-Host "close the Arduino IDE serial monitor if it holds the port."
    exit 1
}

Write-Host "waiting for the board to reset and start ..."

$lines = New-Object System.Collections.Generic.List[string]
$deadline = (Get-Date).AddSeconds($TimeoutSec)
$sawBegin = $false
$samples = 0

try {
    while ((Get-Date) -lt $deadline) {
        try {
            $line = $serial.ReadLine()
        } catch [System.TimeoutException] {
            if ($sawBegin) {
                Write-Host "serial went quiet, stopping."
                break
            }
            continue
        }

        $line = $line.TrimEnd()
        $lines.Add($line) | Out-Null

        if ($line.StartsWith("#")) {
            Write-Host $line
        } else {
            $samples++

            if ($samples % 200 -eq 0) {
                Write-Host "  $samples samples ..."
            }
        }

        if ($line -match "# BEGIN") {
            $sawBegin = $true
        }

        if ($line -match "# END") {
            break
        }
    }
} finally {
    $serial.Close()
}

if (-not $sawBegin) {
    Write-Host ""
    Write-Host "never saw '# BEGIN'. Is measure_km.ino actually flashed?"
}

Set-Content -Path $Out -Value $lines -Encoding ascii

Write-Host ""
Write-Host "captured $samples samples, $($lines.Count) lines -> $Out"

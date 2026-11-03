$ErrorActionPreference = 'Stop'
$stateDir = Join-Path $PSScriptRoot 'data'
$stateFile = Join-Path $stateDir 'process.json'
if (!(Test-Path -LiteralPath $stateFile)) { Write-Output 'Companion is stopped.'; exit 0 }
$state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
$expected = Join-Path $PSScriptRoot 'app\lays_companion.py'
$proc = Get-CimInstance Win32_Process -Filter "ProcessId = $($state.processId)"
if (!$proc) { Remove-Item -LiteralPath $stateFile; Write-Output 'Companion already stopped.'; exit 0 }
if (!$proc.CommandLine.Contains($expected) -or $proc.CreationDate.ToUniversalTime().ToString('o') -ne $state.created) { throw 'PID ownership mismatch; refusing to stop this process.' }
Set-Content -LiteralPath (Join-Path $stateDir 'stop.request') -Value 'stop'
try { Wait-Process -Id $state.processId -Timeout 30 -ErrorAction Stop } catch { throw 'Graceful shutdown has not finished. Inspect logs; no unrelated process was stopped.' }
Remove-Item -LiteralPath $stateFile
Write-Output 'Companion stopped.'

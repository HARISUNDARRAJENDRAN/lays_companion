param([switch]$Execute, [ValidateSet('auto','nvidia','iris','cpu')][string]$VoiceGpu='nvidia', [switch]$NoVoice)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) { $python = Join-Path $PSScriptRoot 'runtime\python.exe' }
. (Join-Path $PSScriptRoot 'app\prepare-tcl.ps1')
Set-LaysTclEnvironment -Python $python
$script = Join-Path $PSScriptRoot 'app\lays_companion.py'
$stateDir = Join-Path $PSScriptRoot 'data'
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
$stateFile = Join-Path $stateDir 'process.json'
$signal = Join-Path $stateDir 'stop.request'
if (Test-Path -LiteralPath $stateFile) {
    $old = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
    $running = Get-CimInstance Win32_Process -Filter "ProcessId = $($old.processId)"
    if ($running -and $running.CommandLine.Contains($script) -and $running.CreationDate.ToUniversalTime().ToString('o') -eq $old.created) {
        throw "Companion already running: $($old.processId). Use stop.ps1 first."
    }
}
if (!(Test-Path -LiteralPath $python)) { throw "Python environment missing: $python" }
& $python -c 'import torch, tkinter, numpy, sounddevice, safetensors, transformers; assert torch.cuda.is_available(), "CUDA unavailable"'
if ($LASTEXITCODE -ne 0) { throw 'Dependency or CUDA preflight failed.' }
Remove-Item -LiteralPath $signal -ErrorAction SilentlyContinue
$mode = if ($Execute) { '--execute' } else { '--dry-run' }
$arguments = @('-u', ('"' + $script + '"'), $mode, '--voice-gpu', $VoiceGpu, '--stop-file', ('"' + $signal + '"'))
if ($NoVoice) { $arguments += '--no-voice' }
$proc = Start-Process -FilePath $python -ArgumentList $arguments -WindowStyle Hidden -WorkingDirectory $PSScriptRoot -RedirectStandardOutput (Join-Path $stateDir 'stdout.log') -RedirectStandardError (Join-Path $stateDir 'stderr.log') -PassThru
$owned = Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)"
if (!$owned) { throw 'Companion exited during launch; inspect stderr.log.' }
@{processId=$proc.Id; created=$owned.CreationDate.ToUniversalTime().ToString('o'); script=$script} | ConvertTo-Json | Set-Content -LiteralPath $stateFile
Write-Output "Started PID $($proc.Id) ($mode). Readiness and errors: $stateDir\stdout.log and stderr.log"

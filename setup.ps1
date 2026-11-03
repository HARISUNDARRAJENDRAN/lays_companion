param([switch]$SkipHexInstall)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location -LiteralPath $root

function Get-Python312 {
    $bundled = Join-Path $root 'runtime\python.exe'
    if (Test-Path -LiteralPath $bundled) { return $bundled }
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & py -3.12 --version *> $null
        if ($LASTEXITCODE -eq 0) { return (Get-Command py).Source }
    }
    $installedPython = Join-Path $root '.python312\python.exe'
    if (!(Test-Path -LiteralPath $installedPython)) {
        $installer = Join-Path $env:TEMP 'lays-python-3.12.10-amd64.exe'
        Invoke-WebRequest 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile $installer
        $signature = Get-AuthenticodeSignature -LiteralPath $installer
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
            throw 'Python installer signature check failed.'
        }
        $pythonDir = Join-Path $root '.python312'
        # Start-Process joins argument arrays into one command line; quote this
        # value so a checkout under a path containing spaces installs correctly.
        $args = @('/quiet','InstallAllUsers=0',('TargetDir="{0}"' -f $pythonDir),'Include_pip=1','Include_tcltk=1','Include_test=0','Include_launcher=0','PrependPath=0','Shortcuts=0')
        $proc = Start-Process -FilePath $installer -ArgumentList $args -Wait -PassThru
        if ($proc.ExitCode -ne 0) { throw "Python installer failed with exit code $($proc.ExitCode)." }
    }
    if (!(Test-Path -LiteralPath $installedPython)) { throw 'Python 3.12 installation did not create python.exe.' }
    return $installedPython
}

$pythonCommand = Get-Python312
$envDir = Join-Path $root '.venv'
if (!(Test-Path -LiteralPath (Join-Path $envDir 'Scripts\python.exe'))) {
    if ($pythonCommand -eq (Get-Command py -ErrorAction SilentlyContinue).Source) {
        & py -3.12 -m venv $envDir
    } else {
        & $pythonCommand -m venv $envDir
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
$python = Join-Path $envDir 'Scripts\python.exe'
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed.' }
Write-Output 'Installing pinned dependencies and CUDA 13.0 PyTorch wheels...'
& $python -m pip install --prefer-binary --extra-index-url 'https://download.pytorch.org/whl/cu130' -r (Join-Path $root 'requirements.lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }

Write-Output 'Checking private checkpoint access on Hugging Face...'
& $python -c "from huggingface_hub import HfApi; HfApi().model_info('Harisundar/lays-r5ft-ckpt', revision='30758a7cd5772ccb65de34ecb013166bb5007d6d')"
if ($LASTEXITCODE -ne 0) {
    Write-Output 'Sign in with an account that has access to Harisundar/lays-r5ft-ckpt.'
    & (Join-Path $envDir 'Scripts\hf.exe') auth login
    if ($LASTEXITCODE -ne 0) { throw 'Hugging Face login failed.' }
    & $python -c "from huggingface_hub import HfApi; HfApi().model_info('Harisundar/lays-r5ft-ckpt', revision='30758a7cd5772ccb65de34ecb013166bb5007d6d')"
    if ($LASTEXITCODE -ne 0) { throw 'The signed-in Hugging Face account does not have checkpoint access.' }
}
Write-Output 'Downloading model files. The round-2 checkpoint requires access to the private Hugging Face repository Harisundar/lays-r5ft-ckpt.'
& $python (Join-Path $root 'app/fetch_models.py')
if ($LASTEXITCODE -ne 0) {
    Write-Output 'If Hugging Face asks for access, run .venv\Scripts\hf.exe auth login, request access to Harisundar/lays-r5ft-ckpt, and rerun Setup.cmd.'
    throw 'Model download failed.'
}
& $python (Join-Path $root 'app/install_desktop_driver.py')
if ($LASTEXITCODE -ne 0) { throw 'Native desktop driver installation failed.' }

if (!$SkipHexInstall) {
    $hexRoot = Join-Path $env:APPDATA 'hex'
    $hexVersion = if (Test-Path (Join-Path $hexRoot 'current-version')) { (Get-Content (Join-Path $hexRoot 'current-version') -Raw).Trim() } else { $null }
    $needsHex = !$hexVersion -or ([version]$hexVersion -lt [version]'3.3.0') -or !(Test-Path (Join-Path $hexRoot "versions\$hexVersion\hex.exe"))
    if ($needsHex) {
        $installerScript = Join-Path $env:TEMP 'lays-install-hex.ps1'
        Invoke-WebRequest 'https://anoma.ly/hex/files/install-windows.ps1' -OutFile $installerScript
        $installerSha = (Get-FileHash -LiteralPath $installerScript -Algorithm SHA256).Hash
        if ($installerSha -ne '7D9AD5BA9E62CE9F65FE39298ADB8B9AA34AEB3D47D85AD381E5A0137D203266') { throw 'Hex installer script hash check failed.' }
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installerScript
        if ($LASTEXITCODE -ne 0) { throw 'Hex installation failed.' }
        $hexVersion = (Get-Content (Join-Path $hexRoot 'current-version') -Raw).Trim()
    }
    $hexExe = Join-Path $hexRoot "versions\$hexVersion\hex.exe"
    Start-Process -FilePath $hexExe -ArgumentList 'app'
    Write-Output 'Hex opened. Finish its microphone/model onboarding once, then start LAYS.'
}

Write-Output 'Checking dependencies and model assets...'
& $python (Join-Path $root 'app/doctor.py')
if ($LASTEXITCODE -ne 0) { throw 'Setup verification failed. See data/dependency-check.json.' }
Write-Output 'Setup complete. Start LAYS with Start.cmd; use Start Dry Run.cmd to log predictions without sending input.'

function Set-LaysTclEnvironment {
    param([string]$Python, [string[]]$PrefixArguments=@())

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        if ($PrefixArguments.Count) { $prefix = & $Python @PrefixArguments -c 'import sys; print(sys.base_prefix)' 2>$null }
        else { $prefix = & $Python -c 'import sys; print(sys.base_prefix)' 2>$null }
    } finally { $ErrorActionPreference = $previousPreference }
    if (!$prefix) { throw "Could not locate Python's Tcl/Tk runtime for $Python." }

    $source = Join-Path $prefix.Trim() 'tcl'
    $sourceInit = Join-Path $source 'tcl8.6\init.tcl'
    $sourceTk = Join-Path $source 'tk8.6\tk.tcl'
    if (!(Test-Path -LiteralPath $sourceInit) -or !(Test-Path -LiteralPath $sourceTk)) {
        throw "Python's Tcl/Tk files are missing under $source."
    }

    $destination = $source
    if ($source -match '(?i)\\OneDrive[^\\]*\\') {
        $destination = Join-Path $env:LOCALAPPDATA 'LaysCompanion\Tcl'
        $destInit = Join-Path $destination 'tcl8.6\init.tcl'
        $destTk = Join-Path $destination 'tk8.6\tk.tcl'
        $copy = !(Test-Path -LiteralPath $destInit) -or !(Test-Path -LiteralPath $destTk)
        if (!$copy) {
            $copy = (Get-FileHash -LiteralPath $sourceInit -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $destInit -Algorithm SHA256).Hash
            if (!$copy) { $copy = (Get-FileHash -LiteralPath $sourceTk -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $destTk -Algorithm SHA256).Hash }
        }
        if ($copy) {
            New-Item -ItemType Directory -Force -Path $destination | Out-Null
            Copy-Item -Path (Join-Path $source '*') -Destination $destination -Recurse -Force
        }
    }

    $env:TCL_LIBRARY = Join-Path $destination 'tcl8.6'
    $env:TK_LIBRARY = Join-Path $destination 'tk8.6'
}

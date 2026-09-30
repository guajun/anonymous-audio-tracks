$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
uv run --project . python analyze.py @args
exit $LASTEXITCODE

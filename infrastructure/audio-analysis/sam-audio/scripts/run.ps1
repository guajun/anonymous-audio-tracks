$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
uv run --project . python scripts/run_inference.py @args
exit $LASTEXITCODE

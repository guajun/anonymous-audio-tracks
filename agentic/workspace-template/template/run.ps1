# run.ps1 - start the Pi audio-analysis workspace (issue #31, Windows).
#
# Usage (PowerShell, from the workspace directory):
#   .\run.ps1                                  # interactive Pi
#   .\run.ps1 "analyze audio/inputs/fixture-a.wav"
#
# What it does:
#   * cd to the workspace root (Pi discovers project config from cwd)
#   * set PI_AUDIO_BRIDGE_ROOT=audio\inputs (controlled root for audio_attach)
#   * read SAM_AUDIO_ROOT from local/config.json when configured
#   * start `pi --approve`: PER-PROCESS project trust (loads this project's
#     .pi settings/extensions). It never writes ~/.pi/agent/trust.json and
#     never changes global settings or authentication.
#
# NOTE ON ENCODING: keep this file ASCII-only and save it as UTF-8 with BOM.
# Windows PowerShell 5.1 reads a BOM-less .ps1 as the system ANSI codepage;
# non-ASCII comment bytes can then swallow the following line (a real bug we
# hit: the `pi` line was silently commented out).
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$env:PI_AUDIO_BRIDGE_ROOT = (Join-Path $PSScriptRoot "audio\inputs")

$cfgPath = Join-Path $PSScriptRoot "local\config.json"
if (Test-Path $cfgPath) {
    $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
    if ($cfg.sam_root -and -not $cfg.sam_root.StartsWith("<")) {
        $env:SAM_AUDIO_ROOT = $cfg.sam_root
    }
}

# The model is pinned by .pi/settings.json to google/gemini-3.8-flash
# (issue #30 frozen research model). Do not override it with --model.
pi --approve @args

# Keep a separate final statement: when `pi ...` is the last statement of a
# -File script, PowerShell drops the child's stdout. This also propagates the
# real exit code.
exit $LASTEXITCODE

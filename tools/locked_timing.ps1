# Lock the clocks, run one benchmark script for each config, then restore the clocks.
# Example (elevated PowerShell, repository root):
#   tools\locked_timing.ps1 -Script bench_gpu.py -Configs configs\gpu_neu_b1.json,configs\gpu_neu_b16.json
param([Parameter(Mandatory)][string]$Configs, [string]$Script = 'bench_gpu.py', [string]$Python = 'python',
      [int]$GpuMHz = 1200)
$repo = Split-Path $PSScriptRoot -Parent
& (Join-Path $PSScriptRoot 'clock_ctl.ps1') -Action lock -GpuMHz $GpuMHz
Start-Sleep 8
try {
    foreach ($c in ($Configs -split ',')) {
        $logFile = [IO.Path]::ChangeExtension($c, '.log')
        & $Python -u (Join-Path $repo "scripts\$Script") --config $c *> $logFile
        "$c exit $LASTEXITCODE (log: $logFile)"
    }
} finally {
    & (Join-Path $PSScriptRoot 'clock_ctl.ps1') -Action unlock
}

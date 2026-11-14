# Lock or restore GPU and CPU clocks for timing runs (Windows, elevated shell).
#   lock:   GPU SM clock fixed with nvidia-smi -lgc; CPU turbo off and minimum = maximum = 100 %
#           of the base clock (powercfg). The previous CPU power settings are saved.
#   unlock: nvidia-smi -rgc and the saved CPU power settings are restored.
# Without an elevated shell, register this script as two scheduled tasks that run with highest
# privileges (one with -Action lock, one with -Action unlock) and start them with schtasks /run.
param([ValidateSet('lock','unlock')][string]$Action = 'lock', [int]$GpuMHz = 1200)
$dir = Join-Path $PSScriptRoot 'clock'
New-Item -ItemType Directory -Force $dir | Out-Null
$log = Join-Path $dir 'clock_ctl.log'
$saved = Join-Path $dir 'cpu_power_saved.json'
function Log($m) { "$(Get-Date -Format o) $m" | Out-File -Append -Encoding utf8 $log }
function Get-ProcValue($alias) {
    # The last two hexadecimal values are the current AC and DC indices (locale independent).
    $hex = @(powercfg /query SCHEME_CURRENT SUB_PROCESSOR $alias | Select-String ':\s*0x([0-9a-fA-F]+)\s*$' |
             ForEach-Object { [Convert]::ToInt32($_.Matches[0].Groups[1].Value, 16) })
    if ($hex.Count -lt 2) { throw "cannot read $alias" }
    @{ ac = $hex[-2]; dc = $hex[-1] }
}
$aliases = 'PERFBOOSTMODE', 'PROCTHROTTLEMIN', 'PROCTHROTTLEMAX'
if ($Action -eq 'lock') {
    powercfg -attributes SUB_PROCESSOR PERFBOOSTMODE -ATTRIB_HIDE
    if (-not (Test-Path $saved)) {
        $orig = @{}; foreach ($a in $aliases) { $orig[$a] = Get-ProcValue $a }
        $orig | ConvertTo-Json | Out-File -Encoding utf8 $saved
    }
    powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE 0
    powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 100
    powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100
    powercfg /setactive SCHEME_CURRENT
    $g = (nvidia-smi -lgc "$GpuMHz,$GpuMHz" 2>&1) -join ' '
    Log "lock gpu=$GpuMHz result: $g"
} else {
    $g = (nvidia-smi -rgc 2>&1) -join ' '
    if (Test-Path $saved) {
        $orig = Get-Content $saved -Raw | ConvertFrom-Json
        foreach ($a in $aliases) {
            powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR $a $orig.$a.ac
            powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR $a $orig.$a.dc
        }
        powercfg /setactive SCHEME_CURRENT
        Remove-Item $saved
    }
    Log "unlock result: $g"
}
Get-Content $log -Tail 1

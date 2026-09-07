param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$Campaign,
    [Parameter(Mandatory=$true)][string]$LocalPaths,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [ValidateSet('preflight-only','validate-plan','execute-authorized','package-only')]
    [string]$Mode='execute-authorized',
    [ValidateSet('ALL','S72','S73','S74','S75','S76')][string]$Stage='ALL',
    [switch]$Resume,
    [switch]$Capacity
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
foreach ($inputFile in @($Python,$Campaign,$LocalPaths)) {
    if (!(Test-Path -LiteralPath $inputFile -PathType Leaf)) { throw "Missing input: $inputFile" }
}
$projectRoot = Split-Path -Parent $PSScriptRoot
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputRoot)
$campaignDirectory = if ($Capacity) { 'stage70_76_capacity' } else { 'stage70_76_architecture' }
$expectedOutput = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "artifacts/$campaignDirectory"))
if ($resolvedOutput -ne $expectedOutput) { throw 'OutputRoot must be this worktree campaign directory' }
$logRoot = Join-Path (Split-Path -Parent $resolvedOutput) ($campaignDirectory + '_launcher_logs')
[System.IO.Directory]::CreateDirectory($logRoot) | Out-Null
# Keep invocation logs outside the signed scientific package and serialize this launcher.
$guard = [System.IO.File]::Open((Join-Path $logRoot 'launcher.lock'),
    [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
$runId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')
$logPath = Join-Path $logRoot ($runId + '.log')
$statePath = Join-Path $logRoot ($runId + '.json')
$exitCode = 1
$record = [ordered]@{ started_utc=[DateTime]::UtcNow.ToString('o'); mode=$Mode;
    capacity=[bool]$Capacity; output_root=$resolvedOutput; status='STARTING'; log=$logPath }
try {
    $runnerArgs=@('-B',(Join-Path $PSScriptRoot 'run_scientific_recovery_v10_stage70_76.py'),
        '--campaign',$Campaign,'--local-paths',$LocalPaths,'--output-root',$resolvedOutput,
        "--$Mode",'--stage',$Stage)
    if ($Capacity) { $runnerArgs += '--capacity' }
    $resumeExisting = $Mode -eq 'execute-authorized' -and
        (Test-Path -LiteralPath (Join-Path $resolvedOutput 'TRAINING_LOCK.json') -PathType Leaf)
    if ($Resume -or $resumeExisting) { $runnerArgs += '--resume' }
    $record['resume'] = [bool]($Resume -or $resumeExisting)
    $record['arguments'] = $runnerArgs
    $record['status'] = 'RUNNING'
    $record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statePath -Encoding UTF8
    # Native stderr warnings must not turn a successful Python run into a PowerShell failure.
    $ErrorActionPreference = 'Continue'
    & $Python @runnerArgs 2>&1 | Tee-Object -FilePath $logPath -ErrorAction Stop
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    $record['status'] = if ($exitCode -eq 0) { 'COMMAND_FINISHED' } else { 'INTERRUPTED' }
} catch {
    $record['status'] = 'INTERRUPTED'
    $record['launcher_error'] = $_.Exception.Message
    throw
} finally {
    $ErrorActionPreference = 'Stop'
    $record['exit_code'] = $exitCode
    $record['ended_utc'] = [DateTime]::UtcNow.ToString('o')
    try { $record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statePath -Encoding UTF8 }
    finally { $guard.Dispose() }
}
# A successful command is not a claim that scientific endpoints or the campaign are complete.
exit $exitCode

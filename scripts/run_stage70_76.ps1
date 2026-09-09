<#
.SYNOPSIS
Run the authorized Stage70-76 graph with durable invocation logs and exact resume.
.DESCRIPTION
The Python runner enforces scientific freeze, source roles, budget and stage gates,
then performs analysis and packaging. An existing training lock enables resume.
CountUsability optionally runs the existing fixed-cohort count reconstruction first;
a preparation failure stops the chain and preserves its exit code and log.
ComparatorInputs optionally verifies the completed native/DINO input bindings and
compiles the existing comparator recipes before training. Native/DINO generation
is not yet connected by this launcher.
.PARAMETER CountUsability
The fixed signed ROW_USABILITY.json for v11 count reconstruction or recovery.
Omit it when using the existing complete count cache.
.PARAMETER CountUsabilitySha256
Exact physical SHA256 of CountUsability. Required together with CountUsability.
.PARAMETER ComparatorInputs
Signed CAPACITY_COMPARATOR_INPUT_BINDINGS for the completed native/DINO caches.
.PARAMETER ComparatorInputsSha256
Exact physical SHA256 of ComparatorInputs. Required together with ComparatorInputs.
#>
param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$Campaign,
    [Parameter(Mandatory=$true)][string]$LocalPaths,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [ValidateSet('preflight-only','validate-plan','execute-authorized','package-only')]
    [string]$Mode='execute-authorized',
    [ValidateSet('ALL','S72','S73','S74','S75','S76')][string]$Stage='ALL',
    [switch]$Resume,
    [switch]$Capacity,
    [string]$CountUsability,
    [string]$CountUsabilitySha256,
    [string]$ComparatorInputs,
    [string]$ComparatorInputsSha256
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if ([bool]$CountUsability -ne [bool]$CountUsabilitySha256) {
    throw 'Count reconstruction requires CountUsability and CountUsabilitySha256 together'
}
if ($CountUsability) {
    if (!$Capacity -or $Mode -ne 'execute-authorized' -or $Stage -ne 'ALL') {
        throw 'Count reconstruction requires Capacity, execute-authorized and Stage ALL'
    }
    if ($CountUsabilitySha256 -notmatch '^[a-fA-F0-9]{64}$') {
        throw 'CountUsabilitySha256 must be a physical SHA256'
    }
    $CountUsability = (Resolve-Path -LiteralPath $CountUsability).ProviderPath
    if ((Get-FileHash -LiteralPath $CountUsability -Algorithm SHA256).Hash -ne $CountUsabilitySha256) {
        throw 'Count usability physical SHA256 changed'
    }
}
if ([bool]$ComparatorInputs -ne [bool]$ComparatorInputsSha256) {
    throw 'Comparator compilation requires ComparatorInputs and ComparatorInputsSha256 together'
}
if ($ComparatorInputs) {
    if (!$Capacity -or $Mode -ne 'execute-authorized' -or $Stage -ne 'ALL') {
        throw 'Comparator compilation requires Capacity, execute-authorized and Stage ALL'
    }
    if ($ComparatorInputsSha256 -notmatch '^[a-fA-F0-9]{64}$') {
        throw 'ComparatorInputsSha256 must be a physical SHA256'
    }
    $ComparatorInputs = (Resolve-Path -LiteralPath $ComparatorInputs).ProviderPath
    if ((Get-FileHash -LiteralPath $ComparatorInputs -Algorithm SHA256).Hash -ne $ComparatorInputsSha256) {
        throw 'Comparator inputs physical SHA256 changed'
    }
}
foreach ($inputFile in @($Python,$Campaign,$LocalPaths)) {
    if (!(Test-Path -LiteralPath $inputFile -PathType Leaf)) { throw "Missing input: $inputFile" }
}
$Python = (Resolve-Path -LiteralPath $Python).ProviderPath
$Campaign = (Resolve-Path -LiteralPath $Campaign).ProviderPath
$LocalPaths = (Resolve-Path -LiteralPath $LocalPaths).ProviderPath
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
    capacity=[bool]$Capacity; output_root=$resolvedOutput; status='STARTING'; log=$logPath;
    preparation_steps=@() }
try {
    $preparation = @()
    if ($CountUsability) {
        $preparation += [ordered]@{ name='capacity_counts'; arguments=@(
            '-B',(Join-Path $PSScriptRoot 'prepare_stage70_capacity.py'),
            '--count-usability',$CountUsability,
            '--count-usability-sha256',$CountUsabilitySha256.ToLowerInvariant(),
            '--output-root',(Join-Path $resolvedOutput 'representation_30000')) }
    }
    if ($ComparatorInputs) {
        $preparation += [ordered]@{ name='capacity_comparators'; arguments=@(
            '-B',(Join-Path $PSScriptRoot 'prepare_stage70_capacity.py'),
            '--comparator-inputs',$ComparatorInputs,
            '--output-root',(Join-Path $resolvedOutput 'compiled_comparators_30000')) }
    }
    foreach ($operation in $preparation) {
        $prepareArgs = $operation['arguments']
        $step = [ordered]@{ name=$operation['name']; arguments=$prepareArgs;
            status='RUNNING'; started_utc=[DateTime]::UtcNow.ToString('o') }
        $record['preparation_steps'] += $step
        $record['status'] = 'PREPARING_INPUTS'
        $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
        $ErrorActionPreference = 'Continue'
        Push-Location -LiteralPath $projectRoot
        try {
            & $Python @prepareArgs 2>&1 | Tee-Object -FilePath $logPath -Append -ErrorAction Stop
            $exitCode = $LASTEXITCODE
        } finally { Pop-Location }
        $ErrorActionPreference = 'Stop'
        $step['exit_code'] = $exitCode
        $step['ended_utc'] = [DateTime]::UtcNow.ToString('o')
        $step['status'] = if ($exitCode -eq 0) { 'COMMAND_FINISHED' } else { 'INTERRUPTED' }
        if ($exitCode -ne 0) {
            $record['status'] = 'INTERRUPTED'
            exit $exitCode
        }
        # Subsequent launcher errors must not inherit the successful preparation exit code.
        $exitCode = 1
    }
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
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
    # Native stderr warnings must not turn a successful Python run into a PowerShell failure.
    $ErrorActionPreference = 'Continue'
    Push-Location -LiteralPath $projectRoot
    try {
        & $Python @runnerArgs 2>&1 | Tee-Object -FilePath $logPath -Append -ErrorAction Stop
        $exitCode = $LASTEXITCODE
    } finally { Pop-Location }
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
    try { $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8 }
    finally { $guard.Dispose() }
}
# A successful command is not a claim that scientific endpoints or the campaign are complete.
exit $exitCode

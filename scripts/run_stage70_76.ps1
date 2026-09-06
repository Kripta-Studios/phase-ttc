param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$Campaign,
    [Parameter(Mandatory=$true)][string]$LocalPaths,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [ValidateSet('preflight-only','validate-plan','execute-authorized','package-only')]
    [string]$Mode='execute-authorized',
    [ValidateSet('ALL','S72','S73','S74','S75','S76')][string]$Stage='ALL',
    [switch]$Resume
)
$ErrorActionPreference='Stop'
foreach ($inputFile in @($Python,$Campaign,$LocalPaths)) {
    if (!(Test-Path -LiteralPath $inputFile -PathType Leaf)) { throw "Missing input: $inputFile" }
}
$runnerArgs=@('-B',(Join-Path $PSScriptRoot 'run_scientific_recovery_v10_stage70_76.py'),
    '--campaign',$Campaign,'--local-paths',$LocalPaths,'--output-root',$OutputRoot,
    "--$Mode",'--stage',$Stage)
if ($Resume) { $runnerArgs += '--resume' }
# The Python owner closes telemetry before packaging. No live transcript is bundled.
& $Python @runnerArgs
exit $LASTEXITCODE

param(
    [Parameter(Mandatory=$true)][ValidateSet('data1','data2')][string]$Task,
    [string]$Device='cpu',
    [string]$RunName='baseline_v1',
    [string]$Python=(Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
    [int]$Epochs=100,
    [int]$Batch=8,
    [string]$Source=''
)
$ErrorActionPreference='Stop'
if (!(Test-Path -LiteralPath $Python) -and !$PSBoundParameters.ContainsKey('Python')) {
    $Python=Join-Path $PSScriptRoot '.tools\python311\python.exe'
}
if (!$PSBoundParameters.ContainsKey('Batch')) {
    $Batch=if($Device -eq 'cpu'){1}else{4}
}
if (!(Test-Path -LiteralPath $Python)) { throw "Python environment not found: $Python. Follow docs/pipeline_guide.md." }
if ($RunName -notmatch '^[A-Za-z0-9_-]+$') { throw 'RunName must contain only letters, digits, underscore or hyphen.' }
$runRoot=Join-Path $PSScriptRoot "artifacts\$Task\$RunName"
if (Test-Path -LiteralPath $runRoot) { throw "Run already exists. Use a new -RunName: $runRoot" }
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$scripts=Join-Path $PSScriptRoot "scripts\$Task"
function Invoke-Step([string]$Script, [string[]]$Arguments) {
    & $Python (Join-Path $scripts "$Script.py") @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Script (exit $LASTEXITCODE)" }
}
Invoke-Step 'pre_eda' @('--output', (Join-Path $runRoot 'pre_eda'))
Invoke-Step 'train' @('--device',$Device,'--epochs',"$Epochs",'--batch',"$Batch",'--output',(Join-Path $runRoot 'train'))
$weights=Join-Path $runRoot 'train\fit\best.pth'
Invoke-Step 'evaluate' @('--weights',$weights,'--device',$Device,'--batch',"$Batch",'--split','val','--select-threshold','--output',(Join-Path $runRoot 'val'))
Invoke-Step 'post_eda' @('--evaluation',(Join-Path $runRoot 'val'),'--training',(Join-Path $runRoot 'train\fit'),'--output',(Join-Path $runRoot 'post_val'))
$threshold=Join-Path $runRoot 'val\threshold.json'
$config=Get-Content -LiteralPath $threshold -Raw | ConvertFrom-Json
if (!$config.constraints_met) { throw 'Validation recall target was not met. Review post_val and improve using validation before opening the final test.' }
Invoke-Step 'evaluate' @('--weights',$weights,'--device',$Device,'--batch',"$Batch",'--split','test','--threshold-config',$threshold,'--output',(Join-Path $runRoot 'test'))
Invoke-Step 'post_eda' @('--evaluation',(Join-Path $runRoot 'test'),'--training',(Join-Path $runRoot 'train\fit'),'--output',(Join-Path $runRoot 'post_test'))
if ($Source) {
    Invoke-Step 'infer' @('--weights',$weights,'--device',$Device,'--source',$Source,'--threshold-config',$threshold,'--output',(Join-Path $runRoot 'inference'))
}
Write-Output "Completed: $runRoot"

param(
    [Parameter(Mandatory=$true)][ValidateSet('data1','data2')][string]$Task,
    [ValidateSet('nano','small','medium','large')][string]$Model='nano',
    [int]$ImgSize=384,
    [string]$Device='cpu',
    [string]$RunName='rfdetr_nano_v1',
    [string]$Python='',
    [int]$Epochs=10,
    [int]$Batch=4,
    [int]$GradAccum=4,
    [int]$Patience=3,
    [string]$Source=''
)
$ErrorActionPreference='Stop'
# $PSScriptRoot is empty inside param defaults on Windows PowerShell 5.1, so resolve here.
if (!$Python) { $Python=Join-Path $PSScriptRoot '.tools\python311\python.exe' }
if (!(Test-Path -LiteralPath $Python)) { throw "Python environment not found: $Python. Follow docs/pipeline_guide.md." }
if ($RunName -notmatch '^[A-Za-z0-9_-]+$') { throw 'RunName must contain only letters, digits, underscore or hyphen.' }
$runRoot=Join-Path $PSScriptRoot "artifacts\$Task\$RunName"
if (Test-Path -LiteralPath $runRoot) { throw "Run already exists. Use a new -RunName: $runRoot" }
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$env:PYTHONIOENCODING='utf-8'
Push-Location $PSScriptRoot
function Invoke-Step([string]$Module, [string[]]$Arguments) {
    Write-Output "[$(Get-Date -Format s)] $Module $($Arguments -join ' ')"
    & $Python -m $Module --task $Task @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Module $($Arguments[1]) (exit $LASTEXITCODE)" }
}
try {
    $common=@('--model',$Model,'--imgsz',"$ImgSize",'--device',$Device,'--batch',"$Batch")
    Invoke-Step 'pipeline.run' @('--action','pre_eda','--output',(Join-Path $runRoot 'pre_eda'))
    Invoke-Step 'pipeline.rfdetr_run' (@('--action','train','--epochs',"$Epochs",'--grad-accum',"$GradAccum",'--patience',"$Patience",'--output',(Join-Path $runRoot 'train')) + $common)
    $weights=Join-Path $runRoot 'train\fit\checkpoint_best_total.pth'
    Invoke-Step 'pipeline.rfdetr_run' (@('--action','evaluate','--weights',$weights,'--split','val','--select-threshold','--benchmark','--output',(Join-Path $runRoot 'val')) + $common)
    Invoke-Step 'pipeline.rfdetr_run' @('--action','post_eda','--evaluation',(Join-Path $runRoot 'val'),'--training',(Join-Path $runRoot 'train\fit'),'--output',(Join-Path $runRoot 'post_val'))
    $threshold=Join-Path $runRoot 'val\threshold.json'
    $config=Get-Content -LiteralPath $threshold -Raw | ConvertFrom-Json
    if (!$config.constraints_met) { throw 'Validation recall target was not met. Review post_val and improve using validation before opening the final test.' }
    Invoke-Step 'pipeline.rfdetr_run' (@('--action','evaluate','--weights',$weights,'--split','test','--threshold-config',$threshold,'--output',(Join-Path $runRoot 'test')) + $common)
    Invoke-Step 'pipeline.rfdetr_run' @('--action','post_eda','--evaluation',(Join-Path $runRoot 'test'),'--training',(Join-Path $runRoot 'train\fit'),'--output',(Join-Path $runRoot 'post_test'))
    if ($Source) {
        Invoke-Step 'pipeline.rfdetr_run' (@('--action','infer','--weights',$weights,'--source',$Source,'--threshold-config',$threshold,'--output',(Join-Path $runRoot 'inference')) + $common)
    }
    Write-Output "Completed: $runRoot"
} finally {
    Pop-Location
}

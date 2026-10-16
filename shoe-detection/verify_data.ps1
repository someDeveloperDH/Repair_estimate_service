param([string]$DatasetDir = (Join-Path $PSScriptRoot 'datasets'))
$ErrorActionPreference='Stop'
$culture=[Globalization.CultureInfo]::InvariantCulture
$dateSplits=@{}
$report=@()
foreach($task in @('data1','data2')) {
    $root=Join-Path $DatasetDir $task
    $manifest=@(Import-Csv (Join-Path $root 'manifest.csv'))
    $classCount=if($task -eq 'data1'){7}else{3}
    $boxes=0
    foreach($split in @('train','val','test')) {
        $images=@(Get-ChildItem (Join-Path $root "images\$split") -File)
        $labels=@(Get-ChildItem (Join-Path $root "labels\$split") -File)
        if($images.Count -eq 0 -or $images.Count -ne $labels.Count) {throw "$task/$split count mismatch or empty split"}
        foreach($image in $images) {
            if($image.Name -match '^(\d{8})') {
                $date=$Matches[1]
                if($dateSplits.ContainsKey($date) -and $dateSplits[$date] -ne $split){throw "Date leakage: $date"}
                $dateSplits[$date]=$split
            }
            $label=Join-Path $root "labels\$split\$([IO.Path]::ChangeExtension($image.Name,'.txt'))"
            foreach($line in [IO.File]::ReadAllLines($label)) {
                $parts=$line.Split(' ')
                if($parts.Count -ne 5){throw "Invalid YOLO line: $label"}
                $cls=[int]$parts[0]
                if($cls -lt 0 -or $cls -ge $classCount){throw "Invalid class: $label"}
                $v=@($parts[1..4] | ForEach-Object {[double]::Parse($_,$culture)})
                foreach($n in $v){if([double]::IsNaN($n) -or [double]::IsInfinity($n) -or $n -lt 0 -or $n -gt 1){throw "Invalid coordinate: $label"}}
                if($v[2] -le 0 -or $v[3] -le 0){throw "Empty box: $label"}
                if(($v[0]-$v[2]/2) -lt -0.000001 -or ($v[0]+$v[2]/2) -gt 1.000001 -or ($v[1]-$v[3]/2) -lt -0.000001 -or ($v[1]+$v[3]/2) -gt 1.000001){throw "Out of bounds: $label"}
                $boxes++
            }
        }
    }
    $summary=Get-Content (Join-Path $root 'summary.json') -Raw | ConvertFrom-Json
    $expected=($summary.exported_boxes_by_class.PSObject.Properties.Value | Measure-Object -Sum).Sum
    if($boxes -ne $expected -or $manifest.Count -ne $summary.exported_images){throw "Summary mismatch: $task"}
    $report += [pscustomobject]@{task=$task;images=$manifest.Count;boxes=$boxes;status='PASS'}
}
$report | ConvertTo-Json | Set-Content (Join-Path $DatasetDir 'verification.json') -Encoding UTF8
$report | Format-Table

param([string]$OutputDir = (Join-Path $PSScriptRoot 'datasets'))
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.Drawing
if (Test-Path -LiteralPath $OutputDir) { throw "Output exists. Choose a new -OutputDir to avoid overwriting: $OutputDir" }
New-Item -ItemType Directory -Path $OutputDir | Out-Null
$utf8 = [Text.UTF8Encoding]::new($false)
$culture = [Globalization.CultureInfo]::InvariantCulture
function Write-Utf8($path, $content) { [IO.File]::WriteAllText($path, [string]$content, $utf8) }
function Get-Split($name) {
    # All photos from one shooting date share the split, across both tasks.
    $group = if ($name -match '^(\d{8})') { $Matches[1] } else { [IO.Path]::GetFileNameWithoutExtension($name) }
    $sha = [Security.Cryptography.SHA256]::Create()
    $hash = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes("shoe-v1:$group")); $sha.Dispose()
    $bucket = [BitConverter]::ToUInt32($hash, 0) % 100
    if ($bucket -lt 80) { return 'train' }; if ($bucket -lt 90) { return 'val' }; return 'test'
}
$pic = [IO.Compression.ZipFile]::OpenRead((Join-Path $PSScriptRoot 'raw\pic.zip'))
try {
    $index = @{}
    foreach ($entry in $pic.Entries) {
        if (!$entry.Name) { continue }
        if ($index.ContainsKey($entry.Name)) { throw "Duplicate image basename: $($entry.Name)" }
        $index[$entry.Name] = $entry
    }
    $summaries = @()
    foreach ($task in @(
        @{ Name='data1'; Zip='raw\data1\shoe_20260924_1030.zip' },
        @{ Name='data2'; Zip='raw\data2\label2.zip' }
    )) {
        $zip = [IO.Compression.ZipFile]::OpenRead((Join-Path $PSScriptRoot $task.Zip))
        try {
            $jsonEntry = $zip.Entries | Where-Object FullName -eq 'annotations/instances_default.json'
            $reader = [IO.StreamReader]::new($jsonEntry.Open())
            try { $data = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
        } finally { $zip.Dispose() }
        $root = Join-Path $OutputDir $task.Name
        foreach ($split in @('train','val','test')) {
            New-Item -ItemType Directory -Path (Join-Path $root "images\$split"),(Join-Path $root "labels\$split") -Force | Out-Null
        }
        $categories = @($data.categories | Sort-Object id)
        $classMap = @{}; $classCounts = @{}
        for ($i=0; $i -lt $categories.Count; $i++) { $classMap[[string]$categories[$i].id]=$i; $classCounts[$categories[$i].name]=0 }
        $byImage = @{}
        foreach ($ann in $data.annotations) {
            $key = [string]$ann.image_id
            if (!$byImage.ContainsKey($key)) { $byImage[$key]=[Collections.Generic.List[object]]::new() }
            $byImage[$key].Add($ann)
        }
        $manifest = [Collections.Generic.List[object]]::new()
        $missing = [Collections.Generic.List[object]]::new()
        $invalid = [Collections.Generic.List[object]]::new()
        $splitCounts = @{train=0;val=0;test=0}
        $unlabeled=0; $aspectMismatch=0; $clipped=0
        foreach ($im in $data.images) {
            $name = [IO.Path]::GetFileName($im.file_name)
            $hasLabels = $byImage.ContainsKey([string]$im.id)
            if (!$index.ContainsKey($name)) { $missing.Add([pscustomobject]@{image_id=$im.id;file_name=$name;has_labels=$hasLabels}); continue }
            if (!$hasLabels) { $unlabeled++; continue }
            $stream=$index[$name].Open()
            try {
                $bitmap=[Drawing.Image]::FromStream($stream)
                try { $w=$bitmap.Width; $h=$bitmap.Height } finally { $bitmap.Dispose() }
            } finally { $stream.Dispose() }
            # Uniform resizing is supported. Cropped or stretched images require review.
            if ([Math]::Abs(($w / $h) / ($im.width / $im.height) - 1) -gt 0.01) {
                $aspectMismatch++; $invalid.Add([pscustomobject]@{image_id=$im.id;annotation_id='';reason='aspect_ratio_mismatch'}); continue
            }
            $lines=[Collections.Generic.List[string]]::new()
            foreach ($ann in $byImage[[string]$im.id]) {
                if (!$classMap.ContainsKey([string]$ann.category_id) -or $ann.bbox.Count -ne 4 -or $im.width -le 0 -or $im.height -le 0) { throw "Invalid annotation schema: $($ann.id)" }
                $b=$ann.bbox
                $x1=[Math]::Max(0,[double]$b[0]); $y1=[Math]::Max(0,[double]$b[1])
                $x2=[Math]::Min($im.width,[double]$b[0]+[double]$b[2]); $y2=[Math]::Min($im.height,[double]$b[1]+[double]$b[3])
                if ($x2 -le $x1 -or $y2 -le $y1) { $invalid.Add([pscustomobject]@{image_id=$im.id;annotation_id=$ann.id;reason='empty_bbox'}); continue }
                if ([Math]::Abs($x1-$b[0]) -gt 0.000001 -or [Math]::Abs($y1-$b[1]) -gt 0.000001 -or [Math]::Abs($x2-($b[0]+$b[2])) -gt 0.000001 -or [Math]::Abs($y2-($b[1]+$b[3])) -gt 0.000001) { $clipped++ }
                $values=@((($x1+$x2)/2/$im.width), (($y1+$y2)/2/$im.height), (($x2-$x1)/$im.width), (($y2-$y1)/$im.height))
                $numbers=@($values | ForEach-Object { $_.ToString('F8',$culture) })
                $class=$classMap[[string]$ann.category_id]
                $lines.Add("$class $($numbers -join ' ')")
                $classCounts[$categories[$class].name]++
            }
            if ($lines.Count -eq 0) { continue }
            $split=Get-Split $name
            $destination=Join-Path $root "images\$split\$name"
            [IO.Compression.ZipFileExtensions]::ExtractToFile($index[$name],$destination)
            $labelName=[IO.Path]::ChangeExtension($name,'.txt')
            Write-Utf8 (Join-Path $root "labels\$split\$labelName") (($lines -join "`n")+"`n")
            $splitCounts[$split]++
            $manifest.Add([pscustomobject]@{file_name=$name;image_id=$im.id;split=$split;boxes=$lines.Count;actual_width=$w;actual_height=$h;label_width=$im.width;label_height=$im.height})
        }
        $yaml=@("path: '$($root.Replace('\','/'))'",'train: images/train','val: images/val','test: images/test','names:')
        for($i=0;$i -lt $categories.Count;$i++) { $yaml += "  ${i}: '$($categories[$i].name)'" }
        Write-Utf8 (Join-Path $root 'data.yaml') (($yaml -join "`n")+"`n")
        $manifest | Export-Csv (Join-Path $root 'manifest.csv') -NoTypeInformation -Encoding UTF8
        $missing | Export-Csv (Join-Path $root 'missing_images.csv') -NoTypeInformation -Encoding UTF8
        Write-Utf8 (Join-Path $root 'invalid_annotations.json') (ConvertTo-Json -InputObject @($invalid.ToArray()) -Depth 5)
        $summary=[ordered]@{task=$task.Name;registered_images=$data.images.Count;source_annotations=$data.annotations.Count;exported_images=$manifest.Count;split_images=$splitCounts;exported_boxes_by_class=$classCounts;missing_images=$missing.Count;missing_labeled_images=@($missing | Where-Object has_labels).Count;excluded_matched_unlabeled_images=$unlabeled;aspect_mismatch_images=$aspectMismatch;invalid_annotations=$invalid.Count;clipped_boxes=$clipped;split_policy='SHA256 shoe-v1 + shooting date; 80/10/10 buckets; labeled matched images only'}
        Write-Utf8 (Join-Path $root 'summary.json') ($summary | ConvertTo-Json -Depth 6)
        $summaries += $summary
        Write-Output ($summary | ConvertTo-Json -Depth 6)
    }
    Write-Utf8 (Join-Path $OutputDir 'summary.json') (ConvertTo-Json -InputObject $summaries -Depth 6)
} finally { $pic.Dispose() }

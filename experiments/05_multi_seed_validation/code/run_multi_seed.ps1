$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'
$Root = '<USER_HOME>\Documents\Codex\2026-08-12\c-users-27356-documents-codex-2026\outputs\vision_mamba_experiments_seed42_20260812\temporal_mamba'
$Python = Join-Path $Root 'temporal_pytorch_env\.venv\Scripts\python.exe'
$Script = Join-Path $Root 'temporal_pytorch_env\train_gaze_structure.py'
$Cache = Join-Path $Root 'temporal_pytorch_runs\densenet121_nine_gaze_features_seed42.pt'
$StatusPath = Join-Path $Root 'multi_seed\status.json'
$Data = '<DATASET_ROOT>'
$items = @()
foreach ($seed in @(43,44)) {
  foreach ($variant in @('mean','clinical','random','row_major')) {
    $items += [ordered]@{seed=$seed; variant=$variant; state='pending'; exit_code=$null}
  }
}
function Save-Status {
  [ordered]@{protocol='Seeds 42/43/44; shared feature cache; seed-specific stratified splits; 100 epochs';updated_at=(Get-Date).ToString('o');runs=$items} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $StatusPath -Encoding utf8
}
New-Item -ItemType Directory -Force -Path (Split-Path $StatusPath), (Join-Path $Root 'multi_seed\logs') | Out-Null
Save-Status
foreach ($item in $items) {
  $folder = "seed_$($item.seed)\$($item.variant)"
  $output = Join-Path $Root "multi_seed\$folder"
  if (Test-Path -LiteralPath (Join-Path $output 'test_results.json')) { $item.state='completed';$item.exit_code=0;Save-Status;continue }
  $item.state='running';Save-Status
  & $Python $Script --data-dir $Data --cache $Cache --output-dir $output --variant $item.variant --epochs 100 --batch-size 64 --seed $item.seed --device cuda:0 *>&1 | Tee-Object -FilePath (Join-Path $Root "multi_seed\logs\seed_$($item.seed)_$($item.variant).log")
  $item.exit_code=$LASTEXITCODE;$item.state=if($LASTEXITCODE -eq 0){'completed'}else{'failed'};Save-Status
  if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
}

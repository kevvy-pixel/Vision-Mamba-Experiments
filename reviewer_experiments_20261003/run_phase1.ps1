param([string]$Root='D:\CondaData\Vision-Mamba-Reviewer-Experiments',[string]$Data='D:\Source_9gaze_composed',[string]$Python='python',[string]$Pretrained=(Join-Path $Root 'pretrained_weights\densenet121-a639ec97.pth'),[string]$Cache=(Join-Path $Root 'dependencies\densenet121_nine_gaze_features_seed42.pt'))
$ErrorActionPreference = 'Stop'
$Temporal = Join-Path $Root 'code\temporal_mamba\train_temporal_ablation.py'
$Structure = Join-Path $Root 'code\temporal_mamba\train_gaze_structure.py'
$E2E = Join-Path $Root 'code\temporal_mamba\train_end_to_end_densenet_mamba.py'
$Out = Join-Path $Root 'outputs\phase1'
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Set-Location $Root
& $Python (Join-Path $Root 'code\make_fixed_split.py') --data-dir $Data --output (Join-Path $Out 'fixed_split.csv') --seed 42

foreach ($seed in @(42,43,44)) {
  foreach ($model in @('mean','mamba')) {
    $dir = Join-Path $Out ("temporal_{0}_seed{1}" -f $model,$seed)
    & $Python $Temporal --data-dir $Data --pretrained $Pretrained --cache $Cache --output-dir $dir --model $model --epochs 100 --batch-size 64 --seed $seed --split-seed 42 --device cuda:0
  }
}

foreach ($variant in @('clinical','random','reverse','row_major','closed_loop','circular_pe','bidirectional','center_ring')) {
  $dir = Join-Path $Out ("structure_{0}_seed42" -f $variant)
  & $Python $Structure --data-dir $Data --cache $Cache --output-dir $dir --variant $variant --epochs 100 --batch-size 64 --seed 42 --split-seed 42 --device cuda:0
}
foreach ($seed in @(43,44)) {
  foreach ($variant in @('clinical','random','row_major')) {
    $dir = Join-Path $Out ("structure_{0}_seed{1}" -f $variant,$seed)
    & $Python $Structure --data-dir $Data --cache $Cache --output-dir $dir --variant $variant --epochs 100 --batch-size 64 --seed $seed --split-seed 42 --device cuda:0
  }
}

foreach ($seed in @(42,43,44)) {
  $dir = Join-Path $Out ("gazemamba_densenet_e2e_seed{0}" -f $seed)
  & $Python $E2E --data-dir $Data --pretrained $Pretrained --output-dir $dir --epochs 100 --batch-size 4 --seed $seed --split-seed 42 --device cuda:0 --amp
}
Write-Output 'PHASE1_COMPLETE'

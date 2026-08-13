$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'

$Bundle = '<USER_HOME>\Documents\Codex\2026-08-12\c-users-27356-documents-codex-2026\outputs\vision_mamba_experiments_seed42_20260812'
$EnvDir = Join-Path $Bundle 'temporal_pytorch_env'
$Python = Join-Path $EnvDir '.venv\Scripts\python.exe'
$Script = Join-Path $EnvDir 'train_temporal_ablation.py'
$Data = '<DATASET_ROOT>'
$Weights = Join-Path $Bundle 'pretrained_weights\densenet121-a639ec97.pth'
$Cache = Join-Path $Bundle 'temporal_pytorch_runs\densenet121_nine_gaze_features_seed42.pt'
$LogDir = Join-Path $Bundle 'logs'
$Status = Join-Path $Bundle 'temporal_status.json'

New-Item -ItemType Directory -Force -Path (Split-Path $Cache), $LogDir | Out-Null

$state = [ordered]@{
    protocol = 'Frozen DenseNet-121 nine-gaze features; clinical order 5-2-3-6-9-8-7-4-1; source 70/20/10; seed 42; 100 epochs'
    implementation = 'Pure PyTorch explicit selective scan; no mamba_ssm/Triton/custom CUDA'
    feature_extraction = 'pending'
    temporal_mamba = 'pending'
    mean_pooling_control = 'pending'
    updated_at = (Get-Date).ToString('o')
}
function Save-State {
    $state.updated_at = (Get-Date).ToString('o')
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Status -Encoding utf8
}

Save-State

if (-not (Test-Path -LiteralPath $Cache)) {
    $state.feature_extraction = 'running'; Save-State
    & $Python $Script --data-dir $Data --pretrained $Weights --output-dir (Join-Path $Bundle 'temporal_pytorch_runs\extract') --cache $Cache --model mamba --extract-only --feature-batch-size 4 --num-workers 0 --seed 42 --device cuda:0 *>&1 | Tee-Object -FilePath (Join-Path $LogDir 'temporal_feature_extraction.log')
    if ($LASTEXITCODE -ne 0) { $state.feature_extraction = 'failed'; Save-State; exit $LASTEXITCODE }
}
$state.feature_extraction = 'completed'; Save-State

# Innovation model first.
$state.temporal_mamba = 'running'; Save-State
& $Python $Script --data-dir $Data --pretrained $Weights --output-dir (Join-Path $Bundle 'temporal_pytorch_runs\clinical_temporal_mamba_seed42') --cache $Cache --model mamba --epochs 100 --batch-size 64 --num-workers 0 --seed 42 --device cuda:0 *>&1 | Tee-Object -FilePath (Join-Path $LogDir 'clinical_temporal_mamba_seed42.log')
if ($LASTEXITCODE -ne 0) { $state.temporal_mamba = 'failed'; Save-State; exit $LASTEXITCODE }
$state.temporal_mamba = 'completed'; Save-State

# Matched no-temporal control over the identical cached features.
$state.mean_pooling_control = 'running'; Save-State
& $Python $Script --data-dir $Data --pretrained $Weights --output-dir (Join-Path $Bundle 'temporal_pytorch_runs\mean_pooling_control_seed42') --cache $Cache --model mean --epochs 100 --batch-size 64 --num-workers 0 --seed 42 --device cuda:0 *>&1 | Tee-Object -FilePath (Join-Path $LogDir 'mean_pooling_control_seed42.log')
if ($LASTEXITCODE -ne 0) { $state.mean_pooling_control = 'failed'; Save-State; exit $LASTEXITCODE }
$state.mean_pooling_control = 'completed'; Save-State


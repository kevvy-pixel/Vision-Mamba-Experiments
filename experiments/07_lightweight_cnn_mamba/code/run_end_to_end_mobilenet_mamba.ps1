$ErrorActionPreference='Stop'
$env:PYTHONUNBUFFERED='1'
$Root='<USER_HOME>\Documents\Codex\2026-08-12\c-users-27356-documents-codex-2026\outputs\vision_mamba_experiments_seed42_20260812\temporal_mamba'
$Bundle=Split-Path $Root
$Python=Join-Path $Root 'temporal_pytorch_env\.venv\Scripts\python.exe'
$Script=Join-Path $Root 'temporal_pytorch_env\train_end_to_end_mobilenet_mamba.py'
$Output=Join-Path $Root 'end_to_end\mobilenet_v3_small_clinical_mamba_mean_seed42'
$Log=Join-Path $Root 'end_to_end\logs\mobilenet_v3_small_clinical_mamba_mean_seed42.log'
$Status=Join-Path $Output 'status.json'
New-Item -ItemType Directory -Force -Path $Output,(Split-Path $Log)|Out-Null
[ordered]@{state='running';started_at=(Get-Date).ToString('o');seed=42;epochs=100;protocol='Stage-1 frozen MobileNet Mamba head initialization, then all parameters jointly fine-tuned'}|ConvertTo-Json|Set-Content -LiteralPath $Status -Encoding utf8
& $Python $Script --data-dir '<DATASET_ROOT>' `
 --backbone-weights (Join-Path $Bundle 'pretrained_weights\mobilenet_v3_small-047dcff4.pth') `
 --head-checkpoint (Join-Path $Root 'lightweight_cnn_poc\mamba_mean_seed42\best.pt') `
 --output-dir $Output --epochs 100 --batch-size 8 --num-workers 0 --seed 42 --device cuda:0 --amp *>&1 | Tee-Object -FilePath $Log
$code=$LASTEXITCODE
[ordered]@{state=if($code -eq 0){'completed'}else{'failed'};exit_code=$code;finished_at=(Get-Date).ToString('o')}|ConvertTo-Json|Set-Content -LiteralPath $Status -Encoding utf8
exit $code

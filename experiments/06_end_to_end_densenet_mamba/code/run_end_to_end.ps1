$ErrorActionPreference='Stop'
$env:PYTHONUNBUFFERED='1'
$Root='<USER_HOME>\Documents\Codex\2026-08-12\c-users-27356-documents-codex-2026\outputs\vision_mamba_experiments_seed42_20260812\temporal_mamba'
$Bundle=Split-Path $Root
$Python=Join-Path $Root 'temporal_pytorch_env\.venv\Scripts\python.exe'
$Output=Join-Path $Root 'end_to_end\densenet121_clinical_mamba_seed42'
$Status=Join-Path $Root 'end_to_end\status.json'
New-Item -ItemType Directory -Force -Path $Output,(Join-Path $Root 'end_to_end\logs')|Out-Null
[ordered]@{state='running';started_at=(Get-Date).ToString('o');protocol='All DenseNet-121 parameters + Clinical Pure-PyTorch Temporal Mamba jointly optimized; batch 4; AMP; activation checkpointing; 100 epochs'}|ConvertTo-Json|Set-Content -LiteralPath $Status -Encoding utf8
& $Python (Join-Path $Root 'temporal_pytorch_env\train_end_to_end_densenet_mamba.py') `
 --data-dir '<DATASET_ROOT>' `
 --pretrained (Join-Path $Bundle 'pretrained_weights\densenet121-a639ec97.pth') `
 --output-dir $Output --epochs 100 --batch-size 4 --num-workers 0 --seed 42 --device cuda:0 --amp *>&1 | Tee-Object -FilePath (Join-Path $Root 'end_to_end\logs\densenet121_clinical_mamba_seed42.log')
$code=$LASTEXITCODE
[ordered]@{state=if($code -eq 0){'completed'}else{'failed'};exit_code=$code;finished_at=(Get-Date).ToString('o')}|ConvertTo-Json|Set-Content -LiteralPath $Status -Encoding utf8
exit $code

$ErrorActionPreference='Stop'
$env:PYTHONUNBUFFERED='1'
$Root='<USER_HOME>\Documents\Codex\2026-08-12\c-users-27356-documents-codex-2026\outputs\vision_mamba_experiments_seed42_20260812\temporal_mamba'
$Python=Join-Path $Root 'temporal_pytorch_env\.venv\Scripts\python.exe'
$Script=Join-Path $Root 'temporal_pytorch_env\train_lightweight_cnn_mamba_attention.py'
$ExperimentRoot=Join-Path $Root 'lightweight_cnn_mamba_attention'
$LogRoot=Join-Path $ExperimentRoot 'logs'
$Status=Join-Path $ExperimentRoot 'status.json'
New-Item -ItemType Directory -Force -Path $ExperimentRoot,$LogRoot|Out-Null
$models=@('cnn_mean','cnn_mamba_mean','cnn_mamba_attention')
[ordered]@{state='running';started_at=(Get-Date).ToString('o');models=$models;seed=42;epochs=100}|ConvertTo-Json|Set-Content -LiteralPath $Status -Encoding utf8
foreach($model in $models){
  $Output=Join-Path $ExperimentRoot ($model+'_seed42')
  & $Python $Script --data-dir '<DATASET_ROOT>' --output-dir $Output --model $model --epochs 100 --batch-size 8 --num-workers 0 --seed 42 --device cuda:0 --amp *>&1 | Tee-Object -FilePath (Join-Path $LogRoot ($model+'_seed42.log'))
  if($LASTEXITCODE -ne 0){
    [ordered]@{state='failed';model=$model;exit_code=$LASTEXITCODE;finished_at=(Get-Date).ToString('o')}|ConvertTo-Json|Set-Content -LiteralPath $Status -Encoding utf8
    exit $LASTEXITCODE
  }
}
[ordered]@{state='completed';exit_code=0;finished_at=(Get-Date).ToString('o');models=$models;seed=42;epochs=100}|ConvertTo-Json|Set-Content -LiteralPath $Status -Encoding utf8

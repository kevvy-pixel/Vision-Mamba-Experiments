$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'

$Python = '<USER_HOME>\miniconda3\envs\pytorch\python.exe'
$Repo = '<USER_HOME>\Documents\Codex\2026-08-12\Vision-Mamba-main'
$Data = '<DATASET_ROOT>'
$Bundle = '<USER_HOME>\Documents\Codex\2026-08-12\c-users-27356-documents-codex-2026\outputs\vision_mamba_experiments_seed42_20260812'
$StatusPath = Join-Path $Bundle 'status.json'

$experiments = @(
    [ordered]@{ name='densenet121'; batch_size=4; state='pending'; exit_code=$null; started_at=$null; finished_at=$null },
    [ordered]@{ name='vit';         batch_size=2; state='pending'; exit_code=$null; started_at=$null; finished_at=$null },
    [ordered]@{ name='swin_b';      batch_size=1; state='pending'; exit_code=$null; started_at=$null; finished_at=$null }
)

function Save-Status {
    $payload = [ordered]@{
        protocol = 'source-only stratified 70/20/10, seed 42, 100 epochs'
        updated_at = (Get-Date).ToString('o')
        experiments = $experiments
        blocked = @(
            [ordered]@{ name='resnet50'; reason='README requires project-specific pretrained_weights/ResNet.pkl, which is not present and has no public URL.' },
            [ordered]@{ name='vim_base_mamba'; reason='Official selective_scan_cuda/causal_conv1d CUDA extensions are unavailable on this Windows host; README environment is Linux-oriented.' },
            [ordered]@{ name='inspiration_ablations'; reason='The repository implements only the linear clinical-order Vim+Mamba model, not the proposed ablation variants.' }
        )
    }
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StatusPath -Encoding utf8
}

Save-Status

foreach ($experiment in $experiments) {
    $name = $experiment.name
    $runDir = Join-Path $Bundle "runs\${name}_source_721_seed42"
    $resultFile = Join-Path $runDir 'test_results.json'
    if (Test-Path -LiteralPath $resultFile) {
        $experiment.state = 'completed'
        $experiment.exit_code = 0
        Save-Status
        continue
    }

    $experiment.state = 'running'
    $experiment.started_at = (Get-Date).ToString('o')
    Save-Status
    $logFile = Join-Path $Bundle "logs\${name}_source_721_seed42.log"
    $weightFile = switch ($name) {
        'densenet121' { Join-Path $Bundle 'pretrained_weights\densenet121-a639ec97.pth' }
        'vit'         { Join-Path $Bundle 'pretrained_weights\vit_base_patch16_224_in21k.pth' }
        'swin_b'      { Join-Path $Bundle 'pretrained_weights\swin_b-68c6b09e.pth' }
    }

    & $Python (Join-Path $Repo 'train_source_domain.py') `
        --source-dir $Data `
        --reference-dir (Join-Path $Repo 'model_sources') `
        --pretrained $weightFile `
        --model $name `
        --output-dir $runDir `
        --epochs 100 `
        --seed 42 `
        --batch-size $experiment.batch_size `
        --num-workers 0 `
        --device cuda:0 *>&1 | Tee-Object -FilePath $logFile

    $experiment.exit_code = $LASTEXITCODE
    $experiment.finished_at = (Get-Date).ToString('o')
    $experiment.state = if ($LASTEXITCODE -eq 0) { 'completed' } else { 'failed' }
    Save-Status
    if ($LASTEXITCODE -ne 0) {
        break
    }
}


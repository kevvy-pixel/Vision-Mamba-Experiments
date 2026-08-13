$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'

$Root = '<USER_HOME>\Documents\Codex\2026-08-12\c-users-27356-documents-codex-2026\outputs\vision_mamba_experiments_seed42_20260812\temporal_mamba'
$Python = Join-Path $Root 'temporal_pytorch_env\.venv\Scripts\python.exe'
$Script = Join-Path $Root 'temporal_pytorch_env\train_gaze_structure.py'
$Cache = Join-Path $Root 'temporal_pytorch_runs\densenet121_nine_gaze_features_seed42.pt'
$Experiments = Join-Path $Root 'experiments'
$Logs = Join-Path $Root 'logs'
$StatusPath = Join-Path $Root 'structure_ablation_status.json'
$Data = '<DATASET_ROOT>'

New-Item -ItemType Directory -Force -Path $Experiments, $Logs | Out-Null

$queue = @(
    [ordered]@{ variant='random';        folder='02_random_order_mamba';   state='pending'; exit_code=$null },
    [ordered]@{ variant='reverse';       folder='03_reverse_order_mamba';  state='pending'; exit_code=$null },
    [ordered]@{ variant='row_major';     folder='04_row_major_mamba';      state='pending'; exit_code=$null },
    [ordered]@{ variant='closed_loop';   folder='05_closed_loop_mamba';    state='pending'; exit_code=$null },
    [ordered]@{ variant='circular_pe';   folder='06_circular_pe_mamba';    state='pending'; exit_code=$null },
    [ordered]@{ variant='bidirectional'; folder='07_bidirectional_mamba';  state='pending'; exit_code=$null },
    [ordered]@{ variant='center_ring';   folder='08_center_ring_mamba';    state='pending'; exit_code=$null }
)

function Save-Status {
    [ordered]@{
        protocol = 'Shared frozen 9x1024 DenseNet features; split 1202/343/171; seed 42; 100 epochs; d_model 192; d_state 16; depth 2'
        updated_at = (Get-Date).ToString('o')
        queue = $queue
        graph_mamba = 'deferred by experimental design'
    } | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $StatusPath -Encoding utf8
}

Save-Status
foreach ($item in $queue) {
    $output = Join-Path $Experiments $item.folder
    $result = Join-Path $output 'test_results.json'
    if (Test-Path -LiteralPath $result) {
        $item.state = 'completed'; $item.exit_code = 0; Save-Status; continue
    }
    $item.state = 'running'; Save-Status
    & $Python $Script `
        --data-dir $Data `
        --cache $Cache `
        --output-dir $output `
        --variant $item.variant `
        --epochs 100 `
        --batch-size 64 `
        --seed 42 `
        --device cuda:0 *>&1 | Tee-Object -FilePath (Join-Path $Logs "$($item.folder).log")
    $item.exit_code = $LASTEXITCODE
    $item.state = if ($LASTEXITCODE -eq 0) { 'completed' } else { 'failed' }
    Save-Status
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}


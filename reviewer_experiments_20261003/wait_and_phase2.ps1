$ErrorActionPreference='Stop'
$Root='D:\CondaData\Vision-Mamba-Reviewer-Experiments'; $Phase1=Join-Path $Root 'outputs\phase1'; $Log=Join-Path $Root 'outputs\phase2_launcher.log'; New-Item -ItemType Directory -Force -Path (Join-Path $Root 'outputs') | Out-Null
while($true){$done=0; foreach($s in 42,43,44){if(Test-Path (Join-Path $Phase1 ("gazemamba_densenet_e2e_seed{0}\test_results.json" -f $s))){$done++}}; if($done -eq 3){break}; Start-Sleep -Seconds 30}
& powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root 'run_phase2.ps1') *>&1 | Tee-Object -FilePath $Log

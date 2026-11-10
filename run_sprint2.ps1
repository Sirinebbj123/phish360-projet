# Sprint 2 - enchaîne les tâches 6 à 12 (à lancer depuis la racine du projet, .venv activé)
# Usage :  powershell -ExecutionPolicy Bypass -File .\run_sprint2.ps1
#          powershell -ExecutionPolicy Bypass -File .\run_sprint2.ps1 -From 8     (reprendre à la tâche 8)
param([int]$From = 6)
$ErrorActionPreference = "Continue"   # les avertissements Python (stderr) ne doivent pas arrêter le script
$env:PYSPARK_SUBMIT_ARGS = "--conf spark.sql.ansi.enabled=false pyspark-shell"   # PySpark 4 (cf. Sprint 1)
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
New-Item -ItemType Directory -Force logs | Out-Null

function Step($n, $title, $cmd) {
    if ($n -lt $From) { Write-Host "-- tâche $n déjà faite, on passe"; return }
    Write-Host "`n===== Tâche $n : $title =====" -ForegroundColor Cyan
    $t0 = Get-Date
    # Tee-Object : on voit la sortie ET elle est gardée dans logs\
    Invoke-Expression "$cmd 2>&1" | Tee-Object -FilePath "logs\tache$n.log"
    if ($LASTEXITCODE -ne 0) { Write-Error "Tâche $n en échec (voir logs\tache$n.log)"; exit 1 }
    Write-Host ("Tâche {0} terminée en {1:N0} min" -f $n, ((Get-Date) - $t0).TotalMinutes) -ForegroundColor Green
}

python src\check_deep.py
Step 6  "CharCNN URL (3 protocoles)"          "python src\url_charcnn.py"
Step 7  "DistilBERT texte HTML"               "python src\html_distilbert.py"
Step 8  "ResNet18 captures"                   "python src\img_resnet18.py"
Step 9  "Fusion des modalités + ablation"     "python src\fusion.py"
Step 10 "Évaluation finale RQ1 / RQ2"         "python src\evaluate_final.py"
Step 12 "Explicabilité et erreurs"            "python src\explain.py --gradcam 6 --tokens 5"
Write-Host "`nSprint 2 terminé : voir results\rapport_sprint2.md et results\figures\" -ForegroundColor Green

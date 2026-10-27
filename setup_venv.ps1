# Tâche 1 - crée l'environnement virtuel du projet Phish360 (à lancer dans le terminal VS Code)
# Usage :  powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1        (CPU)
#          powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1 -Gpu   (carte NVIDIA)
param([switch]$Gpu)

Write-Host "1/4 Création de .venv avec Python 3.11"
py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { Write-Error "Python 3.11 introuvable : installez-le depuis python.org"; exit 1 }

$py = ".\.venv\Scripts\python.exe"
Write-Host "2/4 Mise à jour de pip"
& $py -m pip install --upgrade pip

Write-Host "3/4 Installation des bibliothèques (requirements.txt)"
& $py -m pip install -r requirements.txt

Write-Host "4/4 Installation de PyTorch"
if ($Gpu) {
    & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
} else {
    & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

Write-Host "Terminé. Activez l'environnement avec :  .\.venv\Scripts\Activate.ps1"

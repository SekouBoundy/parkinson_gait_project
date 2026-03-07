"# parkinson_gait_project" 
# Parkinson Gait � Step 1: Inspect & Preprocess

## Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset layout
```
data/raw/
  parkinson/   <- label 1
  control/     <- label 0
```

## Commands (run from project root)
```powershell
python -m src.inspect_dataset
python -m src.preprocess_and_window
python -m src.preprocess_and_window --no_filter
python tests/test_shapes.py
```

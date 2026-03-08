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

## To run with the real PhysioNet dataset:
```powershell
# Run this ONCE every time you open a new terminal
.venv\Scripts\Activate.ps1
```
```powershell
python -m src.inspect_dataset
# After running organize_data.py to sort files into subfolders:
python -m src.preprocess_and_window
# Skip the low-pass filter (optional):
python -m src.preprocess_and_window --no_filter
# Run tests:
python tests/test_shapes.py
#Run Train
python -m src.train_baseline
python -m src.train
```






 
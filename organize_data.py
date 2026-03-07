import shutil,pathlib  
src=pathlib.Path("data/raw/gait-in-parkinsons-disease-1.0.0")  
parkinson=pathlib.Path("data/raw/parkinson")  
control=pathlib.Path("data/raw/control")  
[shutil.move(str(f),control/f.name) for f in src.glob("*.txt") if f.name.startswith("GaCo")]  
[shutil.move(str(f),parkinson/f.name) for f in src.glob("*.txt") if f.name.startswith(("GaPt","SiPt"))]  
print("Parkinson:",len(list(parkinson.glob("*.txt"))))  
print("Control:",len(list(control.glob("*.txt"))))  

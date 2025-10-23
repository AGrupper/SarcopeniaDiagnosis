# C:\CT_Project\code\SarcopeniaDiagnosis\build_manifest.py
from pathlib import Path, PurePosixPath
import csv, json, nibabel as nib
import numpy as np
from patients import patients

BASE = Path(r"C:\CT_Project")
PRE_RAS = BASE/"data_preproc_ras"
PRE     = BASE/"data_preproc"
OUT     = BASE/"meta"; OUT.mkdir(parents=True, exist_ok=True)

def py(v):
    """המרת טיפוסי NumPy לטיפוסים של Python עבור JSON/CSV."""
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (list, tuple)):
        return [py(x) for x in v]
    return v

rows=[]
for pid in patients:
    p = PRE_RAS/f"{pid}_iso_norm_ras.nii.gz"
    if not p.exists():
        p = PRE/f"{pid}_iso_norm.nii.gz"
    img = nib.load(str(p))
    zooms = img.header.get_zooms()

    rec = {
        "patient_id": pid,
        "nifti_path": str(PurePosixPath(p)),
        "shape_x": int(img.shape[0]),
        "shape_y": int(img.shape[1]),
        "shape_z": int(img.shape[2]),
        "zoom_x":  py(zooms[0]),
        "zoom_y":  py(zooms[1]),
        "zoom_z":  py(zooms[2]),
        "l3_index": ""  # יתמלא אחרי שתבחר L3
    }
    rows.append(rec)

# CSV
with open(OUT/"manifest.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# JSON
with open(OUT/"manifest.json","w",encoding="utf-8") as f:
    json.dump([{k: py(v) for k,v in r.items()} for r in rows], f, ensure_ascii=False, indent=2)

print("Wrote:", OUT/"manifest.csv", "and manifest.json")

from pathlib import Path
import nibabel as nib, numpy as np, json
from patients import patients

BASE = Path(r"C:\CT_Project")
PRE  = BASE/"data_preproc_ras"  # להשתמש ב-RAS אם עשית שלב 1; אחרת data_preproc
HU_MIN, HU_MAX = -150, 300

def stats(pid):
    p = PRE/f"{pid}_iso_norm_ras.nii.gz"
    if not p.exists(): p = BASE/"data_preproc"/f"{pid}_iso_norm.nii.gz"
    d = nib.load(str(p)).get_fdata().astype("float32")  # 0..1
    # החזרה ל-HU לפי הספים שלנו
    hu = d*(HU_MAX-HU_MIN)+HU_MIN
    qs = np.percentile(hu, [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5]).round(1).tolist()
    return {"patient": pid, "HU_min_max": [float(hu.min()), float(hu.max())], "percentiles": qs}

for pid in patients:
    print(json.dumps(stats(pid), ensure_ascii=False))

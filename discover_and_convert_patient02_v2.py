import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pydicom

ROOT = r"C:\CT_Project\data_raw\patient02\DICOM"
OUT  = Path(r"C:\CT_Project\data_nifti\patient02_main.nii.gz")

def uniform_key(ds):
    rows = int(getattr(ds, "Rows", 0))
    cols = int(getattr(ds, "Columns", 0))
    iop  = tuple(round(float(x), 6) for x in getattr(ds, "ImageOrientationPatient", [1,0,0,0,1,0]))
    ps   = tuple(round(float(x), 6) for x in getattr(ds, "PixelSpacing", [1,1]))
    return (rows, cols, iop, ps)

def series_in_dir(dirpath):
    try:
        ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dirpath) or []
    except Exception:
        ids = []
    out = []
    for sid in ids:
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dirpath, sid)
        if files:
            out.append((sid, files, dirpath))
    return out

# אסוף את כל הסדרות בכל התיקיות
candidates = []
for dirpath, _, filenames in os.walk(ROOT):
    if not filenames:
        continue
    candidates.extend(series_in_dir(dirpath))

if not candidates:
    raise RuntimeError("לא נמצאו סדרות DICOM ב-patient02.")

best_group = None
best_info  = None  # (sid, desc, dirpath, key)
best_count = -1

for sid, files, dirpath in candidates:
    groups = {}
    desc = "NA"
    for fp in files:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            if desc == "NA":
                desc = getattr(ds, "SeriesDescription", "NA")
            k = uniform_key(ds)
            groups.setdefault(k, []).append((
                fp,
                getattr(ds, "InstanceNumber", None),
                getattr(ds, "ImagePositionPatient", None),
            ))
        except Exception:
            continue
    if not groups:
        continue
    k_big = max(groups, key=lambda k: len(groups[k]))
    if len(groups[k_big]) > best_count:
        best_count = len(groups[k_big])
        best_group = groups[k_big]
        best_info  = (sid, desc, dirpath, k_big)

if not best_group:
    raise RuntimeError("לא נמצאה קבוצה עקבית של פרוסות.")

# מיין לפי מיקום במרחב (נורמל לצירי התמונה)
rows, cols, iop, ps = best_info[3]
row_v = np.array(iop[:3], float)
col_v = np.array(iop[3:], float)
normal = np.cross(row_v, col_v)

def sort_key(item):
    fp, inst, pos = item
    if pos is None:  # נפילה לגיבוי: לפי InstanceNumber
        return inst if inst is not None else 0
    pos = np.array([float(x) for x in pos], float)
    return float(np.dot(pos, normal))

files_sorted = [fp for fp, _, _ in sorted(best_group, key=sort_key)]

print("Selected series:")
print("  Dir     :", best_info[2])
print("  Desc    :", best_info[1])
print("  Files   :", len(files_sorted))
print("  Key     :", best_info[3])

reader = sitk.ImageSeriesReader()
reader.SetFileNames(files_sorted)
img = reader.Execute()
OUT.parent.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(img, str(OUT))
print("\nWrote:", OUT)
print("Size (x,y,z):", img.GetSize())
print("Spacing (mm):", img.GetSpacing())

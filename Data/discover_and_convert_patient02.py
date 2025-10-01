import os
from pathlib import Path
import SimpleITK as sitk
import pydicom

ROOT = r"C:\CT_Project\data_raw\patient02\DICOM"
OUT  = Path(r"C:\CT_Project\data_nifti\patient02_main.nii.gz")

def series_in_dir(dirpath):
    """החזר [(SeriesID, [files...])] לכל סדרה בתיקייה אחת."""
    try:
        ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dirpath) or []
    except Exception:
        ids = []
    out = []
    for sid in ids:
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dirpath, sid)
        if files:
            out.append((sid, files))
    return out

best = None
best_files = []
report = []

for dirpath, _, filenames in os.walk(ROOT):
    if not filenames:
        continue
    for sid, files in series_in_dir(dirpath):
        # תיאור הסדרה (אם קיים) מתוך הקובץ הראשון
        desc = "NA"
        try:
            ds = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
            desc = getattr(ds, "SeriesDescription", "NA")
        except Exception:
            pass
        report.append((len(files), desc, dirpath))
        if len(files) > len(best_files):
            best = (sid, desc, dirpath)
            best_files = files

# דו"ח תמציתי לבקרה
report.sort(reverse=True)
print("Top series by slice-count:")
for cnt, desc, d in report[:10]:
    print(f"  {cnt:4d} | {desc} | {d}")

if not best_files:
    raise RuntimeError("לא נמצאה סדרת DICOM מתאימה ב-patient02.")

# המרה ל-NIfTI
print("\nSelected:")
print(f"  SeriesID: {best[0]}")
print(f"  Desc    : {best[1]}")
print(f"  Dir     : {best[2]}")
print(f"  Files   : {len(best_files)}")

reader = sitk.ImageSeriesReader()
reader.SetFileNames(best_files)
img = reader.Execute()
OUT.parent.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(img, str(OUT))
print(f"\nWrote: {OUT}")
print("Size (x,y,z):", img.GetSize())
print("Spacing (mm):", img.GetSpacing())

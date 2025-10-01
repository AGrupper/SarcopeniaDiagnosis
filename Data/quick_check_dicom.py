import pydicom, glob, os, sys

def first_file(root):
    # מוצא קובץ ראשון (לא תיקייה) בתוך DICOM
    cand = glob.glob(os.path.join(root, "DICOM", "**", "*"), recursive=True)
    cand = [p for p in cand if os.path.isfile(p)]
    return cand[0] if cand else None

for pid in ["patient01", "patient02"]:
    root = rf"C:\CT_Project\data_raw\{pid}"
    f = first_file(root)
    if not f:
        print(pid, "— לא נמצאו קבצים בתיקיית DICOM")
        continue
    try:
        ds = pydicom.dcmread(f, force=True)
        print(
            f"{pid} OK | Modality={getattr(ds,'Modality','NA')} | "
            f"PatientID={getattr(ds,'PatientID','NA')} | "
            f"StudyDate={getattr(ds,'StudyDate','NA')} | "
            f"SliceThickness={getattr(ds,'SliceThickness','NA')}"
        )
    except Exception as e:
        print(pid, "— שגיאת קריאה:", e)
        sys.exit(1)

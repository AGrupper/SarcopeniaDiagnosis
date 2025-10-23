import SimpleITK as sitk
from pathlib import Path
from patients import patients

DATA_NIFTI = Path(r"C:\CT_Project\data_nifti")
DATA_PRE   = Path(r"C:\CT_Project\data_preproc")

def pick_largest_nifti(patient_id: str) -> Path:
    # מחפש את כל קבצי ה-NIfTI של המטופל ובוחר את הגדול ביותר
    cands = list(DATA_NIFTI.glob(f"{patient_id}_*.nii*"))
    if not cands:
        # נסה גם שם ישן בלי סיומת סדרה (למקרה שהומר בשם פשוט)
        cands = list(DATA_NIFTI.glob(f"{patient_id}.nii*"))
    if not cands:
        raise FileNotFoundError(f"No NIfTI found for {patient_id} in {DATA_NIFTI}")
    return max(cands, key=lambda p: p.stat().st_size)

def preprocess(in_path: Path, out_path: Path, hu_min=-150, hu_max=300, iso=(1.0,1.0,1.0)):
    img = sitk.ReadImage(str(in_path))
    # חלון HU לשריר/שומן, נרמול ל-0..1
    clamped = sitk.Clamp(img, sitk.sitkFloat32, hu_min, hu_max)
    norm = (clamped - hu_min) / float(hu_max - hu_min)

    # ריסמפל לאיזו-ווקסל 1×1×1 מ״מ
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(iso)
    in_size = img.GetSize(); in_sp = img.GetSpacing()
    out_size = [int(round(in_size[i]*in_sp[i]/iso[i])) for i in range(3)]
    resampler.SetSize(out_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    out_img = resampler.Execute(norm)
    DATA_PRE.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(out_path))

if __name__ == "__main__":
    for pid in patients:
        src = pick_largest_nifti(pid)
        dst = DATA_PRE / f"{pid}_iso_norm.nii.gz"
        print(f"[{pid}] using source:", src.name)
        preprocess(src, dst)
        print(f"[{pid}] wrote:", dst)

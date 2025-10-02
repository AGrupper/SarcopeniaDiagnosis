from pathlib import Path
import nibabel as nib

BASE = Path(r"C:\CT_Project")
SRC  = BASE/"data_preproc"
DST  = BASE/"data_preproc_ras"; DST.mkdir(parents=True, exist_ok=True)

def to_ras(src: Path, dst: Path):
    img = nib.load(str(src))
    ras = nib.as_closest_canonical(img)  # reorient only (לא משנה ספייסינג/ערכים)
    nib.save(ras, str(dst))
    return img.affine, ras.affine

for pid in ["patient01","patient02"]:
    s = SRC/f"{pid}_iso_norm.nii.gz"
    d = DST/f"{pid}_iso_norm_ras.nii.gz"
    a0,a1 = to_ras(s,d)
    print(f"{pid}: wrote {d.name}")

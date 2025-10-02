from pathlib import Path
import nibabel as nib

BASE = Path(r"C:\CT_Project")
SRC  = BASE/"data_preproc_ras"  # או data_preproc
DST  = BASE/"data_share"; DST.mkdir(parents=True, exist_ok=True)

# רשימת שדות header שכדאי לנקות אם קיימים
FIELDS = [
    ("descrip", b""), ("aux_file", b""), ("db_name", b""), ("intent_name", b"")
]

for pid in ["patient01","patient02"]:
    s = SRC/f"{pid}_iso_norm_ras.nii.gz"
    if not s.exists(): s = BASE/"data_preproc"/f"{pid}_iso_norm.nii.gz"
    img = nib.load(str(s))
    hdr = img.header.copy()
    for k,v in FIELDS:
        if k in hdr: hdr[k] = v
    clean = nib.Nifti1Image(img.get_fdata(), img.affine, header=hdr)
    out = DST/f"{pid}_iso_norm_clean.nii.gz"
    nib.save(clean, str(out))
    print("Wrote:", out)

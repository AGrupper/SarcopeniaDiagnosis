from pathlib import Path
import nibabel as nib, numpy as np, json

BASE = Path(r"C:\CT_Project")
PRE  = BASE / "data_preproc"

def py(v):
    """המרת טיפוסים של NumPy לטיפוסי Python כדי שיהיו JSON-serializable."""
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (list, tuple)):
        return [py(x) for x in v]
    return v

def report(nii: Path):
    img = nib.load(str(nii))
    hdr = img.header
    d   = img.get_fdata(dtype=np.float32)

    shape = tuple(int(x) for x in img.shape)
    zooms = tuple(float(z) for z in hdr.get_zooms())
    nans  = int(np.isnan(d).sum())
    mins, maxs = float(np.nanmin(d)), float(np.nanmax(d))

    aff = img.affine
    zstep = float(np.linalg.norm(aff[:3, 2])) or zooms[2]

    return {
        "file": nii.name,
        "shape": shape,
        "zooms_mm": zooms,
        "z_step_mm": round(zstep, 3),
        "min": round(mins, 3),
        "max": round(maxs, 3),
        "n_nan": nans,
    }

if __name__ == "__main__":
    for pid in ["patient01","patient02"]:
        p = PRE / f"{pid}_iso_norm.nii.gz"
        R = report(p)
        print(json.dumps({k: py(v) for k, v in R.items()}, ensure_ascii=False))

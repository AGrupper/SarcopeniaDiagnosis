from pathlib import Path
import csv, nibabel as nib, numpy as np, time

BASE = Path(r"C:\CT_Project")
NII  = BASE/"data_nifti"
PRE  = BASE/"data_preproc"
OUT  = BASE/"results"; OUT.mkdir(exist_ok=True, parents=True)

rows=[]
for folder in [NII, PRE]:
    for f in sorted(folder.glob("*.nii.gz")):
        img = nib.load(str(f)); hdr=img.header
        d = img.get_fdata(dtype=np.float32)
        rows.append({
            "file": f.name,
            "folder": folder.name,
            "shape": "x".join(map(str,img.shape)),
            "zooms": "x".join(f"{z:.3g}" for z in hdr.get_zooms()),
            "min": f"{np.nanmin(d):.3f}",
            "max": f"{np.nanmax(d):.3f}",
        })

def write_csv_safely(target: Path, rows):
    try:
        with open(target, "w", newline="", encoding="utf-8") as g:
            w=csv.DictWriter(g, fieldnames=["file","folder","shape","zooms","min","max"])
            w.writeheader(); w.writerows(rows)
        print("Wrote:", target)
    except PermissionError:
        alt = target.with_name(target.stem + f"_{int(time.time())}" + target.suffix)
        with open(alt, "w", newline="", encoding="utf-8") as g:
            w=csv.DictWriter(g, fieldnames=["file","folder","shape","zooms","min","max"])
            w.writeheader(); w.writerows(rows)
        print("Target locked. Wrote fallback file:", alt)

write_csv_safely(OUT/"inventory.csv", rows)

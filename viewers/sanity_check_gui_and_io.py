from pathlib import Path
import SimpleITK as sitk
import matplotlib
try:
    matplotlib.use("QtAgg")
except Exception:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

p = Path(r"C:\CT_Project\data_preproc\patient01_iso_norm.nii.gz")
assert p.exists(), f"לא נמצא: {p}"

img = sitk.ReadImage(str(p))
size = img.GetSize()  # (X, Y, Z)
print("Loaded:", p, "| size:", size)

f = sitk.ExtractImageFilter()
f.SetSize([size[0], size[1], 0])     # לוקח פרוסת 2D
f.SetIndex([0, 0, size[2]//2])       # אמצע בציר Z
sl = f.Execute(img)

arr = sitk.GetArrayFromImage(sl)     # <<< 2D (Y, X) — בלי [0]!
print("slice shape:", arr.shape)

plt.imshow(arr, cmap="gray", vmin=0, vmax=1)
plt.title("בדיקת GUI ו־I/O — אם אתה רואה את זה, הכל טוב")
plt.axis("off")
plt.show()

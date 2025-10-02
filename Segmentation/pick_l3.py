# C:\CT_Project\code\SarcopeniaDiagnosis\pick_l3.py
# בחירת פרוסת L3 לכל מטופל ושמירת האינדקס ל-CSV
# שימוש: python pick_l3.py patient01 | patient02

import sys, csv
from pathlib import Path
import numpy as np
import SimpleITK as sitk

# חלון גרפי חיצוני (עוקף SciView). אם אין PyQt5, נשתמש ב-Tk.
import matplotlib
try:
    matplotlib.use("QtAgg")
except Exception:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

BASE = Path(r"C:\CT_Project")
PRE  = BASE / "data_preproc"
META = BASE / "meta"
META.mkdir(parents=True, exist_ok=True)
CSV_PATH = META / "l3_slices.csv"

HELP = "←/A אחורה | →/D קדימה | PgUp -10 | PgDn +10 | Home/End התחלה/סוף | S שמירה | Q/Esc יציאה"

def load_or_init_csv():
    rows = {}
    if CSV_PATH.exists():
        with open(CSV_PATH, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                try:
                    rows[r["patient_id"]] = int(r["slice_index"])
                except Exception:
                    pass
    return rows

def write_csv(rows: dict):
    with open(CSV_PATH, "w", newline='', encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id","slice_index"])
        w.writeheader()
        for pid, k in rows.items():
            w.writerow({"patient_id": pid, "slice_index": int(k)})

def pick_slice(pid: str):
    nii = PRE / f"{pid}_iso_norm.nii.gz"
    if not nii.exists():
        sys.exit(f"לא נמצא קובץ ה-preprocess: {nii}")

    # קוראים את הנפח (ערכים 0..1), לא טוענים פרוסות מיותרות
    img = sitk.ReadImage(str(nii))
    size = list(img.GetSize())   # [X, Y, Z]
    Z = size[2]

    rows = load_or_init_csv()
    k = rows.get(pid, Z // 2)    # נתחיל מאמצע הנפח או מערך קודם

    def get_slice(idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, Z-1))
        f = sitk.ExtractImageFilter()
        f.SetSize([size[0], size[1], 0])   # מחלץ 2D
        f.SetIndex([0, 0, idx])
        sl = f.Execute(img)
        arr2d = sitk.GetArrayFromImage(sl)  # <<< 2D (Y, X) — בלי [0]!
        return np.clip(arr2d, 0.0, 1.0)

    fig, ax = plt.subplots()
    try:
        fig.canvas.manager.set_window_title(f"{pid} | {HELP}")
    except Exception:
        pass

    arr = get_slice(k)
    im = ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
    ttl = ax.set_title(f"{pid} | slice k={k}/{Z-1}")
    plt.axis('off')

    def refresh():
        a = get_slice(k)
        im.set_data(a)
        ttl.set_text(f"{pid} | slice k={k}/{Z-1}")
        fig.canvas.draw_idle()

    def on_key(e):
        nonlocal k, rows
        key = (e.key or "").lower()
        if key in ('right','d'):
            k = min(Z-1, k+1); refresh()
        elif key in ('left','a'):
            k = max(0, k-1); refresh()
        elif key == 'pagedown':
            k = min(Z-1, k+10); refresh()
        elif key == 'pageup':
            k = max(0, k-10); refresh()
        elif key == 'end':
            k = Z-1; refresh()
        elif key == 'home':
            k = 0; refresh()
        elif key in ('s','enter'):
            rows[pid] = int(k)
            write_csv(rows)
            print(f"נשמר: {pid},{k} → {CSV_PATH}")
        elif key in ('q','escape'):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    print(HELP)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("patient01","patient02"):
        sys.exit("שימוש: python pick_l3.py patient01|patient02")
    pick_slice(sys.argv[1])

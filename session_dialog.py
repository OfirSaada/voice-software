# -*- coding: utf-8 -*-
"""
חלון פתיחה משותף ל-transcribe_v3.py ול-trainee_charts.py:
  - קובץ הגיחה (וידאו MP4 או שמע)
  - קובץ נתוני המאמן (CSV) - לא חובה
  - הפרש זמן בין הסרטון ל-CSV
  - זמני התחלה של התרגילים

ההגדרות נשמרות ליד הסרטון בקובץ  <שם הסרטון>_session.json
ונטענות אוטומטית בפעם הבאה שבוחרים את אותו סרטון.
"""
import json
import os
import re

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v")
MEDIA_TYPES = [("וידאו / שמע", "*.mp4 *.mov *.mkv *.avi *.m4v *.mp3 *.wav *.m4a *.flac *.ogg"), ("All", "*.*")]

EXAMPLE = """# שורה לכל תרגיל: זמן התחלה ושם התרגיל. אפשר להוסיף זמן סיום ויעדים.
# דוגמאות (שורות שמתחילות ב-# לא נקראות):
# 02:10 המראה
# 05:30 נסיקה
# 12:00 הקפה גובה=500 מהירות=80
# 21:40 - 24:00 גובה נמוך רדאר=100
# (גובה = ברומטרי מעל פני הים, רדאר = מעל הקרקע, ברגל; מהירות בקשר; כיוון במעלות)
"""


AUTO_WORD = "אוטומטי"


def parse_offset(txt):
    """'אוטומטי' / ריק -> 'auto';  '262' / '4:22' / '-30' -> שניות;  אחרת None."""
    t = str(txt).strip().lower()
    if t in ("", "auto", "a", AUTO_WORD, "אוטו"):
        return "auto"
    m = re.fullmatch(r"(-?)(\d+):(\d{2})", t)
    if m:
        return (-1 if m.group(1) else 1) * (int(m.group(2)) * 60 + int(m.group(3)))
    try:
        return float(t)
    except ValueError:
        return None


def session_path(media_path):
    return os.path.splitext(media_path)[0] + "_session.json"


def load_session(media_path):
    if not media_path:
        return {}
    p = session_path(media_path)
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def save_session(media_path, d):
    try:
        with open(session_path(media_path), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
    except Exception as e:  # noqa: BLE001
        print(f"[אזהרה] לא נשמר קובץ ההגדרות: {e}")


# ---------- פענוח שורות התרגילים ----------
_T = r"(\d{1,2}):(\d{2})(?::(\d{2}))?"
TARGET_KEYS = {
    "גובה": "alt", "alt": "alt", "altitude": "alt",
    "רדאר": "agl", "רדר": "agl", "גובה_קרקע": "agl", "agl": "agl",
    "מהירות": "spd", "speed": "spd", "spd": "spd",
    "כיוון": "hdg", "heading": "hdg", "hdg": "hdg",
}


def _secs(m):
    a, b, c = m.group(1), m.group(2), m.group(3)
    return int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)


def _parse_line(line):
    """שורה אחת -> {name, start, end (או None), targets, targets_txt} או None אם אין זמן."""
    targets, tparts = {}, []
    for m in re.finditer(r"([^\s=]+)\s*=\s*(-?\d+(?:\.\d+)?)", line):
        k = TARGET_KEYS.get(m.group(1).lower())
        if k:
            targets[k] = float(m.group(2))
            tparts.append(f"{m.group(1)}={m.group(2)}")
    line_wo = re.sub(r"[^\s=]+\s*=\s*-?\d+(?:\.\d+)?", " ", line)
    times = list(re.finditer(_T, line_wo))
    if not times:
        return None
    start = _secs(times[0])
    end = _secs(times[1]) if len(times) > 1 else None
    name = re.sub(_T, " ", line_wo)
    name = re.sub(r"[-–—,|]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return {"name": name, "start": start, "end": end, "targets": targets, "targets_txt": " ".join(tparts)}


def parse_exercises(text, dur=None):
    """מחזיר [{name, start, end, targets:{alt,agl,spd,hdg}}] ממוין לפי זמן, ורשימת שגיאות."""
    out, errors = [], []
    for ln_no, raw in enumerate((text or "").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ex = _parse_line(line)
        if ex is None:
            errors.append(f"שורה {ln_no}: לא נמצא זמן (דקות:שניות) - \"{line}\"")
            continue
        ex["name"] = ex["name"] or f"תרגיל {len(out) + 1}"
        if ex["end"] is not None and ex["end"] <= ex["start"]:
            errors.append(f"שורה {ln_no}: זמן הסיום לפני זמן ההתחלה - \"{line}\"")
            ex["end"] = None
        out.append(ex)
    out.sort(key=lambda x: x["start"])
    for i, ex in enumerate(out):
        if ex["end"] is None:
            ex["end"] = out[i + 1]["start"] if i + 1 < len(out) else dur
    return out, errors


def mmss(t):
    t = int(round(t))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60:02d}:{t % 60:02d}"


def rows_from_text(text):
    rows = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ex = _parse_line(line)
        if ex:
            rows.append((ex["name"], mmss(ex["start"]), mmss(ex["end"]) if ex["end"] is not None else "", ex["targets_txt"]))
    return rows


def text_from_rows(rows):
    lines = []
    for name, st, en, tg in rows:
        if not (name or st):
            continue
        lines.append(" ".join(x for x in (st, ("- " + en) if en else "", name, tg) if x))
    return "\n".join(lines)


# שמות תרגילים מוצעים ברשימה (אפשר גם להקליד שם אחר)
EX_NAMES = ["המראה", "נסיקה", "התיישרות", "טיסה ישרה ואופקית", "הטיות", "פניות", "פנייה מיטבית", "הנמכה",
            "תבנית", "הקפה", "גובה נמוך", "שמינייה", "ריחוף", "נחיתה", "אוטורוטציה", "תקלה", "ניווט"]


# ---------- החלון ----------
def ask_session(media_path="", need_media=True, title="הגדרות גיחה"):
    """פותח חלון ומחזיר dict {media, csv, offset, exercises_text} או None אם בוטל."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    res = {}
    root = tk.Tk()
    root.title(title)
    root.resizable(True, True)
    pad = {"padx": 8, "pady": 4}

    v_media = tk.StringVar(value=media_path or "")
    v_csv = tk.StringVar()
    v_off = tk.StringVar(value=AUTO_WORD)

    def fill_from(path):
        s = load_session(path)
        v_csv.set(s.get("csv", ""))
        o = s.get("offset", "auto")
        v_off.set(AUTO_WORD if o in (None, "", "auto") else str(o))
        set_rows(rows_from_text(s.get("exercises_text", "")))

    def pick_media():
        p = filedialog.askopenfilename(title="בחר את קובץ הגיחה", filetypes=MEDIA_TYPES,
                                       initialdir=os.path.dirname(v_media.get()) or None)
        if p:
            v_media.set(p)
            fill_from(p)

    def pick_csv():
        p = filedialog.askopenfilename(title="בחר קובץ נתונים מהמאמן", filetypes=[("CSV", "*.csv"), ("All", "*.*")],
                                       initialdir=os.path.dirname(v_csv.get() or v_media.get()) or None)
        if p:
            v_csv.set(p)

    frm = tk.Frame(root)
    frm.pack(fill="both", expand=True, padx=10, pady=10)
    frm.columnconfigure(1, weight=1)

    r = 0
    if need_media:
        tk.Label(frm, text="קובץ הגיחה - וידאו או שמע", anchor="e").grid(row=r, column=0, columnspan=3, sticky="e", **pad)
        r += 1
        tk.Button(frm, text="בחר", command=pick_media).grid(row=r, column=0, **pad)
        tk.Entry(frm, textvariable=v_media, width=70).grid(row=r, column=1, columnspan=2, sticky="we", **pad)
        r += 1
    else:
        tk.Label(frm, text=os.path.basename(media_path or ""), anchor="e", font=("Segoe UI", 10, "bold")).grid(
            row=r, column=0, columnspan=3, sticky="e", **pad)
        r += 1

    tk.Label(frm, text="קובץ נתונים מהמאמן CSV - לא חובה", anchor="e").grid(row=r, column=0, columnspan=3, sticky="e", **pad)
    r += 1
    tk.Button(frm, text="בחר", command=pick_csv).grid(row=r, column=0, **pad)
    tk.Entry(frm, textvariable=v_csv, width=70).grid(row=r, column=1, sticky="we", **pad)
    tk.Button(frm, text="נקה", command=lambda: v_csv.set("")).grid(row=r, column=2, **pad)
    r += 1

    tk.Label(frm, text="כמה שניות אחרי תחילת קובץ הנתונים התחיל הסרטון", anchor="e").grid(
        row=r, column=1, columnspan=2, sticky="e", **pad)
    tk.Entry(frm, textvariable=v_off, width=10, justify="center").grid(row=r, column=0, **pad)
    r += 1
    tk.Label(frm, text="אוטומטי = מחושב לפי הדיווחים של החניך. אפשר גם לכתוב מספר שניות", anchor="e", fg="#666").grid(
        row=r, column=0, columnspan=3, sticky="e", padx=8)
    r += 1

    tk.Label(frm, text="תרגילים - זמנים בסרטון בדקות:שניות. סיום לא חובה: בלי סיום התרגיל נגמר כשהבא מתחיל",
             anchor="e").grid(row=r, column=0, columnspan=3, sticky="e", **pad)
    r += 1
    tk.Label(frm, text="יעדים לא חובה, למשל  גובה=1500 מהירות=80  או  רדאר=100 כיוון=270", anchor="e", fg="#666").grid(
        row=r, column=0, columnspan=3, sticky="e", padx=8)
    r += 1

    # ---- טבלת התרגילים: שורה לכל תרגיל (שם מהרשימה או הקלדה, התחלה, סיום, יעדים) ----
    from tkinter import ttk

    box = tk.Frame(frm, relief="groove", bd=1)
    box.grid(row=r, column=0, columnspan=3, sticky="nsew", **pad)
    frm.rowconfigure(r, weight=1)
    r += 1
    canvas = tk.Canvas(box, height=260, highlightthickness=0)
    sb = tk.Scrollbar(box, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas)
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    win = canvas.create_window((0, 0), window=inner, anchor="ne")
    canvas.bind("<Configure>", lambda e: (canvas.itemconfigure(win, width=e.width), canvas.coords(win, e.width, 0)))
    canvas.configure(yscrollcommand=sb.set)
    sb.pack(side="left", fill="y")
    canvas.pack(side="right", fill="both", expand=True)
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

    # עמודות משמאל לימין: מחיקה | יעדים | סיום | התחלה | שם  (כך שבעברית השם מימין)
    for c, (w, lbl) in enumerate([(3, ""), (30, "יעדים"), (8, "סיום"), (8, "התחלה"), (22, "שם התרגיל")]):
        tk.Label(inner, text=lbl, font=("Segoe UI", 9, "bold")).grid(row=0, column=c, padx=3, pady=(4, 2))
    inner.columnconfigure(1, weight=1)
    ex_rows = []

    def add_row(vals=("", "", "", "")):
        v = [tk.StringVar(value=x) for x in vals]
        rr = len(ex_rows) + 1
        wid = [
            tk.Button(inner, text="✕", width=2, fg="#a33"),
            tk.Entry(inner, textvariable=v[3], width=30),
            tk.Entry(inner, textvariable=v[2], width=8, justify="center"),
            tk.Entry(inner, textvariable=v[1], width=8, justify="center"),
            ttk.Combobox(inner, textvariable=v[0], values=EX_NAMES, width=20, justify="right"),
        ]
        item = {"vars": v, "w": wid}
        wid[0].configure(command=lambda it=item: del_row(it))
        for c, w_ in enumerate(wid):
            w_.grid(row=rr, column=c, padx=3, pady=2, sticky="we" if c == 1 else "")
        ex_rows.append(item)
        return item

    def del_row(item):
        for w_ in item["w"]:
            w_.destroy()
        ex_rows.remove(item)
        vals = [tuple(x.get() for x in it["vars"]) for it in ex_rows]
        set_rows(vals, keep_empty=True)

    def set_rows(vals, keep_empty=False):
        for it in list(ex_rows):
            for w_ in it["w"]:
                w_.destroy()
        ex_rows.clear()
        for x in vals:
            add_row(x)
        if not keep_empty or not ex_rows:
            for _ in range(max(0, 3 - len(ex_rows))):
                add_row()

    def get_rows():
        return [tuple(x.get().strip() for x in it["vars"]) for it in ex_rows]

    def add_and_focus():
        it = add_row()
        it["w"][4].focus_set()
        root.after(50, lambda: canvas.yview_moveto(1.0))

    tk.Button(frm, text="+ הוסף תרגיל", command=add_and_focus).grid(row=r, column=2, sticky="e", **pad)
    r += 1

    def ok():
        media = v_media.get().strip().strip('"')
        if need_media and not os.path.exists(media):
            messagebox.showerror("שגיאה", "לא נמצא קובץ הגיחה")
            return
        csv = v_csv.get().strip().strip('"')
        if csv and not os.path.exists(csv):
            messagebox.showerror("שגיאה", "לא נמצא קובץ ה-CSV")
            return
        off = parse_offset(v_off.get())
        if off is None:
            messagebox.showerror("שגיאה", "הפרש הזמן: מספר שניות, דקות:שניות, או המילה אוטומטי")
            return
        rows_, errs = get_rows(), []
        for i_, (nm, st, en, tg) in enumerate(rows_, 1):
            if not (nm or st or en or tg):
                continue
            if not re.fullmatch(_T, st):
                errs.append(f"תרגיל {i_} ({nm or 'בלי שם'}): זמן התחלה לא תקין - '{st}' (צריך דקות:שניות, למשל 05:30)")
            if en and not re.fullmatch(_T, en):
                errs.append(f"תרגיל {i_} ({nm or 'בלי שם'}): זמן סיום לא תקין - '{en}'")
        text = text_from_rows([x for x in rows_ if re.fullmatch(_T, x[1] or "")])
        _, errs2 = parse_exercises(text)
        errs += errs2
        if errs and not messagebox.askyesno("שורות לא תקינות", "\n".join(errs[:8]) + "\n\nלהמשיך בכל זאת?"):
            return
        res.update(media=media, csv=csv, offset=off, exercises_text=text)
        root.destroy()

    btns = tk.Frame(frm)
    btns.grid(row=r, column=0, columnspan=3, sticky="e", pady=(8, 0))
    tk.Button(btns, text="ביטול", width=10, command=root.destroy).pack(side="right", padx=4)
    tk.Button(btns, text="התחל", width=12, command=ok, default="active").pack(side="right", padx=4)

    fill_from(media_path)
    root.lift()
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))
    root.mainloop()
    if not res:
        return None
    if res["media"]:
        keep = load_session(res["media"])          # שומרים גם מידע שמור אחר (למשל תוצאת סנכרון קו האופק)
        keep.update({k: res[k] for k in ("csv", "offset", "exercises_text")})
        save_session(res["media"], keep)
    return res

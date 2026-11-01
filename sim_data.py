# -*- coding: utf-8 -*-
"""
קריאת קובץ הנתונים של המאמן (CSV) והמרה לסדרות זמן על ציר הזמן של הסרטון.

מבנה הקובץ שהתקבל (Elements_0.csv):
  Recording Time            - זמן מתחילת ההקלטה, במילישניות
  ...HOCAS...Collective/LateralStick/LongitudinalStick Position/Force - מיקום וכוח על הקולקטיב והסטיק
  Point_1_HOT               - גובה פני הקרקע מתחת למסוק (מטרים מעל פני הים)
  AltitudeAGL               - בקובץ הזה תמיד 0, לכן הגובה מעל הקרקע מחושב: AltitudeMSL - Point_1_HOT
  AltitudeMSL               - גובה מעל פני הים (מטרים)
  Heading / Pitch / Roll    - כיוון, עלרוד, הטיה (רדיאנים)
  Latitude / Longitude      - מיקום (רדיאנים)
  TrueAirSpeed              - מהירות אוויר (מטר לשנייה)
כל עמודה מתעדכנת בקצב משלה (שורות ריקות באמצע) - ממלאים קדימה את הערך האחרון.

היחידות הן הערכה לפי הערכים בקובץ (למשל 456 מ' = 1,500 רגל ו-46 מ'/שנ' = 90 קשר בתחילת הגיחה).
אם יתברר שהן אחרות - משנים כאן:
"""
import csv
import math

import numpy as np

ALT_TO_FT = 3.28084        # מטרים -> רגל
SPD_TO_KT = 1.943844       # מ'/שנ' -> קשר
ANGLE_IS_RAD = True
STEP = 0.5                 # רזולוציה על ציר הזמן (שניות)

# שם עמודה בקובץ (החלק האחרון אחרי הנקודה) -> מפתח פנימי
COLS = {
    "recording time": "time",
    "collectiveposition": "coll",
    "lateralstickposition": "lat_stick",
    "longitudinalstickposition": "lon_stick",
    "collectiveforce": "coll_f",
    "lateralstickforce": "lat_f",
    "longitudinalstickforce": "lon_f",
    "point_1_hot": "terrain",
    "altitudeagl": "agl",
    "altitudemsl": "msl",
    "heading": "hdg",
    "pitch": "pitch",
    "roll": "roll",
    "trueairspeed": "tas",
    "latitude": "lat",
    "longitude": "lon",
}


def _key(col):
    return COLS.get(col.strip().split(".")[-1].strip().lower())


def load_csv(path):
    """מחזיר dict: מפתח -> (זמנים בשניות, ערכים) רק לשורות שבהן לעמודה יש ערך."""
    with open(path, newline="", encoding="utf-8-sig", errors="ignore") as f:
        rd = csv.reader(f)
        header = next(rd)
        keys = [_key(c) for c in header]
        if "time" not in keys:
            raise ValueError("לא נמצאה עמודת זמן (Recording Time) בקובץ")
        ti = keys.index("time")
        cols = {k: ([], []) for k in keys if k and k != "time"}
        idx = [(i, k) for i, k in enumerate(keys) if k and k != "time"]
        for row in rd:
            if len(row) <= ti:
                continue
            try:
                t = float(row[ti])
            except ValueError:
                continue
            for i, k in idx:
                if i < len(row):
                    v = row[i].strip()
                    if v:
                        try:
                            cols[k][0].append(t)
                            cols[k][1].append(float(v))
                        except ValueError:
                            cols[k][0].pop()
    out = {k: (np.asarray(ts, float), np.asarray(vs, float)) for k, (ts, vs) in cols.items() if len(ts) > 1}
    # זמן במילישניות או בשניות
    tmax = max((v[0][-1] for v in out.values()), default=0)
    scale = 1000.0 if tmax > 20000 else 1.0
    out = {k: (t / scale, v) for k, (t, v) in out.items()}
    return out


def _hold(t_src, v_src, t_new):
    """ערך אחרון שידוע בכל זמן (מילוי קדימה)."""
    i = np.searchsorted(t_src, t_new, side="right") - 1
    res = v_src[np.clip(i, 0, len(v_src) - 1)].astype(float)
    res[(i < 0) | (t_new > t_src[-1] + 2.0)] = np.nan
    return res


def _activity(t_src, v_src, t_new, step):
    """כמה הסטיק זז בכל צעד זמן: סכום השינויים המוחלטים במיקום, ב-% מהטווח לשנייה."""
    if len(t_src) < 3:
        return np.full(len(t_new), np.nan)
    rng = np.nanpercentile(v_src, 99) - np.nanpercentile(v_src, 1)
    rng = rng if rng > 1e-6 else 1.0
    d = np.abs(np.diff(v_src)) / rng * 100
    tm = t_src[1:]
    edges = np.append(t_new - step / 2, t_new[-1] + step / 2)
    s, _ = np.histogram(tm, bins=edges, weights=d)
    s = s / step
    k = max(1, int(5 / step))                     # ממוצע נע של 5 שניות - שהגרף יהיה קריא
    s = np.convolve(s, np.ones(k) / k, mode="same")
    s[(t_new < t_src[0]) | (t_new > t_src[-1])] = np.nan
    return s


def build_series(raw, offset, dur):
    """סדרות על ציר הזמן של הסרטון (0..dur). offset = כמה שניות אחרי תחילת ה-CSV התחיל הסרטון."""
    t = np.arange(0, dur + STEP / 2, STEP)
    tc = t + offset                               # הזמן המקביל ב-CSV
    S = {"t": t}

    def get(k):
        return _hold(*raw[k], tc) if k in raw else None

    ang = (lambda x: np.degrees(x)) if ANGLE_IS_RAD else (lambda x: x)
    agl = get("agl")
    msl, terr = get("msl"), get("terrain")
    # "alt" = גובה ברומטרי (מעל פני הים) - זה מה שהחניך קורא ומדווח בקשר
    # "agl" = גובה מעל הקרקע
    if msl is not None:
        S["alt"] = msl * ALT_TO_FT
    if agl is not None and np.nanmax(np.abs(agl)) > 1:
        S["agl"] = agl * ALT_TO_FT
    elif msl is not None and terr is not None:
        S["agl"] = (msl - terr) * ALT_TO_FT
    if "alt" not in S and "agl" in S:
        S["alt"] = S["agl"]
    if "tas" in raw:
        S["spd"] = get("tas") * SPD_TO_KT
    if "roll" in raw:
        S["bank"] = ang(get("roll"))
    if "pitch" in raw:
        S["pitch"] = ang(get("pitch"))
    if "hdg" in raw:
        S["hdg"] = np.mod(ang(get("hdg")), 360)
    src = "msl" if "msl" in raw else ("agl" if "agl" in raw else None)
    if src:
        tt, vv = raw[src]
        vs = np.gradient(vv, tt) if len(tt) > 2 else np.zeros(len(tt))
        # החלקה של 3 שניות
        v = _hold(tt, vs, tc) * ALT_TO_FT * 60
        k = max(1, int(3 / STEP))
        S["vs"] = np.convolve(np.nan_to_num(v), np.ones(k) / k, mode="same")
        S["vs"][np.isnan(v)] = np.nan
    act = [(_activity(*raw[k], tc, STEP)) for k in ("lat_stick", "lon_stick") if k in raw]
    if act:
        S["stick"] = np.nansum(act, axis=0) if len(act) > 1 else act[0]
        S["stick"][np.all([np.isnan(a) for a in act], axis=0)] = np.nan
    if "coll" in raw:
        S["coll"] = _activity(*raw["coll"], tc, STEP)
    covered = ~np.isnan(S.get("alt", S.get("spd", np.full(len(t), np.nan))))
    S["covered"] = covered
    return S


def _ang_diff(a, b):
    return (a - b + 180) % 360 - 180


def window_stats(S, a, b, targets=None, tol=None):
    """סטטיסטיקה של נתוני הטיסה בין a ל-b (שניות בסרטון)."""
    tol = tol or {"alt": 50.0, "agl": 50.0, "spd": 5.0, "hdg": 10.0}
    m = (S["t"] >= a) & (S["t"] < b)
    out = {"cover": float(np.mean(S["covered"][m])) if m.any() else 0.0}
    if out["cover"] < 0.2:
        return out

    def col(k):
        if k not in S:
            return None
        x = S[k][m]
        x = x[~np.isnan(x)]
        return x if len(x) else None

    for k in ("alt", "agl", "spd"):
        x = col(k)
        if x is not None:
            out[k] = {"mean": float(x.mean()), "sd": float(x.std()), "min": float(x.min()), "max": float(x.max())}
    x = col("bank")
    if x is not None:
        out["bank"] = {"max": float(np.abs(x).max()), "mean": float(np.abs(x).mean())}
    x = col("vs")
    if x is not None:
        out["vs"] = {"min": float(x.min()), "max": float(x.max())}
    for k in ("stick", "coll"):
        x = col(k)
        if x is not None:
            out[k] = float(x.mean())
    for k, v in (targets or {}).items():
        x = col(k)
        if x is None:
            continue
        dev = _ang_diff(x, v) if k == "hdg" else x - v
        out.setdefault("tgt", {})[k] = {
            "target": v, "mean_dev": float(np.mean(dev)), "abs_dev": float(np.mean(np.abs(dev))),
            "in_tol": float(np.mean(np.abs(dev) <= tol[k])), "tol": tol[k],
        }
    return out


def round_list(x, nd=1):
    return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), nd) for v in x]


# ---------- חישוב אוטומטי של הפרש הזמן בין הסרטון ל-CSV ----------
# החניך מדווח בקול פרמטרים ("מהירות 80", "גובה 1100", "כיוון 270"). מנסים כל הפרש אפשרי
# ובוחרים את זה שבו הכי הרבה דיווחים מתאימים לנתונים במאמן באותו רגע.
import re as _re

_NUM = r"(\d{1,2},\d{3}|\d{1,4})"
_REP_RX = _re.compile(r"(מהירות|גובה|כיוון)\s*(?:עולה |יורד[ת]? |עולה ל|יורד ל)?(?:ל-?|של\s*)?" + _NUM)
_REP_KEY = {"מהירות": "spd", "גובה": "alt", "כיוון": "hdg"}


def extract_reports(utts):
    """[(זמן בסרטון, סוג, ערך)] מתוך המשפטים של החניך."""
    out = []
    for u in utts:
        for m in _REP_RX.finditer(u["text"]):
            k, v = _REP_KEY[m.group(1)], float(m.group(2).replace(",", ""))
            if (k == "spd" and not 20 <= v <= 180) or (k == "alt" and v < 50) or (k == "hdg" and v > 360):
                continue
            out.append((u["s"], k, v))
    return out


def estimate_offset(raw, reports, max_shift=1800.0, step=0.5):
    """מחזיר dict: offset (שניות), score (חלק הדיווחים שמתאימים), base (ממוצע בהפרשים אחרים),
    n (מספר דיווחים), ok (האם התוצאה אמינה)."""
    res = {"offset": 0.0, "score": 0.0, "base": 0.0, "n": len(reports), "ok": False}
    if len(reports) < 6:
        return res
    tend = max(v[0][-1] for v in raw.values())
    T = np.arange(0, tend + step, step)
    ser = {}
    if "tas" in raw:
        ser["spd"] = [_hold(*raw["tas"], T) * SPD_TO_KT]
    alts = []
    if "msl" in raw:
        alts.append(_hold(*raw["msl"], T) * ALT_TO_FT)
        if "terrain" in raw:
            alts.append((_hold(*raw["msl"], T) - _hold(*raw["terrain"], T)) * ALT_TO_FT)
    if "agl" in raw and np.nanmax(np.abs(raw["agl"][1])) > 1:
        alts.append(_hold(*raw["agl"], T) * ALT_TO_FT)
    if alts:
        ser["alt"] = alts
    if "hdg" in raw:
        h = raw["hdg"][1]
        ser["hdg"] = [np.mod(np.degrees(_hold(raw["hdg"][0], h, T)) if ANGLE_IS_RAD else _hold(raw["hdg"][0], h, T), 360)]
    # לכל דיווח: באילו זמנים ב-CSV הערך מתאים (עם חלון של 4 שניות לפני הדיווח ו-2 אחרי)
    lo_k, hi_k = int(4 / step), int(2 / step)
    rows, times = [], []
    for t, k, v in reports:
        if k not in ser:
            continue
        ok = np.zeros(len(T), bool)
        for arr in ser[k]:
            if k == "spd":
                ok |= np.abs(arr - v) <= 6
            elif k == "hdg":
                ok |= np.abs((arr - v + 180) % 360 - 180) <= 15
            else:
                ok |= np.abs(arr - v) <= max(60.0, 0.08 * v)
        c = np.concatenate([[0], np.cumsum(ok)])
        i = np.arange(len(T))
        a, b = np.clip(i - lo_k, 0, len(T)), np.clip(i + hi_k + 1, 0, len(T))
        rows.append((c[b] - c[a]) > 0)
        times.append(t)
    if len(rows) < 6:
        return res
    M = np.array(rows)                      # דיווחים x זמנים ב-CSV
    times = np.array(times)
    offs = np.arange(-max_shift, max_shift + step, step)
    score = np.zeros(len(offs))
    for j, o in enumerate(offs):
        idx = np.round((times + o) / step).astype(int)
        inside = (idx >= 0) & (idx < len(T))
        if inside.sum() < 0.6 * len(times):
            continue
        score[j] = M[np.nonzero(inside)[0], idx[inside]].mean()
    best = float(score.max())
    top = offs[score >= best - 1e-9]
    off = float(np.median(top))
    far = np.abs(offs - off) > 30
    second = float(score[far].max()) if far.any() else 0.0
    base = float(np.median(score[score > 0])) if (score > 0).any() else 0.0
    res.update(offset=off, score=best, base=base, second=second, n=len(rows),
               ok=bool(len(rows) >= 8 and best >= base + 0.2 and best >= second + 0.08))
    return res


def window_dynamics(S, a, b, targets=None, tol=None):
    """איך הפרמטרים השתנו לאורך התרגיל: כניסה/יציאה, קצב נסיקה/הנמכה, פנייה, הגעה ליעד והחזקתו."""
    tol = tol or {"alt": 50.0, "agl": 50.0, "spd": 5.0, "hdg": 10.0}
    m = (S["t"] >= a) & (S["t"] < b)
    if m.sum() < 4 or np.mean(S["covered"][m]) < 0.2:
        return {}
    t = S["t"][m]
    out = {}

    def s(k):
        return S[k][m] if k in S else None

    for k in ("alt", "agl", "spd"):
        x = s(k)
        if x is not None and np.isfinite(x).sum() > 3:
            ok = np.isfinite(x)
            out[k + "_in"], out[k + "_out"] = float(x[ok][:4].mean()), float(x[ok][-4:].mean())
    x = s("vs")
    if x is not None and np.isfinite(x).any():
        xs = x[np.isfinite(x)]
        out["climb"] = float(xs[xs > 100].mean()) if (xs > 100).sum() > 4 else None
        out["descent"] = float(xs[xs < -100].mean()) if (xs < -100).sum() > 4 else None
    h = s("hdg")
    if h is not None and np.isfinite(h).sum() > 3:
        hh = np.unwrap(np.radians(h[np.isfinite(h)]))
        out["turn"] = float(np.degrees(hh[-1] - hh[0]))
        out["turn_total"] = float(np.degrees(np.abs(np.diff(hh)).sum()))
    bk = s("bank")
    if bk is not None:
        turning = np.abs(bk) > 10
        if turning.sum() > 8:
            out["bank_turn_mean"] = float(np.abs(bk[turning]).mean())
            out["bank_turn_sd"] = float(np.abs(bk[turning]).std())
            al = s("alt")
            if al is not None:
                seg = al[turning & np.isfinite(al)]
                if len(seg) > 4:
                    out["alt_change_in_turns"] = float(seg.max() - seg.min())
        out["turning_pct"] = float(turning.mean())
    for k, v in (targets or {}).items():
        x = s(k)
        if x is None:
            continue
        dev = ((x - v + 180) % 360 - 180) if k == "hdg" else x - v
        inside = np.abs(dev) <= tol.get(k, 50)
        if inside.any():
            i0 = int(np.argmax(inside))
            after = inside[i0:]
            out.setdefault("tgt_dyn", {})[k] = {
                "reach_s": float(t[i0] - a), "held_after": float(after.mean()),
                "worst_after": float(np.nanmax(np.abs(dev[i0:]))) if len(dev[i0:]) else None,
            }
        else:
            out.setdefault("tgt_dyn", {})[k] = {"reach_s": None, "closest": float(np.nanmin(np.abs(dev)))}
    return out

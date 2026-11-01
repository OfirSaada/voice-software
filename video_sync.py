# -*- coding: utf-8 -*-
"""
סנכרון סרטון PFD מול ה-CSV של המאמן לפי קו האופק (גרסת Python של "מסנכרן טלמטריה").

1. מהסרטון נדגמים פריימים במרווחים קבועים. בכל פריים מסווגים פיקסלים של שמיים (כחול) ושל קרקע (חום)
   ומחפשים את המעבר ביניהם; דרך נקודות המעבר מועבר קו בשיטת RANSAC - השיפוע הוא זווית הגלגול,
   והמרחק מסמל המטוס הצהוב משקף את העלרוד.
2. רצף הגלגול (והעלרוד) מהסרטון מוחלק לאורך ה-CSV: לכל הפרש אפשרי מחושב מתאם פירסון על קטע החפיפה,
   והדירוג לפי Fisher z כדי שחפיפה קצרה ומקרית לא תנצח חפיפה ארוכה.
3. סביב השיא מעדנים ברזולוציה של מאית שנייה, ובודקים בכל שליש של הסרטון בנפרד שאין "סחיפה" בזמן.

התוצאה: offset בשניות, כך ש-  זמן ב-CSV = זמן בסרטון + offset  (אותה הגדרה כמו בשאר התוכנה).
"""
import math
import time

import numpy as np

DET_W = 640          # רוחב הפריים לזיהוי (כמו בגרסת הדפדפן)


# ---------- זיהוי קו האופק בפריים ----------
def detect(rgb):
    """rgb: מערך HxWx3 (uint8). מחזיר (m, c, yx, yy) או None.  y = m*x + c  הוא קו האופק."""
    H, W, _ = rgb.shape
    d = rgb.astype(np.int16)
    r, g, b = d[..., 0], d[..., 1], d[..., 2]
    mx, mn = d.max(axis=2), d.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    sky = (sat > 0.35) & (b > r + 50) & (b > g + 10) & (b > 110)
    gnd = (sat > 0.35) & ~sky & (r > b + 45) & (r >= g) & (g > b + 15) & (r > 90)
    yel = (r > 150) & (g > 150) & (b < 110)
    yn = int(yel.sum())
    if yn >= 10:
        ys_, xs_ = np.nonzero(yel)
        yx, yy = float(xs_.mean()), float(ys_.mean())
    else:
        yx = yy = float("nan")

    cols = np.arange(0, W, 2)
    S = sky[:, cols]
    G = gnd[:, cols]
    # קרקע במרחק 1..6 פיקסלים מתחת, ולפחות 4 מתוך 6 פיקסלים מעל הם שמיים
    first_g = np.full(S.shape, -1, int)
    for k in range(6, 0, -1):
        sh = np.zeros_like(G)
        sh[:-k] = G[k:]
        first_g[sh] = k
    cnt = np.zeros(S.shape, int)
    for k in range(0, 6):
        sh = np.zeros_like(S)
        sh[k:] = S[:H - k] if k else S
        cnt += sh
    ok = S & (first_g > 0) & (cnt >= 4)
    ok[:5] = False
    ok[H - 6:] = False
    has = ok.any(axis=0)
    if has.sum() < 15:
        return None
    ylast = H - 1 - np.argmax(ok[::-1], axis=0)          # המעבר הנמוך ביותר בכל עמודה (סריקה מלמטה)
    xs = cols[has].astype(float)
    yy0 = ylast[has]
    ys = (yy0 + (yy0 + first_g[yy0, np.nonzero(has)[0]])) / 2.0
    n = len(xs)
    tol = 2.5 * W / 642
    rng = np.random.default_rng(12345)
    best, bc = None, 0
    for _ in range(80):
        i, j = rng.integers(0, n, 2)
        if xs[i] == xs[j]:
            continue
        m = (ys[j] - ys[i]) / (xs[j] - xs[i])
        c = ys[i] - m * xs[i]
        inl = np.abs(ys - (m * xs + c)) < tol
        cnt_ = int(inl.sum())
        if cnt_ > bc:
            bc, best = cnt_, inl
    if bc < 15:
        return None
    m, c = np.polyfit(xs[best], ys[best], 1)
    return float(m), float(c), yx, yy


# ---------- קריאת הסרטון ----------
def pick_step(dur):
    return max(0.5, math.ceil(dur / 900 * 4) / 4)


def analyze_video(path, step=None, log=print):
    """מחזיר dict: T (זמנים), R (גלגול במעלות), P (עלרוד בפיקסלים), dur, step, n."""
    import av

    t0 = time.time()
    with av.open(path) as inp:
        vs = inp.streams.video[0]
        dur = float(vs.duration * vs.time_base) if vs.duration else float(inp.duration / av.time_base)
        step = step or pick_step(dur)
        w0, h0 = vs.codec_context.width, vs.codec_context.height
        W = min(DET_W, w0)
        H = int(round(h0 * W / w0))
        out = []

        def take(frame, t):
            rgb = frame.reformat(width=W, height=H, format="rgb24").to_ndarray()
            out.append((t, detect(rgb)))

        # ניסיון 1: רק פריימי מפתח (מהיר מאוד). מספיק אם יש פריים מפתח לפחות כל ~2 צעדים.
        vs.codec_context.skip_frame = "NONKEY"
        vs.thread_type = "AUTO"
        keys, t_first = [], None
        for fr in inp.decode(vs):
            if fr.time is None:
                continue
            if t_first is None:
                t_first = fr.time
            keys.append(fr.time - t_first)
        need = dur / step
        gaps = np.diff(keys) if len(keys) > 2 else np.array([1e9])
        use_keys = len(keys) >= 0.5 * need and float(np.median(gaps)) <= 2 * step
        inp.seek(0)
        vs.codec_context.skip_frame = "NONKEY" if use_keys else "DEFAULT"
        log(f"   סרטון: {dur / 60:.1f} דקות, {w0}x{h0}, דגימה כל {step:g} שנ׳ "
            f"({'פריימי מפתח' if use_keys else 'פענוח מלא - יכול לקחת זמן'})")
        nxt, t_first, last = 0.0, None, time.time()
        for fr in inp.decode(vs):
            if fr.time is None:
                continue
            if t_first is None:
                t_first = fr.time
            t = fr.time - t_first
            if t + 1e-3 < nxt:
                continue
            take(fr, t)
            while nxt <= t + 1e-6:
                nxt += step
            if time.time() - last > 5:
                last = time.time()
                log(f"\r      {t / 60:5.1f} / {dur / 60:.1f} דקות", end="")
        log("")
    T = [t for t, r in out if r]
    if not T:
        return {"dur": dur, "step": step, "n": len(out), "T": np.array([]), "R": np.array([]), "P": np.array([]),
                "secs": time.time() - t0}
    cx = np.nanmedian([r[2] for t, r in out if r])
    cy = np.nanmedian([r[3] for t, r in out if r])
    R, P = [], []
    for t, r in out:
        if not r:
            continue
        a = math.atan(r[0])
        R.append(-math.degrees(a))
        P.append(((r[0] * cx + r[1]) - cy) * math.cos(a) if np.isfinite(cx) else np.nan)
    return {"dur": dur, "step": step, "n": len(out), "T": np.array(T), "R": np.array(R), "P": np.array(P),
            "secs": time.time() - t0}


# ---------- חיפוש ההפרש ----------
def _grid(raw, dt=0.1):
    """roll/pitch במעלות על רשת אחידה של 10 הרץ."""
    from sim_data import ANGLE_IS_RAD, _hold

    t_r, v_r = raw["roll"]
    g0, g1 = t_r[0], t_r[-1]
    tg = np.arange(g0, g1, dt)
    k = 180 / math.pi if ANGLE_IS_RAD else 1.0
    gr = np.interp(tg, t_r, v_r) * k
    gp = np.interp(tg, *raw["pitch"]) * k if "pitch" in raw else None
    return tg, gr, gp


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-12 else float("nan")


def _fisher(r, n):
    r = max(-0.999, min(0.999, r))
    return math.atanh(r) * math.sqrt(max(1, n - 3))


def search(raw, V, dt=0.1):
    """מחזיר dict עם offset, rollR, score, runner, drift, confidence ('high'/'medium'/'low') ועוד."""
    if "roll" not in raw:
        raise ValueError("אין עמודת Roll ב-CSV")
    T, R, P = V["T"], V["R"], V["P"]
    if len(T) < 20:
        raise ValueError("קו האופק זוהה בפחות מ-20 פריימים - כנראה שהסרטון לא מציג מסך PFD")
    tg, gr, gp = _grid(raw, dt)
    g0, G = tg[0], len(tg)
    usep = gp is not None and np.isfinite(P).all() and P.std() > 0.5
    wR, wP = (0.7, 0.3) if usep else (1.0, 0.0)
    span, csv_span = T[-1] - T[0], (G - 1) * dt
    min_ov = min(max(20.0, 0.15 * span), 0.8 * csv_span, 0.8 * span)
    j_min = int(math.ceil((min_ov - T[-1]) / dt))
    j_max = int(math.floor((csv_span - min_ov - T[0]) / dt))
    if j_max < j_min:
        raise ValueError("קובץ ה-CSV קצר מדי לסנכרון")
    q = T / dt
    i0 = np.floor(q).astype(int)
    fr = q - i0
    js = np.arange(j_min, j_max + 1)
    score = np.full(len(js), np.nan)
    zz = np.full(len(js), -np.inf)
    for a in range(0, len(js), 2000):
        jj = js[a:a + 2000]
        for qi, j in enumerate(jj):
            g = i0 + j
            okk = (g >= 0) & (g < G - 1)
            n = int(okk.sum())
            if n < 15:
                continue
            Tk = T[okk]
            if Tk[-1] - Tk[0] < 0.9 * min_ov:
                continue
            Rk = R[okk]
            if Rk.var() < 1:
                continue
            gg, ff = g[okk], fr[okk]
            x = gr[gg] + (gr[gg + 1] - gr[gg]) * ff
            s = wR * _pearson(Rk, x)
            if wP:
                y = gp[gg] + (gp[gg + 1] - gp[gg]) * ff
                rp = _pearson(P[okk], y)
                s += wP * (rp if np.isfinite(rp) else 0)
            if not np.isfinite(s):
                continue
            score[a + qi] = s
            zz[a + qi] = _fisher(s, n)
    if not np.isfinite(zz).any():
        raise ValueError("לא נמצא קטע חפיפה שבו המסוק מתמרן")
    bq = int(np.argmax(zz))
    ex = int(round(10 / dt))
    far = np.abs(np.arange(len(js)) - bq) > ex
    rq = int(np.argmax(np.where(far, zz, -np.inf))) if far.any() and np.isfinite(zz[far]).any() else -1

    def corr_at(L, Tq, Yq, arr):
        x = np.interp(L + Tq, tg, arr, left=np.nan, right=np.nan)
        okk = np.isfinite(x) & np.isfinite(Yq)
        return _pearson(Yq[okk], x[okk]) if okk.sum() >= 10 else float("nan")

    def comb(L):
        r = corr_at(L, T, R, gr)
        return wR * r + wP * corr_at(L, T, P, gp) if wP else r

    b0 = g0 + js[bq] * dt
    cands = np.arange(b0 - 0.15, b0 + 0.1501, 0.01)
    vals = [comb(L) for L in cands]
    best = float(cands[int(np.nanargmax(vals))])
    # סחיפה: התאמה נפרדת לכל שליש של הסרטון
    x_all = np.interp(best + T, tg, gr, left=np.nan, right=np.nan)
    ov = np.isfinite(x_all)
    Tov, Rov = T[ov], R[ov]
    thirds = []
    if len(Tov) > 24:
        oA, oS = Tov[0], Tov[-1] - Tov[0]
        for k in range(3):
            m = (Tov >= oA + oS * k / 3) & (Tov <= oA + oS * (k + 1) / 3)
            if m.sum() < 8 or Rov[m].std() < 2:
                continue
            Ls = np.arange(best - 2, best + 2.001, 0.02)
            cs = [corr_at(L, Tov[m], Rov[m], gr) for L in Ls]
            thirds.append(float(Ls[int(np.nanargmax(cs))]))
    drift = (max(thirds) - min(thirds)) if len(thirds) >= 2 else float("nan")
    rollR = corr_at(best, T, R, gr)
    res = {"offset": best, "rollR": rollR, "score": float(score[bq]),
           "runner": float(score[rq]) if rq >= 0 else float("nan"), "drift": drift,
           "samples": int(ov.sum()), "frames": V["n"], "detected": len(T), "rollStd": float(Rov.std()) if len(Rov) else 0.0,
           "csv_start": float(tg[0]), "csv_end": float(tg[-1]), "dur": V["dur"], "used_pitch": bool(wP)}
    margin = res["score"] - (res["runner"] if np.isfinite(res["runner"]) else -1)
    if res["rollStd"] < 2:
        res["confidence"] = "low"
    elif rollR >= 0.9 and margin >= 0.2 and not (drift > 0.5):
        res["confidence"] = "high"
    elif rollR >= 0.75 and margin >= 0.1:
        res["confidence"] = "medium"
    else:
        res["confidence"] = "low"
    # לגרף הבדיקה בדף: גלגול מהסרטון מול ה-CSV אחרי ההזזה
    res["check"] = {"t": [round(float(x), 2) for x in T],
                    "video": [round(float(x), 1) for x in R],
                    "csv": [None if not np.isfinite(x) else round(float(x), 1) for x in x_all]}
    return res


CONF_TXT = {"high": "התאמה ודאית", "medium": "התאמה סבירה", "low": "התאמה חלשה"}

# -*- coding: utf-8 -*-
"""
גרפים של החניך: עוצמה, גובה צליל (F0) וקצב דיבור לאורך הגיחה.

מחושב ישירות מההקלטה, כל 0.05 שנייה, רק בזמנים שבהם החניך מדבר
(לפי הזמנים המדויקים של כל משפט ב-transcription_result.html).
כשהחניך לא מדבר - הגרף יורד ל-0.

  - בלי בחירה: כל הגיחה.
  - בחירת טווח: "מדקה" / "עד דקה" בשורה למעלה, או גרירה עם העכבר על אחד הגרפים.
  - מעבר עם העכבר: זמן, ערך, ומה החניך אמר באותו רגע. לחיצה מנגנת את הרגע בהקלטה.

הרצה: אוטומטית בסוף transcribe_v3.py, או ידנית (מתמלול שכבר נעשה).
הכל מקומי - בלי אינטרנט.
"""
import html
import json
import os
import re
import sys
import time
import urllib.parse
import webbrowser

import numpy as np
from scipy.signal import medfilt

HERE = os.path.dirname(os.path.abspath(__file__))
# אפשר לתת כארגומנט את האקסל או את ה-HTML של התמלול - בכל מקרה משתמשים ב-HTML שבאותה תיקייה
# --auto = הופעל מתוך transcribe_v3.py (ההגדרות כבר נבחרו שם, לא פותחים חלון)
AUTO = "--auto" in sys.argv
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
_arg = _args[0] if _args else os.path.join(HERE, "transcription_result.html")
sys.path.insert(0, HERE)
BASE_DIR = os.path.dirname(os.path.abspath(_arg))
TRANSCRIPT_HTML = os.path.join(BASE_DIR, "transcription_result.html")
OUT = os.path.join(BASE_DIR, "trainee_charts.html")
TRAINEE = "חניך"

HOP_SEC = 0.05          # רזולוציה: נקודה כל 0.05 שנייה
SR = 16000
F0_MIN, F0_MAX = 70, 400
F0_BRIDGE_SEC = 0.25    # חורים קצרים בתוך משפט (עיצורים) בגובה הצליל - ממולאים

if not os.path.exists(TRANSCRIPT_HTML):
    print(f"לא נמצא {TRANSCRIPT_HTML}\nקודם צריך להריץ את transcribe_v3.py")
    raise SystemExit(1)

with open(TRANSCRIPT_HTML, encoding="utf-8", errors="ignore") as f:
    page_src = f.read()

# ---- 1. זמני המשפטים של החניך ----
blocks = re.findall(
    r'data-start="([\d.]+)" data-end="([\d.]+)".*?class="speaker-name">(.*?)</span>.*?class="text-content">(.*?)</div>',
    page_src, re.S,
)
ALL = sorted(
    [(float(a), float(b), html.unescape(spk).strip(), html.unescape(t).strip()) for a, b, spk, t in blocks
     if float(b) > float(a)],
    key=lambda x: x[0],
)
utts = [
    {"s": float(a), "e": float(b), "text": html.unescape(t).strip()}
    for a, b, spk, t in blocks
    if html.unescape(spk).strip() == TRAINEE and float(b) > float(a)
]
if not utts:
    print("לא נמצאו משפטים של החניך בקובץ התמלול.")
    raise SystemExit(1)

m = re.search(r'<source src="([^"]+)"', page_src)
audio_src = m.group(1) if m else ""
audio_path = urllib.parse.unquote(re.sub(r"^file:/+", "", audio_src)) if audio_src else ""
if audio_path and not os.path.exists(audio_path) and os.path.exists("/" + audio_path):
    audio_path = "/" + audio_path
if not audio_path or not os.path.exists(audio_path):
    print(f"לא נמצאה ההקלטה: {audio_path!r}")
    raise SystemExit(1)

title_name = os.path.splitext(os.path.basename(audio_path))[0]
MEDIA_PATH = audio_path

# ---- הגדרות הגיחה: קובץ נתוני מאמן (CSV), הפרש זמן, תרגילים ----
try:
    from session_dialog import load_session, parse_exercises, ask_session
except ImportError:
    load_session = parse_exercises = ask_session = None
SESSION = load_session(MEDIA_PATH) if load_session else {}
if ask_session and not AUTO:
    try:
        _r = ask_session(MEDIA_PATH, need_media=False, title="גרפי חניך - נתוני מאמן ותרגילים")
        if _r is None:
            print("בוטל.")
            raise SystemExit(0)
        SESSION = _r
    except SystemExit:
        raise
    except Exception as _e:  # noqa: BLE001
        print(f"[אזהרה] חלון ההגדרות לא נפתח ({_e}) - משתמש בהגדרות השמורות")
# סרטון: משתמשים בשמע שחולץ ממנו (קובץ _audio.wav שליד הסרטון)
_wav = os.path.splitext(audio_path)[0] + "_audio.wav"
if audio_path.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".m4v")) and os.path.exists(_wav):
    audio_path = _wav

# ---- 2. טעינת ההקלטה ----
print(f"טוען הקלטה: {audio_path}")
t0 = time.time()
try:
    from faster_whisper import decode_audio

    y = decode_audio(audio_path, sampling_rate=SR)
except Exception:  # noqa: BLE001
    import librosa

    y, _ = librosa.load(audio_path, sr=SR, mono=True)
y = np.asarray(y, dtype=np.float32)
dur = len(y) / SR

EXERCISES, _ex_err = (parse_exercises(SESSION.get("exercises_text", ""), dur) if parse_exercises else ([], []))
for _e in _ex_err:
    print(f"   [תרגילים] {_e}")
EXERCISES = [e for e in EXERCISES if e["start"] < dur]
for _e in EXERCISES:
    _e["end"] = min(_e["end"] or dur, dur)
if EXERCISES:
    print(f"   תרגילים: {len(EXERCISES)}")

FL = None          # סדרות נתוני הטיסה על ציר הזמן של הסרטון
SYNC_NOTE = ""
VSYNC = None       # תוצאת הסנכרון לפי קו האופק (לגרף הבדיקה)


def _hms(x):
    x = abs(x)
    return f"{int(x) // 60}:{x % 60:04.1f}"


def auto_offset(raw):
    """הפרש הזמן בין הסרטון ל-CSV: (1) לפי קו האופק במסך ה-PFD בסרטון, (2) לפי הדיווחים של החניך בקשר.
    מחזיר (offset, הסבר, תוצאת-סרטון-או-None)."""
    import sim_data

    vres = None
    is_video = MEDIA_PATH.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".m4v"))
    if is_video and "roll" in raw:
        # תוצאה שמורה (הניתוח של הסרטון לוקח זמן) - תקפה כל עוד הסרטון וה-CSV לא השתנו
        key = {"video": [os.path.getsize(MEDIA_PATH), int(os.path.getmtime(MEDIA_PATH))],
               "csv": [SESSION["csv"], os.path.getsize(SESSION["csv"])]}
        try:
            from session_dialog import load_session as _ls, save_session as _ss
            stored = _ls(MEDIA_PATH)
        except Exception:  # noqa: BLE001
            stored, _ss = {}, None
        vs_ = stored.get("video_sync")
        if vs_ and vs_.get("key") == key:
            vres = vs_["res"]
            print(f"   סנכרון לפי קו האופק (שמור): {vres['offset']:+.2f} שנ׳, {vres.get('confidence')}")
        else:
            try:
                import video_sync

                print("   מסנכרן לפי קו האופק במסך ה-PFD בסרטון...")
                V = video_sync.analyze_video(MEDIA_PATH)
                print(f"   קו האופק זוהה ב-{len(V['T'])} מתוך {V['n']} פריימים ({V['secs']:.0f} שנ׳)")
                vres = video_sync.search(raw, V)
                print(f"   תוצאה: {vres['offset']:+.2f} שנ׳, מתאם גלגול {vres['rollR']:.2f}, {vres['confidence']}")
            except Exception as e:  # noqa: BLE001
                print(f"   [סנכרון לפי קו האופק] לא הצליח: {e}")
                vres = {"error": str(e)}
            if _ss:
                stored["video_sync"] = {"key": key, "res": vres}
                _ss(MEDIA_PATH, stored)
        if vres and "error" in vres:
            vres = None
    sp = sim_data.estimate_offset(raw, sim_data.extract_reports(utts))
    sp_txt = (f"לפי הדיווחים של החניך בקשר: {sp['offset']:+.1f} שנ׳ ({100 * sp['score']:.0f}% מ-{sp['n']} הדיווחים מתאימים)"
              if sp["ok"] else "")
    if vres and vres["confidence"] in ("high", "medium"):
        off = vres["offset"]
        conf = {"high": "התאמה ודאית", "medium": "התאמה סבירה"}[vres["confidence"]]
        note = (f"הפרש הזמן חושב לפי קו האופק בסרטון ({conf}, מתאם גלגול {vres['rollR']:.2f}): הסרטון התחיל "
                f"{_hms(off)} דקות {'אחרי' if off >= 0 else 'לפני'} תחילת קובץ הנתונים "
                f"(זמן ב-CSV = זמן בסרטון {'+' if off >= 0 else '−'} {abs(off):.2f} שנ׳).")
        if sp["ok"]:
            d = abs(sp["offset"] - off)
            note += (f" אומת גם לפי הדיווחים של החניך בקשר (הפרש של {d:.1f} שנ׳ בין השיטות)." if d <= 5 else
                     f" שים לב: לפי הדיווחים של החניך בקשר יצא הפרש אחר ({sp['offset']:+.1f} שנ׳) - כדאי לבדוק.")
        return off, note, vres
    if sp["ok"]:
        note = (f"הפרש הזמן חושב לפי הדיווחים של החניך בקשר: הסרטון התחיל {_hms(sp['offset'])} דקות "
                f"{'אחרי' if sp['offset'] >= 0 else 'לפני'} תחילת קובץ הנתונים "
                f"({100 * sp['score']:.0f}% מ-{sp['n']} הדיווחים - גובה, מהירות, כיוון - מתאימים, "
                f"לעומת {100 * sp['base']:.0f}% בהפרשים אחרים).")
        if vres:
            note += f" הסנכרון לפי קו האופק בסרטון יצא חלש ({vres['offset']:+.1f} שנ׳, מתאם {vres['rollR']:.2f}) ולא שימש."
        return sp["offset"], note, vres
    note = ("לא הצלחתי לחשב את הפרש הזמן אוטומטית - הונח שהסרטון וה-CSV מתחילים יחד. "
            "אפשר לכתוב את ההפרש ידנית בחלון ההגדרות.")
    if vres:
        note += f" (קו האופק: {vres['offset']:+.1f} שנ׳ בהתאמה חלשה; {sp_txt or 'אין מספיק דיווחים בקשר'})"
    return 0.0, note, vres
if SESSION.get("csv"):
    try:
        import sim_data

        print(f"טוען נתוני מאמן: {SESSION['csv']}")
        _raw = sim_data.load_csv(SESSION["csv"])
        _off = SESSION.get("offset", "auto")
        if _off in (None, "", "auto"):
            _off, SYNC_NOTE, VSYNC = auto_offset(_raw)
            print("   " + SYNC_NOTE)
        else:
            _off = float(_off)
            SYNC_NOTE = f"הפרש זמן (הוזן ידנית): {_off:+.0f} שניות."
        FL = sim_data.build_series(_raw, float(_off), dur)
        _cov = float(np.mean(FL["covered"]))
        print(f"   {len(_raw)} עמודות, כיסוי של הסרטון: {100 * _cov:.0f}%")
        if _cov < 0.05:
            print("   [אזהרה] הנתונים לא חופפים לסרטון - כדאי לבדוק את הפרש הזמן")
            FL = None
    except Exception as _e:  # noqa: BLE001
        print(f"   [אזהרה] קריאת ה-CSV נכשלה: {_e}")
        FL = None

import librosa  # noqa: E402

hop = int(HOP_SEC * SR)

# ---- 3. חישוב לכל משפט של החניך ----
print(f"מחשב עוצמה וגובה צליל ל-{len(utts)} משפטים של החניך...")
T, VOL, F0, RATE = [], [], [], []
for u in utts:
    a, b = int(u["s"] * SR), int(u["e"] * SR)
    seg = y[a:b]
    if len(seg) < 2048:
        seg = np.pad(seg, (0, 2048 - len(seg)))
    n_words = len(re.findall(r"\S+", u["text"]))
    rate = n_words / max(u["e"] - u["s"], 0.3)
    rms = librosa.feature.rms(y=seg, frame_length=1024, hop_length=hop, center=True)[0] * 100
    # YIN (ולא pYIN): בקשר רדיו pYIN מסמן כמעט הכל כ"לא קולי". YIN נותן ערך לכל פריים,
    # ואנחנו משאירים רק פריימים שיש בהם קול (עוצמה מעל סף יחסי למשפט),
    # ומחליקים בחציון כדי להוריד קפיצות אוקטבה.
    f0 = librosa.yin(seg, fmin=F0_MIN, fmax=F0_MAX, sr=SR, frame_length=2048, hop_length=hop, center=True)
    n0 = min(len(f0), len(rms))
    f0, r = f0[:n0].astype(float), rms[:n0]
    thr = max(0.25 * np.percentile(r, 90), 0.05)
    f0 = medfilt(f0, 5) if len(f0) >= 5 else f0
    f0[(r < thr) | (f0 <= F0_MIN * 1.02) | (f0 >= F0_MAX * 0.98)] = np.nan
    # גישור על חורים קצרים בתוך המשפט
    bridge = int(F0_BRIDGE_SEC / HOP_SEC)
    idx = np.where(~np.isnan(f0))[0]
    for i0, i1 in zip(idx[:-1], idx[1:]):
        if 1 < i1 - i0 <= bridge:
            f0[i0 + 1:i1] = np.linspace(f0[i0], f0[i1], i1 - i0 + 1)[1:-1]
    n = min(len(rms), len(f0))
    nin = max(1, min(n, int((u["e"] - u["s"]) / HOP_SEC) + 1))
    fv = f0[:nin][~np.isnan(f0[:nin])]
    u["f0m"] = float(np.mean(fv)) if len(fv) >= 3 else None
    u["f0s"] = float(np.std(fv)) if len(fv) >= 5 else None
    u["volm"] = float(np.mean(rms[:nin]))
    u["rate"] = rate
    for k in range(n):
        t = u["s"] + k * HOP_SEC
        if t > u["e"]:
            break
        T.append(round(t, 2))
        VOL.append(round(float(rms[k]), 2))
        F0.append(None if np.isnan(f0[k]) else round(float(f0[k]), 1))
        RATE.append(round(rate, 2))
print(f"   {len(T)} נקודות, {time.time() - t0:.0f} שניות")

# ---- 4. מדד עומס יחסי: כמה כל משפט חורג מה"רגיל" של אותו חניך בגיחה הזו ----
# z חסין (חציון ו-MAD) לכל רכיב: גובה צליל, שונות גובה צליל, עוצמה, קצב. המדד = ממוצע הרכיבים.
def robust_z(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if len(arr) < 10:
        return [None] * len(vals)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med)) * 1.4826 or (np.std(arr) or 1.0)
    return [None if v is None else float(np.clip((v - med) / mad, -4, 4)) for v in vals]


Z = {k: robust_z([u[k] for u in utts]) for k in ("f0m", "f0s", "volm", "rate")}
for i, u in enumerate(utts):
    parts = [Z[k][i] for k in Z if Z[k][i] is not None]
    u["idx"] = round(float(np.mean(parts)), 2) if parts else None
    u["z"] = [None if Z[k][i] is None else round(Z[k][i], 2) for k in ("f0m", "f0s", "volm", "rate")]

# ---- 5. זמן תגובה: מקריאה/שאלה של מדריכה או ממיין ועד שהחניך עונה ----
OTHERS = ("מדריכה", "ממיין")
RESP_MAX = 10.0      # אין תשובה תוך 10 שניות = לא נענה
QUESTION_RX = re.compile(
    r"\?|^(?:מה|למה|איך|איזה|איזו|איפה|כמה|האם|מתי|אתה|קיבלת|שומע)\b|\b(?:האם|קיבלת|שומע|אתה רוצה|אתה רואה)\b"
)
resp = []
i = 0
while i < len(ALL):
    if ALL[i][2] not in OTHERS:
        i += 1
        continue
    j = i          # תור = רצף משפטים של מדריכה/ממיין בלי שהחניך מדבר באמצע
    while j + 1 < len(ALL) and ALL[j + 1][2] in OTHERS and ALL[j + 1][0] - ALL[j][1] < 3.0:
        j += 1
    turn = ALL[i:j + 1]
    p_end, last = turn[-1][1], turn[-1]
    nxt = next((x for x in ALL[j + 1:] if x[2] == TRAINEE), None)
    other_between = nxt is not None and any(
        x[2] in OTHERS for x in ALL[j + 1:] if x[0] < nxt[0])
    is_q = bool(QUESTION_RX.search(last[3])) or any(QUESTION_RX.search(x[3]) for x in turn[-2:])
    lat = None
    if nxt is not None and not other_between and nxt[0] - p_end <= RESP_MAX:
        lat = round(max(0.0, nxt[0] - p_end), 2)
    resp.append({
        "t": round(p_end, 2), "who": last[2], "q": is_q, "lat": lat,
        "p": " ".join(x[3] for x in turn[-2:])[:160],
        "r": (nxt[3][:120] if lat is not None else ""),
    })
    i = j + 1

# ---- 6. אירועים: התערבות / העברת שליטה / תקלה-תרחיש ----
EVENT_RX = [
    ("control", "העברת שליטה", re.compile(r"אתה לוקח|אני לוקח|קח שליטה|לוקח שליטה|תן לי שליטה|מסור שליטה")),
    ("interv", "התערבות", re.compile(r"מתערב|התערבתי|מתערבת")),
    ("fault", "תקלה / תרחיש", re.compile(r"תקלה|תקלת|כשל|נתקלנו|הותקלנו")),
]
events = []
for a0, b0, spk, txt in ALL:
    for key, _, rx in EVENT_RX:
        if rx.search(txt):
            events.append({"t": round(a0, 2), "type": key, "who": spk, "text": txt[:140]})
            break

print(f"   תורות של מדריכה/ממיין: {len(resp)}, אירועים: {len(events)}")


# ---- 7. סיכום החניך בגיחה (נוצר אוטומטית מהנתונים, בלי AI ובלי רשת) ----
def mmss(t):
    t = max(0, int(round(t)))
    return f"{t // 60:02d}:{t % 60:02d}"


def ts(t):          # זמן לחיץ בדף
    return f'<a href="#" class="ts" data-t="{t:.1f}">{mmss(t)}</a>'


def sg(x):          # מספר עם סימן, בכיוון שמאל-ימין (שלא יתהפך בעברית)
    return f'<bdi dir="ltr">{x:+.1f}</bdi>'


def rng(a, b):      # טווח זמנים "12:00–12:30"
    return f'<bdi dir="ltr">{ts(a)}–{mmss(b)}</bdi>'


def q(txt, n=70):   # ציטוט קצר
    txt = (txt or "").strip()
    return "«" + html.escape(txt if len(txt) <= n else txt[:n].rsplit(" ", 1)[0] + "…") + "»"


def nwords(txt):
    return len(re.findall(r"\S+", txt or ""))


sections = []
talk = sum(u["e"] - u["s"] for u in utts)
words = sum(nwords(u["text"]) for u in utts)

# כללי
items = [
    f"הגיחה נמשכה {mmss(dur)} דקות. החניך דיבר בסך הכול {talk / 60:.1f} דקות "
    f"({100 * talk / dur:.0f}% מהזמן), ב-{len(utts)} משפטים וכ-{words:,} מילים "
    f"- קצב ממוצע של {words / max(talk, 1):.1f} מילים לשנייה בזמן שהוא מדבר."
]
thirds = []
for k in range(3):
    a3, b3 = k * dur / 3, (k + 1) * dur / 3
    w3 = sum(nwords(u["text"]) for u in utts if a3 <= u["s"] < b3)
    thirds.append(w3 / max((b3 - a3) / 60, 1e-6))
names3 = ["בשליש הראשון", "בשליש האמצעי", "בשליש האחרון"]
hi3, lo3 = int(np.argmax(thirds)), int(np.argmin(thirds))
if max(thirds) > 1.25 * max(min(thirds), 1e-6):
    items.append(f"הוא דיבר הכי הרבה {names3[hi3]} של הגיחה (כ-{thirds[hi3]:.0f} מילים לדקה) "
                 f"והכי מעט {names3[lo3]} (כ-{thirds[lo3]:.0f}).")
else:
    items.append(f"כמות הדיבור שלו הייתה דומה לאורך כל הגיחה (כ-{np.mean(thirds):.0f} מילים לדקה).")
# קטע צפוף / שתיקה ארוכה
dens_w = np.zeros(int(dur) + 2)
for u in utts:
    a0, b0 = int(u["s"]), max(int(u["s"]) + 1, int(u["e"]) + 1)
    dens_w[a0:b0] += nwords(u["text"]) / (b0 - a0)
if len(dens_w) > 31:
    roll = np.convolve(dens_w, np.ones(30), mode="valid")
    i_max = int(np.argmax(roll))
    items.append(f"קטע הדיבור הצפוף ביותר שלו: {rng(i_max, i_max + 30)} "
                 f"({roll[i_max]:.0f} מילים ב-30 שניות).")
gaps = [(utts[i]["e"], utts[i + 1]["s"]) for i in range(len(utts) - 1)]
if gaps:
    g0, g1 = max(gaps, key=lambda g: g[1] - g[0])
    if g1 - g0 > 60:
        items.append(f"השתיקה הארוכה ביותר שלו: {(g1 - g0) / 60:.1f} דקות ({rng(g0, g1)}).")
sections.append(("כללי", items))

# קול ועומס
items = []
f0_all = [v for v in F0 if v is not None]
vol_all = VOL
if f0_all:
    items.append(f"גובה הצליל הרגיל שלו בגיחה הזו: כ-{np.median(f0_all):.0f} Hz. "
                 f"עוצמה ממוצעת בזמן דיבור: {np.mean(vol_all):.1f} (יחסית).")
ZN = ["גובה צליל", "שונות בגובה הצליל", "עוצמה", "קצב"]
peaks = []
# רק משפטים של שנייה ומעלה ו-3 מילים ומעלה ("אוקיי" קצר נותן מדידת קול לא אמינה)
for u in sorted([u for u in utts if u["idx"] is not None and u["e"] - u["s"] >= 1.0 and nwords(u["text"]) >= 3],
                key=lambda u: -u["idx"]):
    if u["idx"] < 1.0 or len(peaks) >= 3:
        break
    if all(abs(u["s"] - p["s"]) > 60 for p in peaks):
        peaks.append(u)
if peaks:
    lines = []
    for u in sorted(peaks, key=lambda u: u["s"]):
        z = u["z"] or []
        main = max(((ZN[i], z[i]) for i in range(len(z)) if z[i] is not None), key=lambda x: x[1], default=None)
        why = f", בעיקר {main[0]}" if main and main[1] > 0.5 else ""
        lines.append(f"{ts(u['s'])} (מדד {sg(u['idx'])}{why}) {q(u['text'])}")
    items.append("הרגעים שבהם הקול שלו חרג הכי הרבה מהרגיל שלו (מדד עומס יחסי): " + "; ".join(lines) + ".")
else:
    items.append("לא היו רגעים בולטים שבהם הקול שלו חרג משמעותית מהרגיל שלו.")
half = dur / 2


def mean_idx(a0, b0):
    us = [u for u in utts if u["idx"] is not None and a0 <= u["s"] < b0]
    w = sum(u["e"] - u["s"] for u in us)
    return sum(u["idx"] * (u["e"] - u["s"]) for u in us) / w if w else None


i1, i2 = mean_idx(0, half), mean_idx(half, dur)
if i1 is not None and i2 is not None:
    if i2 - i1 > 0.3:
        items.append(f"במחצית השנייה של הגיחה מדד העומס שלו היה גבוה יותר מבמחצית הראשונה "
                     f"({sg(i2)} לעומת {sg(i1)}).")
    elif i1 - i2 > 0.3:
        items.append(f"במחצית השנייה של הגיחה הוא נשמע רגוע יותר מבמחצית הראשונה "
                     f"(מדד {sg(i2)} לעומת {sg(i1)}).")
    else:
        items.append(f"מדד העומס שלו היה יציב בין שתי מחציות הגיחה ({sg(i1)} ו-{sg(i2)}).")
sections.append(("קול ועומס", items))

# תגובה לשאלות
items = []
qs = [r for r in resp if r["q"]]
ans = [r for r in qs if r["lat"] is not None]
if qs:
    med = float(np.median([r["lat"] for r in ans])) if ans else None
    t = f"המדריכה והממיין פנו אליו ב-{len(qs)} שאלות וקריאות, והוא ענה על {len(ans)} מהן"
    t += f" (זמן תגובה חציוני: {med:.1f} שנ׳)." if med is not None else "."
    items.append(t)
    miss = [r for r in qs if r["lat"] is None]
    if miss:
        items.append("לא נמצאה תשובה שלו ל: " + "; ".join(
            f"{ts(r['t'])} {html.escape(r['who'])}: {q(r['p'], 60)}" for r in miss[:3]) + ".")
    slow = sorted([r for r in ans if r["lat"] >= 2.0], key=lambda r: -r["lat"])[:3]
    if slow:
        items.append("התגובות האיטיות ביותר שלו: " + "; ".join(
            f"{ts(r['t'])} - {r['lat']:.1f} שנ׳ אחרי {q(r['p'], 50)}" for r in sorted(slow, key=lambda r: r["t"])) + ".")
    elif ans:
        items.append("לא היו תגובות איטיות (כולן תוך פחות מ-2 שניות).")
else:
    items.append("לא זוהו שאלות או קריאות של המדריכה או הממיין אליו.")
sections.append(("תגובה לשאלות וקריאות", items))

# התערבויות ושליטה
items = []
by = {k: [e for e in events if e["type"] == k] for k, _, _ in EVENT_RX}
if by["control"]:
    items.append(f"{len(by['control'])} אמירות של העברת שליטה, בזמנים: " +
                 ", ".join(ts(e["t"]) for e in by["control"][:8]) + ".")
    after = [mean_idx(e["t"], e["t"] + 60) for e in by["control"]]
    after = [x for x in after if x is not None]
    allm = mean_idx(0, dur)
    if after and allm is not None:
        a_m = float(np.mean(after))
        if a_m - allm > 0.3:
            items.append(f"בדקה שאחרי העברת שליטה מדד העומס שלו היה גבוה מהרגיל ({sg(a_m)} לעומת {sg(allm)} בממוצע).")
        else:
            items.append(f"בדקה שאחרי העברת שליטה מדד העומס שלו לא היה שונה מהותית מהממוצע ({sg(a_m)} לעומת {sg(allm)}).")
else:
    items.append("לא זוהתה העברת שליטה בגיחה.")
if by["interv"]:
    items.append(f"{len(by['interv'])} התערבויות של המדריכה/ממיין: " + ", ".join(ts(e["t"]) for e in by["interv"][:8]) + ".")
if by["fault"]:
    items.append(f"{len(by['fault'])} אזכורים של תקלות או תרחישים: " + ", ".join(ts(e["t"]) for e in by["fault"][:8]) + ".")
sections.append(("שליטה, התערבויות ותקלות", items))

sections.append(("חשוב לדעת", [
    "הסיכום נוצר אוטומטית מהתמלול ומניתוח הקול, ולא מחליף את ההערכה של המדריכה והממיין.",
    "הוא תלוי באיכות התמלול ובזיהוי הדוברים - משפט ששויך לדובר הלא נכון ישפיע על המספרים.",
    "מדדי הקול והעומס יחסיים לחניך עצמו בגיחה הזו, ולא מתאימים להשוואה בין חניכים.",
]))

# ---- 8. הערכת החניך לפי נושאים: ראיות מהקוד + ניסוח ע"י מודל שפה מקומי (בלי רשת) ----
# הקוד אוסף ראיות לכל נושא (עם זמנים וציטוטים), והמודל המקומי (DictaLM 2.0, עברית)
# מקבל כל נושא בנפרד ומנסח הערכה קצרה. בלי המודל - מוצגות הראיות בלבד.
USE_LOCAL_LLM = True
LLM_PATH = os.path.join(HERE, "models", "llm", "dictalm2.0-instruct.Q4_K_M.gguf")

CORR_RX = re.compile(r"^(?:לא[,. ]|רגע|תקבע|תשים|תוריד|תעלה|תמשיך|תפנה|תקפיד|תגדיר|אל ת)|שים לב|תקפיד|תבקר|תצמיד|"
                     r"תוריד|תגדיר|\bלמה\b|אתה זוכר|איך אתה")
PROBLEM_RX = re.compile(r"לא הצלחתי|טעיתי|פספסתי|חרגתי|לא טוב|לא מדויק|נפגע|מקשה|קשה לי|הסתבכתי|איבדתי|"
                        r"לא שמעתי|לא הבנתי|ירדה במהירות|עודף")
DEBRIEF_RX = re.compile(r"תחקיר|תחקור|הנקודה|הייתי צריך|לא הצלחתי|אני צריך ל|צריך לשים לב|לשפר|בפעם הבאה")


def tt(t):   # זמן כטקסט רגיל למודל
    return mmss(t)


def others_said(a0, b0):
    return [x for x in ALL if x[2] in OTHERS and a0 <= x[0] <= b0]


topics = []   # (כותרת, הנחיה למודל, [שורות ראיה כטקסט], [שורות ראיה כ-HTML])

# 1. תפקוד ותקשורת
ev_t, ev_h = [], []
ev_t.append(f"החניך דיבר {talk / 60:.1f} דקות מתוך {dur / 60:.0f} ({100 * talk / dur:.0f}%), {words} מילים.")
ev_t.append("קצב דיבור לפי שלישים של הגיחה (מילים לדקה): " + ", ".join(f"{x:.0f}" for x in thirds) + ".")
if qs:
    med = float(np.median([r["lat"] for r in ans])) if ans else None
    ev_t.append(f"נשאל {len(qs)} שאלות/קריאות, ענה על {len(ans)}" + (f", זמן תגובה חציוני {med:.1f} שנ׳." if med is not None else "."))
    for r in [r for r in qs if r["lat"] is None][:3]:
        ev_t.append(f'{tt(r["t"])} לא ענה ל{r["who"]}: "{r["p"][:90]}"')
    for r in sorted([r for r in ans if r["lat"] >= 2.0], key=lambda r: -r["lat"])[:3]:
        ev_t.append(f'{tt(r["t"])} ענה רק אחרי {r["lat"]:.1f} שנ׳ ל: "{r["p"][:80]}"')
for x in [u for u in utts if re.search(r"לא שמעתי|לא הבנתי|תחזור", u["text"])][:3]:
    ev_t.append(f'{tt(x["s"])} החניך: "{x["text"][:90]}"')
topics.append(("תפקוד ותקשורת",
               "תאר את התפקוד הכללי של החניך ואת התקשורת שלו בגיחה: כמה ואיך דיבר, איך הגיב לשאלות ולקריאות.",
               ev_t))

# 2. עמידה בלחץ
ev_t = []
if i1 is not None and i2 is not None:
    ev_t.append(f"מדד עומס קולי (0 = הרגיל שלו): מחצית ראשונה {i1:+.1f}, מחצית שנייה {i2:+.1f}.")
for u in sorted(peaks, key=lambda u: u["s"]):
    ctx = others_said(u["s"] - 30, u["s"])
    c = f' | 30 שנ׳ לפני: {ctx[-1][2]} אמר "{ctx[-1][3][:70]}"' if ctx else " | לפני כן לא היו הערות של מדריכה/ממיין"
    ev_t.append(f'{tt(u["s"])} שיא עומס ({u["idx"]:+.1f}): החניך אמר "{u["text"][:70]}"{c}')
for e in events:
    if e["type"] == "control":
        m_after = mean_idx(e["t"], e["t"] + 60)
        ev_t.append(f'{tt(e["t"])} העברת שליטה ("{e["text"][:60]}")' +
                    (f", מדד עומס בדקה שאחרי: {m_after:+.1f}" if m_after is not None else ""))
    elif e["type"] == "fault":
        ev_t.append(f'{tt(e["t"])} תקלה/תרחיש: "{e["text"][:70]}"')
topics.append(("עמידה בלחץ",
               "הערך איך החניך התמודד עם לחץ ועומס בגיחה: האם נשמע לחוץ, מתי, מה קרה לפני כן, והאם התאושש.",
               ev_t))

# ---- תרגילים ונתוני טיסה (מהחלון של זמני התרגילים ומקובץ ה-CSV של המאמן) ----
EX_TOL = {"alt": 50.0, "agl": 50.0, "spd": 5.0, "hdg": 10.0}   # טווח "בתוך היעד": ±50 רגל, ±5 קשר, ±10 מעלות
TGT_NAME = {"alt": ("גובה", "רגל"), "agl": ("גובה רדאר", "רגל"), "spd": ("מהירות", "קשר"), "hdg": ("כיוון", "מעלות")}
corr_all = [x for x in ALL if x[2] in OTHERS and CORR_RX.search(x[3])]
FL_BASE = None
if FL is not None:
    import sim_data

    FL_BASE = sim_data.window_stats(FL, 0, dur)


def flight_text(st, html_mode=False):
    """תיאור קצר של נתוני הטיסה בחלון זמן."""
    if not st or st.get("cover", 0) < 0.2:
        return ""
    p = []
    if "alt" in st:
        a = st["alt"]
        p.append(f"גובה (ברומטרי) {a['mean']:.0f} רגל בממוצע (בין {a['min']:.0f} ל-{a['max']:.0f}, סטיית תקן {a['sd']:.0f})")
    if "agl" in st:
        a = st["agl"]
        p.append(f"מעל הקרקע {a['mean']:.0f} רגל בממוצע (מינימום {a['min']:.0f})")
    if "spd" in st:
        v = st["spd"]
        p.append(f"מהירות {v['mean']:.0f} קשר בממוצע (בין {v['min']:.0f} ל-{v['max']:.0f}, סטיית תקן {v['sd']:.1f})")
    if "bank" in st:
        p.append(f"הטיה מקסימלית {st['bank']['max']:.0f} מעלות")
    if "stick" in st and FL_BASE and FL_BASE.get("stick") and st is not FL_BASE:
        r = st["stick"] / FL_BASE["stick"]
        p.append(f"פעילות סטיק פי {r:.1f} מהממוצע בגיחה" if abs(r - 1) > 0.15 else "פעילות סטיק רגילה")
    for k, g in st.get("tgt", {}).items():
        nm, un = TGT_NAME[k]
        p.append(f"יעד {nm} {g['target']:.0f}: סטייה ממוצעת {g['abs_dev']:.0f} {un}, "
                 f"{100 * g['in_tol']:.0f}% מהזמן בטווח ±{g['tol']:.0f}")
    return "; ".join(p)


EX_ROWS = []
for ex in EXERCISES:
    a0, b0 = ex["start"], ex["end"]
    us = [u for u in utts if a0 <= u["s"] < b0]
    w = sum(nwords(u["text"]) for u in us)
    tk_ = sum(u["e"] - u["s"] for u in us)
    pk = max((u for u in us if u["idx"] is not None), key=lambda u: u["idx"], default=None)
    cr = [x for x in corr_all if a0 <= x[0] < b0]
    q_ = [r for r in resp if r["q"] and a0 <= r["t"] < b0]
    an = [r["lat"] for r in q_ if r["lat"] is not None]
    fst = sim_data.window_stats(FL, a0, b0, ex["targets"], EX_TOL) if FL is not None else None
    fdy = sim_data.window_dynamics(FL, a0, b0, ex["targets"], EX_TOL) if FL is not None else None
    EX_ROWS.append({
        "name": ex["name"], "s": a0, "e": b0, "targets": ex["targets"],
        "wpm": w / max((b0 - a0) / 60, 1e-6), "talk": tk_, "words": w,
        "idx": mean_idx(a0, b0), "peak": pk, "corr": cr,
        "nq": len(q_), "nunans": len(q_) - len(an), "lat": float(np.median(an)) if an else None,
        "fl": fst, "dyn": fdy or {}, "utts": us,
        "others": [x for x in ALL if x[2] in OTHERS and a0 <= x[0] < b0],
    })

def dyn_text(d, targets):
    """שינויי הפרמטרים לאורך התרגיל, כטקסט."""
    p = []
    if "alt_in" in d:
        p.append(f"גובה בכניסה {d['alt_in']:.0f} וביציאה {d['alt_out']:.0f} רגל")
    if "agl_in" in d:
        p.append(f"מעל הקרקע {d['agl_in']:.0f} -> {d['agl_out']:.0f} רגל")
    if "spd_in" in d:
        p.append(f"מהירות בכניסה {d['spd_in']:.0f} וביציאה {d['spd_out']:.0f} קשר")
    if d.get("climb"):
        p.append(f"קצב נסיקה ממוצע {d['climb']:.0f} רגל לדקה")
    if d.get("descent"):
        p.append(f"קצב הנמכה ממוצע {abs(d['descent']):.0f} רגל לדקה")
    if "turn_total" in d and d["turn_total"] > 30:
        p.append(f"סה״כ פניות {d['turn_total']:.0f} מעלות (שינוי כיוון נטו {d['turn']:+.0f})")
    if "bank_turn_mean" in d:
        p.append(f"בפניות: הטיה ממוצעת {d['bank_turn_mean']:.0f} מעלות, תנודתיות ההטיה {d['bank_turn_sd']:.0f} מעלות"
                 + (f", שינוי גובה במהלך הפניות {d['alt_change_in_turns']:.0f} רגל" if "alt_change_in_turns" in d else ""))
    for k, g in d.get("tgt_dyn", {}).items():
        nm, un = TGT_NAME[k]
        if g.get("reach_s") is None:
            p.append(f"לא הגיע ליעד ה{nm} {targets[k]:.0f} (הכי קרוב: {g['closest']:.0f} {un})")
        else:
            p.append(f"הגיע ליעד ה{nm} אחרי {g['reach_s']:.0f} שנ׳, ומשם החזיק אותו {100 * g['held_after']:.0f}% מהזמן "
                     f"(סטייה מקסימלית {g['worst_after']:.0f} {un})")
    return "; ".join(p)


def exercise_evidence(r):
    ev = [f'זמן: מ-{tt(r["s"])} עד {tt(r["e"])} ({(r["e"] - r["s"]) / 60:.1f} דקות).']
    if r["targets"]:
        ev.append("יעדים שהוגדרו: " + ", ".join(f"{TGT_NAME[k][0]} {v:.0f} {TGT_NAME[k][1]}" for k, v in r["targets"].items()) + ".")
    fx = flight_text(r["fl"])
    if fx:
        ev.append("נתוני טיסה מהמאמן: " + fx + ".")
    dx = dyn_text(r["dyn"], r["targets"])
    if dx:
        ev.append("לאורך התרגיל: " + dx + ".")
    ev.append(f"קול: {r['wpm']:.0f} מילים לדקה" +
              (f", מדד עומס {r['idx']:+.1f} (0 = הרגיל שלו)" if r["idx"] is not None else "") +
              (f", שיא {r['peak']['idx']:+.1f} ב-{tt(r['peak']['s'])}" if r["peak"] else "") +
              (f"; נשאל {r['nq']} שאלות, לא ענה על {r['nunans']}" if r["nq"] else "") + ".")
    ev.append(f"הערות/תיקונים של המדריכה והממיין בתרגיל: {len(r['corr'])}.")
    oth = " | ".join(f'{tt(x[0])} {x[2]}: "{x[3][:80]}"' for x in r["others"])
    if oth:
        ev.append("מה אמרו המדריכה והממיין: " + oth[:700])
    tr = " | ".join(f'{tt(u["s"])} "{u["text"][:70]}"' for u in r["utts"] if nwords(u["text"]) >= 3)
    if tr:
        ev.append("מה החניך אמר בקשר: " + tr[:900])
    return ev


for r in EX_ROWS:
    r["ev"] = exercise_evidence(r)

ev_x = []
if FL is not None and not EX_ROWS:
    ev_x.append("נתוני טיסה לכל הגיחה: " + flight_text(FL_BASE) + ".")
if ev_x:
    topics.append(("ביצוע לפי נתוני הטיסה",
                   "הערך את ההטסה של החניך לפי נתוני הטיסה מהמאמן (יציבות גובה ומהירות, הטיות, פעילות על ההגאים).",
                   ev_x))

# סיכום אוטומטי: חלק על התרגילים / נתוני הטיסה
items = []
for r in EX_ROWS:
    fx = flight_text(r["fl"])
    items.append(f"<b>{html.escape(r['name'])}</b> ({ts(r['s'])}–{ts(r['e'])}): " +
                 (html.escape(fx) + ". " if fx else "") +
                 f"{r['wpm']:.0f} מילים לדקה" +
                 (f", מדד עומס {sg(r['idx'])}" if r["idx"] is not None else "") +
                 f", {len(r['corr'])} הערות של המדריכה/הממיין.")
if FL is not None and not EX_ROWS:
    items.append(html.escape("כל הגיחה: " + flight_text(FL_BASE) + "."))
if items and SYNC_NOTE:
    items.append(html.escape(SYNC_NOTE))
if items:
    sections.insert(len(sections) - 1, ("תרגילים ונתוני טיסה" if EX_ROWS else "נתוני טיסה", items))

# 3. ביצוע התרגילים
ev_t = []
corr = [x for x in ALL if x[2] in OTHERS and CORR_RX.search(x[3])]
ev_t.append(f"מספר ההערות/תיקונים של המדריכה והממיין במהלך הגיחה: {len(corr)}.")
ev_t += [f'{tt(x[0])} {x[2]}: "{x[3][:90]}"' for x in corr[:10]]
ctrl = [e for e in events if e["type"] == "control"]
if ctrl:
    ev_t.append(f"העברות שליטה: {len(ctrl)} (בזמנים {', '.join(tt(e['t']) for e in ctrl)}).")
probs = [u for u in utts if PROBLEM_RX.search(u["text"])]
ev_t += [f'{tt(u["s"])} החניך: "{u["text"][:90]}"' for u in probs[:6]]
topics.append(("ביצוע התרגילים",
               "הערך את איכות ביצוע התרגילים לפי ההערות של המדריכה והממיין, העברות השליטה, והבעיות שהחניך עצמו דיווח עליהן.",
               ev_t))

# 4. תחקיר עצמי ומודעות
ev_t = []
blocks, last_end = [], -1
for u in utts:
    if DEBRIEF_RX.search(u["text"]) and u["s"] > last_end:
        follow = [v["text"] for v in utts if u["s"] <= v["s"] <= u["s"] + 35]
        blocks.append((u["s"], " ".join(follow)))
        last_end = u["s"] + 35
ev_t += [f'{tt(t0)} החניך בתחקיר: "{txt[:230]}"' for t0, txt in blocks[:6]]
if not blocks:
    ev_t.append("לא נמצאו קטעי תחקיר עצמי של החניך.")
topics.append(("תחקיר עצמי ומודעות",
               "הערך את איכות התחקיר העצמי של החניך: האם זיהה את הבעיות שלו, האם הסביר סיבות, והאם הציע מה לשפר.",
               ev_t))


def make_llm():
    if not USE_LOCAL_LLM:
        return None, "כבוי (USE_LOCAL_LLM = False)"
    if not os.path.exists(LLM_PATH):
        return None, f"לא נמצא קובץ המודל: {LLM_PATH}"
    try:
        from llama_cpp import Llama
    except ImportError:
        return None, "הספרייה llama-cpp-python לא מותקנת"
    try:
        return Llama(model_path=LLM_PATH, n_ctx=6144, n_threads=max(2, (os.cpu_count() or 4) - 1),
                     n_gpu_layers=-1, verbose=False), None
    except Exception as e:  # noqa: BLE001
        return None, f"שגיאה בטעינת המודל: {e}"


def finish_text(txt, cut):
    """אם המודל נקטע באמצע (נגמרו לו המילים) - חותכים חזרה לסוף המשפט השלם האחרון,
    ומורידים שורות-כותרת שנשארו בלי תוכן (למשל "בנוסף, 1:")."""
    lines = [ln.rstrip() for ln in txt.strip().splitlines()]
    if cut:
        body = "\n".join(lines)
        m = list(re.finditer(r"[.!?](?=[\s\"')]|$)", body))
        if m:
            body = body[:m[-1].end()]
        lines = body.splitlines()
    out = []
    for ln in lines:
        out.append(ln)
    # כותרות/פתיחים בלי המשך בסוף הטקסט
    while out and (not out[-1].strip() or out[-1].strip().endswith(":") or len(out[-1].split()) <= 2):
        out.pop()
    # שורות-שארית קצרות שמסתיימות בנקודתיים באמצע (כותרת שהמודל לא המשיך)
    res = []
    for i, ln in enumerate(out):
        st = ln.strip()
        nxt = out[i + 1].strip() if i + 1 < len(out) else ""
        if st.endswith(":") and len(st.split()) <= 3 and not (nxt.startswith("-") or nxt.startswith("•")
                                                               or re.match(r"\d+[.)]", nxt)):
            continue
        res.append(ln)
    return "\n".join(res).strip()


def ask(llm, text, max_tokens=700):
    # פורמט ההנחיות של DictaLM 2.0 (מבוסס Mistral): [INST] ... [/INST]
    out = llm(f"[INST] {text} [/INST]", max_tokens=max_tokens, temperature=0.3, top_p=0.9,
              repeat_penalty=1.15, stop=["[INST]", "[/INST]", "</s>"])
    ch = out["choices"][0]
    txt = re.sub(r"\[/?INST\]|</?s>", "", ch["text"])
    return finish_text(txt, ch.get("finish_reason") == "length")


TOPIC_PROMPT = """אתה מדריך טיסה מנוסה שכותב הערכה על חניך אחרי גיחה בסימולטור מסוקים.
הנושא: {title}.
המשימה: {task}

להלן ראיות שנאספו אוטומטית מהקלטת הגיחה (התמלול אוטומטי ועלול להכיל שגיאות כתיב):
{evidence}

כתוב פסקה אחת של 3-5 משפטים שלמים בעברית (לא יותר), בגוף שלישי, שמסבירים מה הראיות מלמדות על החניך בנושא הזה - פרשנות ולא חזרה על המספרים.
ציין זמן (דקות:שניות) כשאתה מתייחס לרגע מסוים. הסתמך רק על הראיות. אם הן לא מספיקות למסקנה - כתוב זאת. אל תאבחן מצב רגשי או רפואי."""

EX_PROMPT = """אתה מדריך טיסה מנוסה במסוקים. כתוב חוות דעת על ההטסה של חניך בתרגיל "{name}" בגיחה בסימולטור.
להלן נתונים מהתרגיל - נתוני טיסה מהמאמן, ניתוח הקול ומה נאמר בקשר (התמלול אוטומטי ועלול להכיל שגיאות):
{evidence}

התייחס ל: שליטה בפרמטרים (גובה, מהירות, כיוון, הטיה) ויציבותם, הגעה ליעדים והחזקתם אם הוגדרו,
התיקונים וההערות שקיבל ואיך הגיב אליהם, ותקשורת ועומס. הסתמך רק על הנתונים. אל תאבחן מצב רגשי או רפואי.
ענה בדיוק בפורמט הזה:
חוות דעת: <3-4 משפטים שלמים בעברית, עם זמנים (דקות:שניות) כשאתה מתייחס לרגע מסוים>
ציון לתרגיל: <מספר שלם מ-1 עד 6, כש-6 = ביצוע מצוין>"""


def parse_ex(txt):
    m = re.search(r"ציון לתרגיל\s*[:：]?\s*([1-6])", txt)
    sc = int(m.group(1)) if m else None
    body = re.split(r"ציון לתרגיל", txt)[0]
    body = re.sub(r"^\s*חוות דעת\s*[:：]\s*", "", body.strip())
    return sc, body.strip()


FINAL_PROMPT = """אתה מדריך טיסה מנוסה. להלן הערכות שנכתבו על חניך אחרי גיחה בסימולטור:
{parts}

על סמך ההערכות האלה בלבד, כתוב בעברית שתי רשימות קצרות:
נקודות חוזק:
- ...
נקודות לשיפור:
- ...
עד 3 פריטים בכל רשימה, כל פריט משפט שלם אחד."""

# ---- ציון התאמה למסוקים (1-6) ----
# 6 = הכי מתאים. אפשר לשנות את התיאורים כאן לפי הסולם שלכם.
SCORE_SCALE = """1 - לא מתאים: קשיים חוזרים בביצוע, לא מתקן גם אחרי הערות, מאבד שליטה תחת לחץ.
2 - התאמה נמוכה: קשיים משמעותיים ברוב הגיחה, שיפור מועט.
3 - התאמה גבולית: חלק מהתרגילים בסדר, אבל חוזר על טעויות ונלחץ בעומס.
4 - מתאים: ביצוע סביר, מתקן אחרי הערות, תקשורת תקינה, לחץ מורגש אבל בשליטה.
5 - מתאים מאוד: ביצוע טוב, מעט תיקונים, רגוע תחת עומס, תחקיר עצמי מדויק.
6 - מצטיין: ביצוע מדויק כמעט בלי הערות, שולט היטב בלחץ, מזהה ומתקן בעצמו."""

SCORE_PROMPT = """אתה ממיין מנוסה לקורס טיס שקובע לחניך ציון התאמה למסוקים אחרי גיחה בסימולטור.

סולם הציונים:
{scale}

נתונים אוטומטיים מהגיחה:
{facts}

הערכה לפי נושאים:
{parts}

סיכום:
{final}

קבע ציון שלם אחד בין 1 ל-6 לפי הסולם, על סמך כל המידע למעלה בלבד. ענה בדיוק בפורמט הזה:
ציון: <מספר>
נימוק: <2-3 משפטים שלמים שמסבירים למה הציון הזה ולא אחד מעליו או מתחתיו>"""


def strip_tags(x):
    return html.unescape(re.sub(r"<[^>]+>", "", x))


def parse_score(txt):
    m = re.search(r"ציון\s*[:：]?\s*([1-6])(?:\.\d)?", txt) or re.search(r"(?<![\d:.])([1-6])(?![\d:.])", txt)
    sc = int(m.group(1)) if m else None
    why = re.split(r"נימוק\s*[:：]", txt, maxsplit=1)
    why = why[1] if len(why) > 1 else re.sub(r"ציון\s*[:：]?\s*[1-6](?:\.\d)?", "", txt)
    return sc, why.strip()

llm, llm_err = make_llm()
llm_out = {}
if llm:
    print(f"מריץ מודל שפה מקומי (DictaLM) על {len(EX_ROWS)} תרגילים + {len(topics)} נושאים + סיכום + ציון - כמה דקות...")
    t1 = time.time()
    try:
        for r in EX_ROWS:
            print(f"   תרגיל: {r['name']}...")
            r["llm"] = parse_ex(ask(llm, EX_PROMPT.format(name=r["name"], evidence="\n".join(f"- {x}" for x in r["ev"])[:3800]),
                                    max_tokens=420))
        for title, task, ev in topics:
            print(f"   {title}...")
            llm_out[title] = ask(llm, TOPIC_PROMPT.format(title=title, task=task, evidence="\n".join(f"- {x}" for x in ev)[:3500]))
        print("   סיכום...")
        ex_part = "\n".join(f'- {r["name"]}' + (f' (ציון {r["llm"][0]})' if r.get("llm") and r["llm"][0] else "") +
                            f': {r["llm"][1] if r.get("llm") else ""}' for r in EX_ROWS)
        all_parts = ((f"[חוות דעת לפי תרגיל]\n{ex_part}\n\n" if EX_ROWS else "") +
                     "\n\n".join(f"[{t}]\n{llm_out.get(t, '')}" for t, _, _ in topics))
        llm_out["__final__"] = ask(llm, FINAL_PROMPT.format(parts=all_parts[:5500]), max_tokens=700)
        print("   ציון התאמה...")
        facts = "\n".join(f"- {strip_tags(it)}" for title, its in sections if title != "חשוב לדעת" for it in its)
        sc_txt = ask(llm, SCORE_PROMPT.format(
            scale=SCORE_SCALE, facts=facts[:2000],
            parts=all_parts[:4500],
            final=llm_out["__final__"][:1200]), max_tokens=300)
        llm_out["__score__"] = parse_score(sc_txt)
        print(f"   הסתיים ({time.time() - t1:.0f} שניות)")
    except Exception as e:  # noqa: BLE001
        llm_err = f"שגיאה בהרצת המודל: {e}"
        print(f"   [מודל מקומי] {llm_err}")
else:
    print(f"   [מודל מקומי] {llm_err}")


def linkify(esc):
    return re.sub(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", lambda m: ts(int(m.group(1)) * 60 + int(m.group(2))), esc)


def text_to_html(txt):
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        esc = linkify(html.escape(line.lstrip("-•* ").strip()))
        if line.endswith(":") and len(line) < 60:
            out.append(f"<h3>{esc}</h3>")
        elif line[:1] in "-•*" or re.match(r"^\d+[.)]", line):
            out.append(f"<p class='li'>• {esc}</p>")
        else:
            out.append(f"<p>{esc}</p>")
    return "".join(out)


parts = []
if llm_err:
    parts.append(f'<p class="u">ההערכה של המודל המקומי לא נוצרה: {html.escape(llm_err)}. '
                 'מוצגות רק הראיות. להפעלה: להריץ פעם אחת את prepare_offline.py.</p>')
for title, task, ev in topics:
    body = text_to_html(llm_out[title]) if title in llm_out else ""
    ev_html = "".join(f"<li>{linkify(html.escape(x))}</li>" for x in ev)
    parts.append(f"<h3>{html.escape(title)}</h3>{body}"
                 f"<details{'' if body else ' open'}><summary>הראיות שעליהן זה מבוסס</summary><ul>{ev_html}</ul></details>")
if "__final__" in llm_out:
    parts.append('<h3 class="final">סיכום</h3>' + text_to_html(llm_out["__final__"]))
sc, why = llm_out.get("__score__", (None, ""))
if sc:
    parts.append(f'<div class="score"><div class="score-num"><span>{sc}</span><small>מתוך 6</small></div>'
                 f'<div class="score-txt"><h3>ציון התאמה למסוקים</h3>{text_to_html(why)}'
                 f'<details><summary>סולם הציונים</summary><ul>'
                 + "".join(f"<li>{html.escape(x)}</li>" for x in SCORE_SCALE.splitlines()) +
                 '</ul></details></div></div>')
elif "__score__" in llm_out:
    parts.append('<p class="u">המודל לא החזיר ציון בפורמט תקין.</p>')
if llm_out:
    parts.append('<p class="u">נכתב ע״י מודל שפה שרץ על המחשב (DictaLM 2.0), בלי רשת, על סמך הראיות שמתחת לכל נושא. '
                 'מודל כזה יכול לטעות או להגזים - כדאי לבדוק מול הראיות, הגרפים וההקלטה. '
                 'הציון הוא הצעה של המודל ולא החלטת מיון.</p>')
llm_html = "".join(parts)

summary_html = "".join(
    f'<h3>{html.escape(title)}</h3><ul>' + "".join(f"<li>{it}</li>" for it in its) + "</ul>"
    for title, its in sections
)

data = {
    "dur": round(dur, 2),
    "hop": HOP_SEC,
    "t": T, "vol": VOL, "f0": F0, "rate": RATE,
    "utts": [[round(u["s"], 2), round(u["e"], 2), u["text"], u["idx"], u["z"]] for u in utts],
    "resp": resp,
    "events": events,
    "evtypes": [[k, name] for k, name, _ in EVENT_RX],
    "ex": [[r["name"], round(r["s"], 1), round(r["e"], 1), r["targets"]] for r in EX_ROWS],
    "sync": SYNC_NOTE,
    "vsync": (VSYNC or {}).get("check"),
    "fl": ({"step": sim_data.STEP, **{k: sim_data.round_list(FL[k], 1) for k in ("alt", "agl", "spd", "bank", "vs", "stick") if k in FL}}
           if FL is not None else None),
}

# ---- טבלת התרגילים ----
def _cell(x, fmt_, empty="—"):
    return empty if x is None else fmt_.format(x)


ex_html = ""
if EX_ROWS:
    has_fl = FL is not None
    has_sc = any(r.get("llm") and r["llm"][0] for r in EX_ROWS)
    head = ("<tr><th>תרגיל</th><th>זמן</th><th>משך</th>" +
            ("<th>גובה (רגל)</th><th>מעל הקרקע (רגל)</th><th>מהירות (קשר)</th><th>הטיה מקס׳</th><th>פעילות סטיק</th><th>בתוך היעד</th>" if has_fl else "") +
            "<th>מדד עומס</th><th>מילים לדקה</th><th>הערות</th>" + ("<th>ציון</th>" if has_sc else "") + "</tr>")
    rows = []
    for r in EX_ROWS:
        f = r["fl"] or {}
        cells = [f'<td><a href="#" class="exl" data-a="{r["s"]:.1f}" data-b="{r["e"]:.1f}">{html.escape(r["name"])}</a></td>',
                 f'<td class="num">{ts(r["s"])}</td>', f'<td><bdi>{(r["e"] - r["s"]) / 60:.1f}</bdi> דק׳</td>']
        if has_fl:
            a, v, g = f.get("alt"), f.get("spd"), f.get("agl")
            cells.append(f'<td class="num">{a["mean"]:.0f} <span class="u">±{a["sd"]:.0f}</span></td>' if a else "<td>—</td>")
            cells.append(f'<td><bdi>{g["mean"]:.0f}</bdi> <span class="u">(מינ׳ <bdi>{g["min"]:.0f}</bdi>)</span></td>' if g else "<td>—</td>")
            cells.append(f'<td class="num">{v["mean"]:.0f} <span class="u">±{v["sd"]:.1f}</span></td>' if v else "<td>—</td>")
            cells.append(f'<td class="num">{f["bank"]["max"]:.0f}°</td>' if "bank" in f else "<td>—</td>")
            sr = (f["stick"] / FL_BASE["stick"]) if ("stick" in f and FL_BASE and FL_BASE.get("stick")) else None
            cells.append(f'<td class="num">×{sr:.1f}</td>' if sr is not None else "<td>—</td>")
            tg = f.get("tgt", {})
            cells.append("<td>" + ("<br>".join(f'{TGT_NAME[k][0]} {g["target"]:.0f}: <b>{100 * g["in_tol"]:.0f}%</b>'
                                               for k, g in tg.items()) or '<span class="u">אין יעד</span>') + "</td>")
        cells.append(f'<td class="num">{sg(r["idx"]) if r["idx"] is not None else "—"}</td>')
        cells.append(f'<td class="num">{r["wpm"]:.0f}</td>')
        cells.append(f'<td class="num">{len(r["corr"])}</td>')
        if has_sc:
            sc_ = r.get("llm", (None, ""))[0]
            cells.append(f'<td><span class="chip">{sc_}</span></td>' if sc_ else "<td>—</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    note = ("פעילות סטיק = כמה הסטיק זז ביחס לממוצע של החניך בגיחה (×1 = רגיל, גבוה = תיקונים רבים). "
            f"בתוך היעד = אחוז הזמן בטווח ±{EX_TOL['alt']:.0f} רגל / ±{EX_TOL['spd']:.0f} קשר / ±{EX_TOL['hdg']:.0f}°. "
            if has_fl else "")
    ex_html = (f'<div class="card"><h2>תרגילים</h2><p class="u">{note}לחיצה על שם תרגיל מציגה אותו בגרפים.</p>'
               f'<div class="tblwrap"><table class="ex"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div></div>')

    # חוות דעת לפי תרגיל
    blocks_ = []
    for i, r in enumerate(EX_ROWS):
        sc_, txt_ = r.get("llm", (None, ""))
        ev_html = "".join(f"<li>{linkify(html.escape(x))}</li>" for x in r["ev"])
        blocks_.append(
            f'<div class="exev"><div class="exev-h">'
            + (f'<span class="chip big">{sc_}<small>/6</small></span>' if sc_ else "")
            + f'<h3><a href="#" class="exl" data-a="{r["s"]:.1f}" data-b="{r["e"]:.1f}">{html.escape(r["name"])}</a> '
              f'<span class="u"><bdi dir="ltr">{mmss(r["s"])}–{mmss(r["e"])}</bdi></span></h3></div>'
            + (text_to_html(txt_) if txt_ else "")
            + f'<details{"" if txt_ else (" open" if i == 0 else "")}><summary>הנתונים שעליהם זה מבוסס</summary><ul>{ev_html}</ul></details></div>')
    note_ = ("נכתב ע״י המודל המקומי לכל תרגיל בנפרד, לפי נתוני הטיסה, הקול ומה שנאמר בקשר באותו תרגיל. "
             "הציון (1-6) הוא הצעה של המודל." if has_sc else
             "בלי המודל המקומי מוצגים רק הנתונים של כל תרגיל.")
    ex_html += (f'<div class="card summary"><h2>חוות דעת לפי תרגיל</h2><p class="u">{note_} לחיצה על שם תרגיל מציגה אותו בגרפים.</p>'
                + "".join(blocks_) + "</div>")


page = r"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>גרפי חניך</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-1-fill: rgba(42,120,214,0.14); --brush: rgba(42,120,214,0.12);
  --ev1: #2a78d6; --ev2: #eb6834; --ev3: #1baf7a;
  --band0: rgba(42,120,214,0.07); --band1: rgba(11,11,11,0.045);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-1-fill: rgba(57,135,229,0.18); --brush: rgba(57,135,229,0.16);
    --ev1: #3987e5; --ev2: #d95926; --ev3: #199e70;
    --band0: rgba(57,135,229,0.12); --band1: rgba(255,255,255,0.05);
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text-primary);
       font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 20px 16px 40px; }
h1 { font-size: 1.35rem; margin: 0 0 4px; }
.sub { color: var(--text-secondary); font-size: .9rem; margin: 0 0 16px; }
.filters { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 8px; }
.filters label { color: var(--text-secondary); font-size: .9rem; }
.filters input[type=text] { width: 76px; padding: 6px 8px; border: 1px solid var(--border); border-radius: 6px;
  background: var(--surface-1); color: var(--text-primary); font: inherit; text-align: center;
  font-variant-numeric: tabular-nums; }
button { padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border); cursor: pointer;
  background: var(--surface-1); color: var(--text-primary); font: inherit; }
button.primary { background: var(--series-1); color: #fff; border-color: transparent; }
.range-label { color: var(--text-secondary); font-size: .9rem; margin-inline-start: 6px; }
.err { color: #c0392b; font-size: .85rem; }
.hint { color: var(--muted); font-size: .8rem; margin: 0 0 12px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-bottom: 14px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; }
.tile .k { color: var(--text-secondary); font-size: .85rem; }
.tile .v { font-size: 1.5rem; font-weight: 600; margin-top: 2px; }
.u { color: var(--muted); font-size: .85rem; font-weight: 400; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 14px 6px; margin-bottom: 12px; }
.card h2 { font-size: 1rem; margin: 0; }
.chart { width: 100%; height: 190px; direction: ltr; touch-action: none; }
.chart svg { display: block; width: 100%; height: 100%; cursor: crosshair; user-select: none; }
.tip { position: fixed; pointer-events: none; background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: .85rem;
  box-shadow: 0 4px 16px rgba(0,0,0,.15); max-width: 320px; display: none; z-index: 10; direction: rtl; }
.tip .tt { color: var(--text-secondary); font-variant-numeric: tabular-nums; }
.tip .tv { font-weight: 600; margin: 2px 0; }
.tip .silent { color: var(--muted); }
audio { width: 100%; margin: 0 0 14px; }
details { margin-top: 10px; }
summary { cursor: pointer; color: var(--text-secondary); }
table { width: 100%; border-collapse: collapse; font-size: .85rem; margin-top: 8px; }
th, td { border-bottom: 1px solid var(--grid); padding: 5px 6px; text-align: right; }
td.num { font-variant-numeric: tabular-nums; direction: ltr; text-align: left; }
.tblwrap { overflow-x: auto; }
table.ex td.num { text-align: right; }
table.ex td, table.ex th { white-space: nowrap; padding: 5px 5px; }
table.ex a.exl { font-weight: 600; color: var(--series-1); text-decoration: none; }
.chip { display: inline-block; min-width: 26px; padding: 2px 8px; border-radius: 12px; text-align: center;
  background: var(--series-1-fill); color: var(--series-1); font-weight: 700; font-variant-numeric: tabular-nums; }
.chip.big { font-size: 1.25rem; padding: 4px 12px; flex: 0 0 auto; }
.chip small { font-size: .7rem; font-weight: 400; color: var(--text-secondary); }
.exev { border-top: 1px solid var(--grid); padding: 10px 0 6px; }
.exev:first-of-type { border-top: 0; }
.exev-h { display: flex; align-items: center; gap: 10px; }
.exev-h h3 { margin: 0; }
.exev a.exl { color: var(--series-1); text-decoration: none; }
h2.sec { font-size: 1.05rem; margin: 22px 0 8px; color: var(--text-secondary); }
.heat { width: 100%; height: 96px; direction: ltr; }
.heat svg { display: block; width: 100%; height: 100%; cursor: pointer; }
.legend { display: flex; align-items: center; gap: 8px; color: var(--text-secondary); font-size: .8rem;
  margin: 2px 0 6px; direction: rtl; flex-wrap: wrap; }
.legend .ramp { width: 160px; height: 10px; border-radius: 3px; direction: ltr; }
.legend .sw { width: 14px; height: 10px; border-radius: 3px; background: var(--grid); display: inline-block; }
.zoom { display: inline-flex; gap: 4px; direction: ltr; }
.zoom button { padding: 6px 10px; min-width: 34px; }
.summary h3 { font-size: .95rem; margin: 14px 0 4px; }
.summary ul { margin: 0; padding-inline-start: 20px; line-height: 1.7; }
.summary li { margin-bottom: 4px; }
.summary p { margin: 4px 0; line-height: 1.7; }
.summary p.li { margin-inline-start: 10px; }
.summary details { margin: 4px 0 10px; font-size: .88rem; color: var(--text-secondary); }
.summary h3.final { margin-top: 20px; border-top: 1px solid var(--grid); padding-top: 12px; }
.summary .score { display: flex; gap: 16px; align-items: flex-start; margin-top: 18px; padding: 14px 16px;
  border: 1px solid var(--border); border-radius: 10px; background: var(--page); }
.summary .score-num { flex: 0 0 auto; min-width: 72px; text-align: center; }
.summary .score-num span { display: block; font-size: 44px; font-weight: 700; line-height: 1; color: var(--series-1); }
.summary .score-num small { color: var(--text-secondary); font-size: 12px; }
.summary .score-txt { flex: 1; min-width: 0; }
.summary .score-txt h3 { margin-top: 0; }
.summary code { direction: ltr; unicode-bidi: isolate; font-size: .8rem; }
.summary a.ts { color: var(--series-1); font-variant-numeric: tabular-nums; text-decoration: none;
  border-bottom: 1px dashed currentColor; direction: ltr; unicode-bidi: isolate; }
.smooth { color: var(--text-secondary); font-size: .9rem; display: inline-flex; align-items: center; gap: 6px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>גרפי חניך — __TITLE__</h1>
  <p class="sub">נמדד מההקלטה כל 0.05 שנייה, רק בזמן שהחניך מדבר. כשהחניך לא מדבר הגרף יורד ל-0.</p>

  <div class="filters">
    <label for="from">מדקה</label><input type="text" id="from" placeholder="00:00" inputmode="numeric">
    <label for="to">עד דקה</label><input type="text" id="to" placeholder="__END__" inputmode="numeric">
    <button class="primary" id="apply">הצג טווח</button>
    <button id="all">הכל</button>
    <span class="zoom" role="group" aria-label="זום">
      <button id="zin" title="זום פנימה (Ctrl + גלגלת למעלה)">＋</button>
      <button id="zout" title="זום החוצה (Ctrl + גלגלת למטה)">－</button>
      <button id="pleft" title="הזזה אחורה בזמן (Shift + גלגלת)">◀</button>
      <button id="pright" title="הזזה קדימה בזמן (Shift + גלגלת)">▶</button>
    </span>
    <label class="smooth"><input type="checkbox" id="smooth"> החלקה (חצי שנייה)</label>
    <span class="range-label" id="rangeLabel"></span>
    <span class="err" id="err"></span>
  </div>
  <p class="hint">זום: <b>Ctrl + גלגלת</b> על גרף (מתקרב למקום של העכבר) או ＋/－. הזזה בזמן: <b>Shift + גלגלת</b> או ◀ ▶. גרירה על גרף בוחרת טווח, לחיצה מנגנת את הרגע בהקלטה.</p>

  <audio id="player" controls preload="none"><source src="__AUDIO__"></audio>

  <div class="tiles" id="tiles"></div>
  __EXER__
  <div id="charts"></div>

  <div class="card summary">
    <h2>סיכום החניך בגיחה</h2>
    <p class="u">נוצר אוטומטית מהנתונים של כל הגיחה. לחיצה על זמן מציגה את הרגע בגרפים.</p>
    __SUMMARY__
  </div>

  <div class="card summary">
    <h2>הערכת החניך לפי נושאים</h2>
    __LLM__
  </div>

  <details>
    <summary>המשפטים של החניך בטווח</summary>
    <table><thead><tr><th>זמן</th><th>משך (שנ׳)</th><th>עוצמה ממוצעת</th><th>גובה צליל ממוצע (Hz)</th><th>קצב (מילים/שנ׳)</th><th>מה נאמר</th></tr></thead>
    <tbody id="tbody"></tbody></table>
  </details>
</div>
<div class="tip" id="tip"></div>

<script>
const D = __DATA__;
const N = D.t.length;
const METRICS = [
  {key: "vol",  name: "עוצמה",           unit: "(יחסית)",      dec: 1},
  {key: "f0",   name: "גובה צליל (תדר)", unit: "Hz",           dec: 0},
  {key: "rate", name: "קצב דיבור",       unit: "מילים לשנייה", dec: 1},
];
let range = [0, D.dur];
const audio = document.getElementById("player");
const tip = document.getElementById("tip");
const fmt = s => { s = Math.max(0, Math.floor(s)); return String(Math.floor(s/60)).padStart(2,"0") + ":" + String(s%60).padStart(2,"0"); };
const fmtF = s => fmt(s) + "." + Math.floor((s % 1) * 10);
function parseT(v) {
  v = (v || "").trim(); if (!v) return null;
  let m = v.match(/^(\d+):(\d{1,2})$/); if (m) return (+m[1])*60 + (+m[2]);
  if (/^\d+(\.\d+)?$/.test(v)) return parseFloat(v) * 60;   // "12" = דקה 12
  return NaN;
}
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function lowerBound(arr, x) { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo+hi) >> 1; if (arr[m] < x) lo = m+1; else hi = m; } return lo; }

// קנה מידה קבוע לכל מדד (לא קופץ כשמשנים טווח); ערכים חריגים (1% עליון) נחתכים לראש הגרף
METRICS.forEach(M => {
  const v = D[M.key].filter(x => x != null && x > 0).sort((a,b) => a-b);
  M.hi = (v.length ? v[Math.floor(0.99*(v.length-1))] : 1) * 1.15;
});

const UTT_STARTS = D.utts.map(u => u[0]);
function utterAt(t) {
  const i = lowerBound(UTT_STARTS, t + 1e-6) - 1;
  return (i >= 0 && D.utts[i][1] >= t) ? D.utts[i] : null;
}

// נקודה לכל פיקסל: ממוצע הערכים בזמן דיבור באותו פיקסל, או 0 אם החניך לא דיבר
// המשפט של החניך שתופס הכי הרבה זמן בתוך [t0, t1] (פיקסל אחד יכול לכסות כמה שניות)
function utterIn(t0, t1) {
  let best = null, bestOv = 0;
  for (let i = Math.max(0, lowerBound(UTT_STARTS, t0) - 1); i < D.utts.length && D.utts[i][0] <= t1; i++) {
    const u = D.utts[i], ov = Math.min(u[1], t1) - Math.max(u[0], t0);
    if (ov > bestOv) { bestOv = ov; best = u; }
  }
  return best;
}

function binned(M, a, b, nb) {
  const sum = new Float64Array(nb), cnt = new Uint32Array(nb);
  const span = b - a, arr = D[M.key];
  const smooth = document.getElementById("smooth").checked ? 5 : 0;   // ±5 נקודות = חצי שנייה
  for (let i = lowerBound(D.t, a); i < N && D.t[i] <= b; i++) {
    const t = D.t[i];
    const k = Math.min(nb - 1, Math.floor((t - a) / span * nb));
    let v = arr[i];
    if (smooth) {
      let s = 0, c = 0;
      for (let j = Math.max(0, i - smooth); j <= Math.min(N - 1, i + smooth); j++)
        if (Math.abs(D.t[j] - t) <= smooth * D.hop + 1e-6 && arr[j] != null) { s += arr[j]; c++; }
      v = c ? s / c : null;
    }
    if (v != null) { sum[k] += v; cnt[k]++; }
  }
  const out = new Array(nb);
  for (let k = 0; k < nb; k++) out[k] = cnt[k] ? sum[k] / cnt[k] : 0;   // לא מדבר -> 0
  return out;
}

function timeTicks(lo, hi) {
  const span = hi - lo, c = [1,2,5,10,15,30,60,120,300,600,900,1200,1800];
  const step = c.find(s => span/s <= 8) || 3600;
  const out = []; for (let v = Math.ceil(lo/step)*step; v <= hi; v += step) out.push(v); return out;
}
function niceTicks(hi, n) {
  const step0 = hi / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1,2,5,10].map(k => k*mag).find(s => hi/s <= n) || 10*mag;
  const out = []; for (let v = 0; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6)); return out;
}

function drawChart(M, host) {
  const W = host.clientWidth, H = host.clientHeight, ml = 46, mr = 12, mt = 8, mb = 24;
  const pw = Math.max(50, W - ml - mr), ph = H - mt - mb;
  const [a, b] = range, span = b - a;
  // נקודה לכל פיקסל, אבל לא יותר מנקודה לכל 0.05 שנייה
  const nb = Math.max(2, Math.min(Math.round(pw), Math.ceil(span / D.hop)));
  const vals = binned(M, a, b, nb);
  const X = k => ml + (k + 0.5) / nb * pw;
  const Xt = t => ml + (t - a) / span * pw;
  const Y = v => mt + ph - Math.min(v, M.hi) / M.hi * ph;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${M.name}">`;
  for (const v of niceTicks(M.hi, 4)) {
    s += `<line x1="${ml}" x2="${ml+pw}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)" stroke-width="1"/>`;
    s += `<text x="${ml-6}" y="${Y(v)+4}" text-anchor="end" font-size="11" fill="var(--muted)" style="font-variant-numeric:tabular-nums">${v}</text>`;
  }
  for (const t of timeTicks(a, b))
    s += `<text x="${Xt(t)}" y="${H-6}" text-anchor="middle" font-size="11" fill="var(--muted)" style="font-variant-numeric:tabular-nums">${fmt(t)}</text>`;
  let line = "";
  for (let k = 0; k < nb; k++) line += (k ? "L" : "M") + X(k).toFixed(1) + " " + Y(vals[k]).toFixed(1);
  s += `<path d="${line}L${X(nb-1).toFixed(1)} ${Y(0)}L${X(0).toFixed(1)} ${Y(0)}Z" fill="var(--series-1-fill)" stroke="none"/>`;
  s += `<path d="${line}" fill="none" stroke="var(--series-1)" stroke-width="${nb > pw*0.6 ? 1.2 : 2}" stroke-linejoin="round"/>`;
  s += `<line x1="${ml}" x2="${ml+pw}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--axis)" stroke-width="1"/>`;
  s += `<line class="xh" y1="${mt}" y2="${mt+ph}" stroke="var(--muted)" stroke-width="1" visibility="hidden"/>`;
  s += `<circle class="dot" r="4.5" fill="var(--series-1)" stroke="var(--surface-1)" stroke-width="2" visibility="hidden"/>`;
  s += `<rect class="brush" y="${mt}" height="${ph}" fill="var(--brush)" visibility="hidden"/>`;
  s += `<rect x="${ml}" y="${mt}" width="${pw}" height="${ph}" fill="transparent"/></svg>`;
  host.innerHTML = s;

  const svg = host.querySelector("svg"), xh = svg.querySelector(".xh"), dot = svg.querySelector(".dot"), br = svg.querySelector(".brush");
  const lx = e => { const r = svg.getBoundingClientRect(); return (e.clientX - r.left) * W / r.width; };
  const tAt = x => a + (Math.min(Math.max(x, ml), ml+pw) - ml) / pw * span;
  let dragFrom = null;
  svg.addEventListener("pointermove", e => {
    const x = lx(e);
    if (dragFrom != null) {
      const x0 = Math.max(Math.min(dragFrom, x), ml), x1 = Math.min(Math.max(dragFrom, x), ml+pw);
      br.setAttribute("x", x0); br.setAttribute("width", Math.max(0, x1 - x0)); br.setAttribute("visibility", "visible");
    }
    if (x < ml || x > ml + pw) { hide(); return; }
    const k = Math.min(nb - 1, Math.max(0, Math.floor((x - ml) / pw * nb)));
    const t = a + (k + 0.5) / nb * span, v = vals[k];
    const tb0 = a + k / nb * span, tb1 = a + (k + 1) / nb * span;   // הזמן שהפיקסל הזה מכסה
    xh.setAttribute("x1", X(k)); xh.setAttribute("x2", X(k)); xh.setAttribute("visibility", "visible");
    dot.setAttribute("cx", X(k)); dot.setAttribute("cy", Y(v)); dot.setAttribute("visibility", "visible");
    const u = utterIn(tb0, tb1);
    const tlabel = (tb1 - tb0) >= 1 ? `<bdi dir="ltr">${fmt(tb0)} – ${fmt(tb1)}</bdi>` : fmtF(t);
    tip.innerHTML = `<div class="tt">${tlabel}</div>` +
      (u ? `<div class="tv">${M.name}: ${v > 0 ? v.toFixed(M.dec) + " " + M.unit : "—"}</div>` +
           `<div>${escapeHtml(u[2])}</div><div class="silent">(משפט שמתחיל ב-${fmt(u[0])})</div>`
         : `<div class="silent">החניך לא מדבר</div>`);
    tip.style.display = "block";
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = (e.clientY + 14) + "px";
  });
  function hide() { xh.setAttribute("visibility","hidden"); dot.setAttribute("visibility","hidden"); tip.style.display = "none"; }
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("pointerdown", e => { dragFrom = lx(e); svg.setPointerCapture(e.pointerId); });
  svg.addEventListener("pointerup", e => {
    const x = lx(e), from = dragFrom; dragFrom = null; br.setAttribute("visibility","hidden");
    if (from == null) return;
    if (Math.abs(x - from) > 6) setRange(tAt(Math.min(from, x)), tAt(Math.max(from, x)));
    else { try { audio.currentTime = Math.max(0, tAt(x) - 1); audio.play(); } catch (_) {} }
  });
}

function stats(a, b) {
  const i0 = lowerBound(D.t, a), i1 = lowerBound(D.t, b + 1e-9);
  const res = {};
  METRICS.forEach(M => {
    let s = 0, c = 0;
    for (let i = i0; i < i1; i++) { const v = D[M.key][i]; if (v != null) { s += v; c++; } }
    res[M.key] = c ? s / c : null;
  });
  const us = D.utts.filter(u => u[1] > a && u[0] < b);
  const talk = us.reduce((s, u) => s + Math.min(u[1], b) - Math.max(u[0], a), 0);
  const words = us.reduce((s, u) => s + (u[2].match(/\S+/g) || []).length, 0);
  res.rate = talk > 0 ? words / talk : null;   // קצב אמיתי: מילים / זמן דיבור
  res.talk = talk; res.n = us.length; res.us = us;
  return res;
}

// ---- מילים בחלון נע של 30 שניות, בצעדים של שנייה (0-30, 1-31, 2-32 ...) ----
const WIN = 30;
const NSEC = Math.ceil(D.dur);
const DENS = new Float64Array(NSEC + 1);          // מילים בכל שנייה (משפט מתפזר לאורך הזמן שלו)
D.utts.forEach(([s0, e0, txt]) => {
  const n = (txt.match(/\S+/g) || []).length, d = Math.max(e0 - s0, 0.01);
  for (let k = Math.floor(s0); k <= Math.min(NSEC, Math.floor(e0)); k++) {
    const ov = Math.min(e0, k + 1) - Math.max(s0, k);
    if (ov > 0) DENS[k] += n * ov / d;
  }
});
const PREF = new Float64Array(NSEC + 2);
for (let k = 0; k <= NSEC; k++) PREF[k + 1] = PREF[k] + DENS[k];
// SLIDE[i] = כמה מילים בין שנייה i לשנייה i+30
const NW = Math.max(1, NSEC - WIN + 1);
const SLIDE = new Float64Array(NW);
for (let i = 0; i < NW; i++) SLIDE[i] = PREF[Math.min(NSEC + 1, i + WIN)] - PREF[i];
const SMAX = Math.max(1, ...SLIDE);

function drawSlide(host) {
  const W = host.clientWidth, H = host.clientHeight, ml = 46, mr = 12, mt = 8, mb = 24;
  const pw = Math.max(50, W - ml - mr), ph = H - mt - mb;
  const [a, b] = range, span = b - a;
  const hi = SMAX * 1.1;
  const Xt = t => ml + (t - a) / span * pw;
  const Y = v => mt + ph - Math.min(v, hi) / hi * ph;
  // כל חלון מצויר באמצע שלו (חלון 0-30 בנקודה 15), כדי שיתיישר עם הגרפים האחרים
  const i0 = Math.max(0, Math.floor(a - WIN / 2)), i1 = Math.min(NW - 1, Math.ceil(b - WIN / 2));
  const nb = Math.max(2, Math.min(Math.round(pw), i1 - i0 + 1));
  const sum = new Float64Array(nb), cnt = new Uint32Array(nb);
  for (let i = i0; i <= i1; i++) {
    const tc = i + WIN / 2; if (tc < a || tc > b) continue;
    const k = Math.min(nb - 1, Math.floor((tc - a) / span * nb));
    sum[k] += SLIDE[i]; cnt[k]++;
  }
  const pts = [];
  for (let k = 0; k < nb; k++) if (cnt[k]) pts.push([a + (k + 0.5) / nb * span, sum[k] / cnt[k]]);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="מילים ב-30 שניות">`;
  for (const v of niceTicks(hi, 4)) {
    s += `<line x1="${ml}" x2="${ml+pw}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)" stroke-width="1"/>`;
    s += `<text x="${ml-6}" y="${Y(v)+4}" text-anchor="end" font-size="11" fill="var(--muted)" style="font-variant-numeric:tabular-nums">${v}</text>`;
  }
  for (const t of timeTicks(a, b))
    s += `<text x="${Xt(t)}" y="${H-6}" text-anchor="middle" font-size="11" fill="var(--muted)" style="font-variant-numeric:tabular-nums">${fmt(t)}</text>`;
  if (pts.length > 1) {
    const line = pts.map((p, j) => (j ? "L" : "M") + Xt(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join("");
    s += `<path d="${line}L${Xt(pts[pts.length-1][0]).toFixed(1)} ${Y(0)}L${Xt(pts[0][0]).toFixed(1)} ${Y(0)}Z" fill="var(--series-1-fill)"/>`;
    s += `<path d="${line}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
  }
  s += `<line x1="${ml}" x2="${ml+pw}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--axis)" stroke-width="1"/>`;
  s += `<rect class="win" y="${mt}" height="${ph}" fill="var(--brush)" visibility="hidden"/>`;
  s += `<line class="xh" y1="${mt}" y2="${mt+ph}" stroke="var(--muted)" stroke-width="1" visibility="hidden"/>`;
  s += `<circle class="dot" r="4.5" fill="var(--series-1)" stroke="var(--surface-1)" stroke-width="2" visibility="hidden"/>`;
  s += `<rect class="brush" y="${mt}" height="${ph}" fill="var(--brush)" visibility="hidden"/>`;
  s += `<rect x="${ml}" y="${mt}" width="${pw}" height="${ph}" fill="transparent"/></svg>`;
  host.innerHTML = s;

  const svg = host.querySelector("svg"), xh = svg.querySelector(".xh"), dot = svg.querySelector(".dot");
  const win = svg.querySelector(".win"), br = svg.querySelector(".brush");
  const lx = e => { const r = svg.getBoundingClientRect(); return (e.clientX - r.left) * W / r.width; };
  const tAt = x => a + (Math.min(Math.max(x, ml), ml+pw) - ml) / pw * span;
  const iAt = x => Math.min(NW - 1, Math.max(0, Math.round(tAt(x) - WIN / 2)));
  let dragFrom = null;
  svg.addEventListener("pointermove", e => {
    const x = lx(e);
    if (dragFrom != null) {
      const x0 = Math.max(Math.min(dragFrom, x), ml), x1 = Math.min(Math.max(dragFrom, x), ml+pw);
      br.setAttribute("x", x0); br.setAttribute("width", Math.max(0, x1 - x0)); br.setAttribute("visibility", "visible");
    }
    if (x < ml || x > ml + pw) { hide(); return; }
    const i = iAt(x), tc = i + WIN / 2, v = SLIDE[i];
    xh.setAttribute("x1", Xt(tc)); xh.setAttribute("x2", Xt(tc)); xh.setAttribute("visibility", "visible");
    dot.setAttribute("cx", Xt(tc)); dot.setAttribute("cy", Y(v)); dot.setAttribute("visibility", "visible");
    const wx0 = Math.max(ml, Xt(i)), wx1 = Math.min(ml + pw, Xt(i + WIN));      // החלון עצמו מודגש
    win.setAttribute("x", wx0); win.setAttribute("width", Math.max(0, wx1 - wx0)); win.setAttribute("visibility", "visible");
    tip.innerHTML = `<div class="tt"><bdi dir="ltr">${fmt(i)} – ${fmt(i + WIN)}</bdi></div><div class="tv">${v.toFixed(1)} מילים ב-30 שניות</div>`;
    tip.style.display = "block";
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = (e.clientY + 14) + "px";
  });
  function hide() { [xh, dot, win].forEach(el => el.setAttribute("visibility", "hidden")); tip.style.display = "none"; }
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("pointerdown", e => { dragFrom = lx(e); svg.setPointerCapture(e.pointerId); });
  svg.addEventListener("pointerup", e => {
    const x = lx(e), from = dragFrom; dragFrom = null; br.setAttribute("visibility", "hidden");
    if (from == null) return;
    if (Math.abs(x - from) > 6) setRange(tAt(Math.min(from, x)), tAt(Math.max(from, x)));
    else { const i = iAt(x); setRange(i, Math.min(D.dur, i + WIN)); }   // לחיצה = הצגת החלון
  });
}


// עזר משותף: גרירה = בחירת טווח, לחיצה = פעולה, ריחוף = חלונית
function attach(svg, W, ml, pw, mt, ph, a, span, onHover, onClick) {
  const xh = svg.querySelector(".xh"), br = svg.querySelector(".brush");
  const lx = e => { const r = svg.getBoundingClientRect(); return (e.clientX - r.left) * W / r.width; };
  const tAt = x => a + (Math.min(Math.max(x, ml), ml + pw) - ml) / pw * span;
  let dragFrom = null;
  svg.addEventListener("pointermove", e => {
    const x = lx(e);
    if (dragFrom != null) {
      const x0 = Math.max(Math.min(dragFrom, x), ml), x1 = Math.min(Math.max(dragFrom, x), ml + pw);
      br.setAttribute("x", x0); br.setAttribute("width", Math.max(0, x1 - x0)); br.setAttribute("visibility", "visible");
    }
    const h = (x >= ml && x <= ml + pw) ? onHover(tAt(x), x) : null;
    if (!h) { xh && xh.setAttribute("visibility", "hidden"); tip.style.display = "none"; return; }
    if (xh) { xh.setAttribute("x1", h.x); xh.setAttribute("x2", h.x); xh.setAttribute("visibility", "visible"); }
    tip.innerHTML = h.html; tip.style.display = "block";
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = (e.clientY + 14) + "px";
  });
  svg.addEventListener("pointerleave", () => { xh && xh.setAttribute("visibility", "hidden"); tip.style.display = "none"; });
  svg.addEventListener("pointerdown", e => { dragFrom = lx(e); svg.setPointerCapture(e.pointerId); });
  svg.addEventListener("pointerup", e => {
    const x = lx(e), from = dragFrom; dragFrom = null; br.setAttribute("visibility", "hidden");
    if (from == null) return;
    if (Math.abs(x - from) > 6) setRange(tAt(Math.min(from, x)), tAt(Math.max(from, x)));
    else onClick(tAt(x));
  });
}
function playAt(t) { try { audio.currentTime = Math.max(0, t - 1); audio.play(); } catch (_) {} }
function frame(W, H, ml, pw, mt, ph, a, b, yTicks, Y, label) {
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${label}">`;
  for (const [v, txt] of yTicks) {
    s += `<line x1="${ml}" x2="${ml+pw}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)" stroke-width="1"/>`;
    s += `<text x="${ml-6}" y="${Y(v)+4}" text-anchor="end" font-size="11" fill="var(--muted)" style="font-variant-numeric:tabular-nums">${txt}</text>`;
  }
  const Xt = t => ml + (t - a) / (b - a) * pw;
  for (const t of timeTicks(a, b))
    s += `<text x="${Xt(t)}" y="${mt+ph+16}" text-anchor="middle" font-size="11" fill="var(--muted)" style="font-variant-numeric:tabular-nums">${fmt(t)}</text>`;
  return s;
}
const TAIL = (mt, ph) => `<line class="xh" y1="${mt}" y2="${mt+ph}" stroke="var(--muted)" stroke-width="1" visibility="hidden"/>` +
  `<rect class="brush" y="${mt}" height="${ph}" fill="var(--brush)" visibility="hidden"/>`;

// ======== מדדים נוספים: עומס יחסי, זמן תגובה, אירועים ========
// --- 1. מדד עומס יחסי (0 = הרגיל של החניך בגיחה הזו) ---
const ZNAMES = ["גובה צליל", "שונות גובה צליל", "עוצמה", "קצב"];
function drawStress(host) {
  const W = host.clientWidth, H = host.clientHeight, ml = 46, mr = 12, mt = 8, mb = 24;
  const pw = Math.max(50, W - ml - mr), ph = H - mt - mb;
  const [a, b] = range, span = b - a, LIM = 2.5;
  const Xt = t => ml + (t - a) / span * pw;
  const Y = v => mt + ph / 2 - Math.max(-LIM, Math.min(LIM, v)) / LIM * (ph / 2);
  let s = frame(W, H, ml, pw, mt, ph, a, b, [[2, "+2"], [1, "+1"], [0, "רגיל"], [-1, "-1"], [-2, "-2"]], Y, "מדד עומס יחסי");
  s += `<line x1="${ml}" x2="${ml+pw}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--axis)" stroke-width="1.5"/>`;
  const us = D.utts.filter(u => u[3] != null && u[1] > a && u[0] < b);
  // קו: ממוצע נע של 30 שניות (משוקלל לפי משך המשפט); נקטע כשאין דיבור בחלון
  const step = Math.max(1, span / pw * 2), pts = [];
  for (let t = a; t <= b; t += step) {
    let sw = 0, sv = 0;
    for (const u of us) { const ov = Math.min(u[1], t + 15) - Math.max(u[0], t - 15); if (ov > 0) { sw += ov; sv += ov * u[3]; } }
    pts.push(sw >= 1 ? [t, sv / sw] : null);
  }
  for (const u of us) {
    const tc = (u[0] + u[1]) / 2; if (tc < a || tc > b) continue;
    s += `<circle cx="${Xt(tc).toFixed(1)}" cy="${Y(u[3]).toFixed(1)}" r="2.6" fill="var(--series-1-soft, rgba(42,120,214,.3))"/>`;
  }
  let d = "", pen = false;
  for (const p of pts) { if (!p) { pen = false; continue; } d += (pen ? "L" : "M") + Xt(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1); pen = true; }
  s += `<path d="${d}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
  s += TAIL(mt, ph) + `<rect x="${ml}" y="${mt}" width="${pw}" height="${ph}" fill="transparent"/></svg>`;
  host.innerHTML = s;
  attach(host.querySelector("svg"), W, ml, pw, mt, ph, a, span, (t, x) => {
    let best = null, bd = Infinity;
    for (const u of us) { const dd = Math.abs((u[0] + u[1]) / 2 - t); if (dd < bd) { bd = dd; best = u; } }
    if (!best || Math.abs(Xt((best[0] + best[1]) / 2) - x) > 30) return null;
    const z = best[4] || [];
    const parts = ZNAMES.map((n, i) => z[i] == null ? "" : `${n}: ${z[i] > 0 ? "+" : ""}${z[i].toFixed(1)}`).filter(Boolean).join(" · ");
    return { x: Xt((best[0] + best[1]) / 2), html: `<div class="tt">${fmt(best[0])}</div><div class="tv">מדד: ${best[3] > 0 ? "+" : ""}${best[3].toFixed(2)}</div><div>${escapeHtml(best[2])}</div><div class="silent">${parts}</div>` };
  }, playAt);
}

// --- 2. זמן תגובה לשאלות/קריאות ---
function drawResp(host) {
  const W = host.clientWidth, H = host.clientHeight, ml = 46, mr = 12, mt = 14, mb = 24;
  const pw = Math.max(50, W - ml - mr), ph = H - mt - mb;
  const [a, b] = range, span = b - a, MAXL = 6;
  const Xt = t => ml + (t - a) / span * pw;
  const Y = v => mt + ph - Math.min(v, MAXL) / MAXL * ph;
  let s = frame(W, H, ml, pw, mt, ph, a, b, [[0, "0"], [2, "2"], [4, "4"], [6, "6+"]], Y, "זמן תגובה");
  s += `<text x="${ml+pw}" y="${mt-3}" text-anchor="end" font-size="11" fill="var(--muted)">○ = לא נענה</text>`;
  const rs = D.resp.filter(r => r.q && r.t >= a && r.t <= b);
  for (const r of rs) {
    const x = Xt(r.t).toFixed(1);
    if (r.lat == null) s += `<circle cx="${x}" cy="${Y(MAXL)}" r="5" fill="var(--surface-1)" stroke="var(--series-1)" stroke-width="2"/>`;
    else {
      s += `<line x1="${x}" x2="${x}" y1="${Y(0)}" y2="${Y(r.lat)}" stroke="var(--series-1)" stroke-width="2" stroke-linecap="round"/>`;
      s += `<circle cx="${x}" cy="${Y(r.lat)}" r="4" fill="var(--series-1)" stroke="var(--surface-1)" stroke-width="2"/>`;
    }
  }
  s += `<line x1="${ml}" x2="${ml+pw}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--axis)" stroke-width="1"/>`;
  s += TAIL(mt, ph) + `<rect x="${ml}" y="${mt}" width="${pw}" height="${ph}" fill="transparent"/></svg>`;
  host.innerHTML = s;
  attach(host.querySelector("svg"), W, ml, pw, mt, ph, a, span, (t, x) => {
    let best = null, bd = Infinity;
    for (const r of rs) { const dd = Math.abs(Xt(r.t) - x); if (dd < bd) { bd = dd; best = r; } }
    if (!best || bd > 14) return null;
    return { x: Xt(best.t), html: `<div class="tt">${fmt(best.t)} · ${escapeHtml(best.who)}</div>` +
      `<div>${escapeHtml(best.p)}</div>` +
      (best.lat == null ? `<div class="tv">לא נענה (אין תשובה תוך 10 שנ׳)</div>`
                        : `<div class="tv">תגובה אחרי ${best.lat.toFixed(1)} שנ׳</div><div class="silent">החניך: ${escapeHtml(best.r)}</div>`) };
  }, t => { let best = null, bd = Infinity; for (const r of rs) { const dd = Math.abs(r.t - t); if (dd < bd) { bd = dd; best = r; } } playAt(best ? best.t - 2 : t); });
}

// --- 3. אירועים: העברת שליטה / התערבות / תקלה ---
const EV_STYLE = { control: {c: "var(--ev1)", shape: "tri"}, interv: {c: "var(--ev2)", shape: "sq"}, fault: {c: "var(--ev3)", shape: "dot"} };
function marker(shape, x, y, c) {
  if (shape === "tri") return `<path d="M${x} ${y-7}L${x+7} ${y+5}L${x-7} ${y+5}Z" fill="${c}" stroke="var(--surface-1)" stroke-width="1.5"/>`;
  if (shape === "sq") return `<rect x="${x-6}" y="${y-6}" width="12" height="12" rx="2" fill="${c}" stroke="var(--surface-1)" stroke-width="1.5"/>`;
  return `<circle cx="${x}" cy="${y}" r="6" fill="${c}" stroke="var(--surface-1)" stroke-width="1.5"/>`;
}
function drawEvents(host) {
  const W = host.clientWidth, H = host.clientHeight, ml = 46, mr = 12, mt = 6, mb = 24;
  const pw = Math.max(50, W - ml - mr), ph = H - mt - mb;
  const [a, b] = range, span = b - a;
  const Xt = t => ml + (t - a) / span * pw;
  const rows = D.evtypes.map(x => x[0]);
  const Yr = k => mt + (rows.indexOf(k) + 0.5) / rows.length * ph;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="אירועים">`;
  rows.forEach(k => s += `<line x1="${ml}" x2="${ml+pw}" y1="${Yr(k)}" y2="${Yr(k)}" stroke="var(--grid)" stroke-width="1"/>`);
  for (const t of timeTicks(a, b))
    s += `<text x="${Xt(t)}" y="${H-6}" text-anchor="middle" font-size="11" fill="var(--muted)" style="font-variant-numeric:tabular-nums">${fmt(t)}</text>`;
  const ev = D.events.filter(e => e.t >= a && e.t <= b);
  for (const e of ev) { const st = EV_STYLE[e.type]; s += marker(st.shape, Xt(e.t).toFixed(1), Yr(e.type), st.c); }
  s += TAIL(mt, ph) + `<rect x="${ml}" y="${mt}" width="${pw}" height="${ph}" fill="transparent"/></svg>`;
  host.innerHTML = s;
  const names = Object.fromEntries(D.evtypes);
  attach(host.querySelector("svg"), W, ml, pw, mt, ph, a, span, (t, x) => {
    let best = null, bd = Infinity;
    for (const e of ev) { const dd = Math.abs(Xt(e.t) - x); if (dd < bd) { bd = dd; best = e; } }
    if (!best || bd > 14) return null;
    return { x: Xt(best.t), html: `<div class="tt">${fmt(best.t)} · ${escapeHtml(best.who)}</div><div class="tv">${names[best.type]}</div><div>${escapeHtml(best.text)}</div>` };
  }, t => { let best = null, bd = Infinity; for (const e of ev) { const dd = Math.abs(e.t - t); if (dd < bd) { bd = dd; best = e; } } playAt(best ? best.t : t); });
}

// ======== תרגילים ונתוני טיסה מהמאמן ========
const EX = D.ex || [];
const exAt = t => EX.find(e => t >= e[1] && t < e[2]);
const FLM = [
  {key: "alt",   name: "גובה",            unit: "רגל, ברומטרי (מעל פני הים) — כמו שהחניך מדווח", dec: 0, tgt: "alt"},
  {key: "agl",   name: "גובה מעל הקרקע", unit: "רגל",                        dec: 0, tgt: "agl"},
  {key: "spd",   name: "מהירות",          unit: "קשר",                        dec: 0, tgt: "spd"},
  {key: "bank",  name: "הטיה",            unit: "מעלות (חיובי = ימינה)",      dec: 0, signed: true},
  {key: "vs",    name: "מהירות אנכית",    unit: "רגל לדקה (חיובי = עלייה)",   dec: 0, signed: true},
  {key: "stick", name: "פעילות סטיק",     unit: "כמה הסטיק זז (% מהטווח לשנייה) — גבוה = הרבה תיקונים", dec: 0},
].filter(F => D.fl && D.fl[F.key]);
function rangeTicks(lo, hi, n) {
  const step0 = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(step0 || 1)));
  const step = [1, 2, 2.5, 5, 10].map(k => k * mag).find(s => (hi - lo) / s <= n) || 10 * mag;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6)); return out;
}
FLM.forEach(F => {
  const v = D.fl[F.key].filter(x => x != null).sort((a, b) => a - b);
  const q = p => v.length ? v[Math.floor(p * (v.length - 1))] : 0;
  if (F.signed) { const m = Math.max(Math.abs(q(0.005)), Math.abs(q(0.995)), 1) * 1.1; F.lo = -m; F.hi = m; }
  else { F.lo = Math.min(0, q(0.005)); F.hi = Math.max(q(0.995) * 1.1, 1); }
  EX.forEach(e => { if (F.tgt && e[3] && e[3][F.tgt] != null) F.hi = Math.max(F.hi, e[3][F.tgt] * 1.1); });
});
function exBands(a, b, Xt, mt, ph, labels) {
  let s = "";
  EX.forEach((e, i) => {
    if (e[2] <= a || e[1] >= b) return;
    const x0 = Xt(Math.max(e[1], a)), x1 = Xt(Math.min(e[2], b));
    s += `<rect x="${x0}" y="${mt}" width="${Math.max(0, x1 - x0)}" height="${ph}" fill="var(--band${i % 2})"/>`;
    if (e[1] >= a) s += `<line x1="${x0}" x2="${x0}" y1="${mt}" y2="${mt + ph}" stroke="var(--axis)" stroke-width="1" stroke-dasharray="2 3"/>`;
    if (labels && x1 - x0 > 40)
      s += `<text x="${(x0 + x1) / 2}" y="${mt + 12}" text-anchor="middle" font-size="11" fill="var(--text-secondary)">${escapeHtml(e[0].length > (x1 - x0) / 7 ? e[0].slice(0, Math.max(1, Math.floor((x1 - x0) / 7) - 1)) + "…" : e[0])}</text>`;
  });
  return s;
}
function drawFlight(F, host) {
  const W = host.clientWidth, H = host.clientHeight, ml = 46, mr = 12, mt = 8, mb = 24;
  const pw = Math.max(50, W - ml - mr), ph = H - mt - mb;
  const [a, b] = range, span = b - a, st = D.fl.step, arr = D.fl[F.key];
  const nb = Math.max(2, Math.min(Math.round(pw), Math.ceil(span / st)));
  const mean = new Array(nb).fill(null), mn = new Array(nb).fill(null), mx = new Array(nb).fill(null);
  for (let k = 0; k < nb; k++) {
    const i0 = Math.max(0, Math.floor((a + k / nb * span) / st)), i1 = Math.min(arr.length - 1, Math.floor((a + (k + 1) / nb * span) / st));
    let s2 = 0, c = 0, lo = Infinity, hi = -Infinity;
    for (let i = i0; i <= Math.max(i0, i1); i++) { const v = arr[i]; if (v != null) { s2 += v; c++; lo = Math.min(lo, v); hi = Math.max(hi, v); } }
    if (c) { mean[k] = s2 / c; mn[k] = lo; mx[k] = hi; }
  }
  const X = k => ml + (k + 0.5) / nb * pw, Xt = t => ml + (t - a) / span * pw;
  const cl = v => Math.min(Math.max(v, F.lo), F.hi);
  const Y = v => mt + ph - (cl(v) - F.lo) / (F.hi - F.lo) * ph;
  let s = frame(W, H, ml, pw, mt, ph, a, b, rangeTicks(F.lo, F.hi, 4).map(v => [v, v]), Y, F.name);
  s += exBands(a, b, Xt, mt, ph, false);
  // יעדים (קו מקווקו בכל תרגיל שהוגדר לו יעד)
  if (F.tgt) EX.forEach(e => {
    const v = e[3] && e[3][F.tgt]; if (v == null || e[2] <= a || e[1] >= b) return;
    s += `<line x1="${Xt(Math.max(e[1], a))}" x2="${Xt(Math.min(e[2], b))}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--ev2)" stroke-width="2" stroke-dasharray="6 4"/>`;
  });
  if (F.lo < 0) s += `<line x1="${ml}" x2="${ml + pw}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--axis)" stroke-width="1"/>`;
  // טווח מינימום-מקסימום בכל פיקסל + קו ממוצע (נקטע כשאין נתונים)
  let env = "", line = "", open = false, seg = [];
  const flush = () => { if (seg.length) env += "M" + seg.map(k => X(k).toFixed(1) + " " + Y(mx[k]).toFixed(1)).join("L") + "L" + seg.slice().reverse().map(k => X(k).toFixed(1) + " " + Y(mn[k]).toFixed(1)).join("L") + "Z"; seg = []; };
  for (let k = 0; k < nb; k++) {
    if (mean[k] == null) { open = false; flush(); continue; }
    line += (open ? "L" : "M") + X(k).toFixed(1) + " " + Y(mean[k]).toFixed(1); open = true; seg.push(k);
  }
  flush();
  s += `<path d="${env}" fill="var(--series-1-fill)" stroke="none"/>`;
  s += `<path d="${line}" fill="none" stroke="var(--series-1)" stroke-width="${nb > pw * 0.6 ? 1.2 : 2}" stroke-linejoin="round"/>`;
  s += TAIL(mt, ph) + `<rect x="${ml}" y="${mt}" width="${pw}" height="${ph}" fill="transparent"/></svg>`;
  host.innerHTML = s;
  attach(host.querySelector("svg"), W, ml, pw, mt, ph, a, span, t => {
    const k = Math.min(nb - 1, Math.max(0, Math.floor((t - a) / span * nb)));
    const e = exAt(t), u = utterIn(a + k / nb * span, a + (k + 1) / nb * span), v = mean[k];
    const tg = F.tgt && e && e[3] && e[3][F.tgt];
    return {x: X(k), html: `<div class="tt">${fmtF(t)}</div>` +
      `<div class="tv">${F.name}: ${v == null ? "אין נתונים" : v.toFixed(F.dec) + " " + F.unit.split(/[ ,(—]/)[0]}</div>` +
      (tg != null ? `<div>יעד: ${tg}</div>` : "") +
      (e ? `<div>תרגיל: ${escapeHtml(e[0])}</div>` : "") +
      (u ? `<div class="silent">החניך: ${escapeHtml(u[2])}</div>` : "")};
  }, playAt);
}
function drawVsync(host) {
  const W = host.clientWidth, H = host.clientHeight, ml = 46, mr = 12, mt = 8, mb = 24;
  const pw = Math.max(50, W - ml - mr), ph = H - mt - mb, [a, b] = range, span = b - a, V = D.vsync;
  let lim = 10; V.video.concat(V.csv).forEach(v => { if (v != null) lim = Math.max(lim, Math.abs(v)); });
  lim = Math.min(90, lim * 1.1);
  const Xt = t => ml + (t - a) / span * pw, Y = v => mt + ph - (Math.max(-lim, Math.min(lim, v)) + lim) / (2 * lim) * ph;
  let s = frame(W, H, ml, pw, mt, ph, a, b, rangeTicks(-lim, lim, 4).map(v => [v, v]), Y, "בדיקת סנכרון");
  s += `<line x1="${ml}" x2="${ml + pw}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--axis)" stroke-width="1"/>`;
  let line = "", open = false;
  for (let k = 0; k < V.t.length; k++) {
    const t = V.t[k], v = V.csv[k];
    if (t < a || t > b || v == null) { open = false; continue; }
    line += (open ? "L" : "M") + Xt(t).toFixed(1) + " " + Y(v).toFixed(1); open = true;
  }
  s += `<path d="${line}" fill="none" stroke="var(--series-1)" stroke-width="1.5" stroke-linejoin="round"/>`;
  for (let k = 0; k < V.t.length; k++) {
    const t = V.t[k]; if (t < a || t > b) continue;
    s += `<circle cx="${Xt(t).toFixed(1)}" cy="${Y(V.video[k]).toFixed(1)}" r="2.2" fill="var(--ev2)"/>`;
  }
  s += TAIL(mt, ph) + `<rect x="${ml}" y="${mt}" width="${pw}" height="${ph}" fill="transparent"/></svg>`;
  host.innerHTML = s;
  attach(host.querySelector("svg"), W, ml, pw, mt, ph, a, span, t => {
    const k = Math.min(V.t.length - 1, Math.max(0, lowerBound(V.t, t)));
    return {x: Xt(V.t[k]), html: `<div class="tt">${fmtF(V.t[k])}</div><div class="tv">גלגול בסרטון: ${V.video[k]}°</div>` +
      `<div>גלגול ב-CSV: ${V.csv[k] == null ? "—" : V.csv[k] + "°"}</div>`};
  }, playAt);
}
function drawExStrip(host) {
  const W = host.clientWidth, H = host.clientHeight, ml = 46, mr = 12, mt = 4, ph = H - 26;
  const pw = Math.max(50, W - ml - mr), [a, b] = range, span = b - a, Xt = t => ml + (t - a) / span * pw;
  let s = frame(W, H, ml, pw, mt, ph, a, b, [], () => 0, "תרגילים");
  s += exBands(a, b, Xt, mt, ph, true);
  s += TAIL(mt, ph) + `<rect x="${ml}" y="${mt}" width="${pw}" height="${ph}" fill="transparent"/></svg>`;
  host.innerHTML = s;
  attach(host.querySelector("svg"), W, ml, pw, mt, ph, a, span, t => {
    const e = exAt(t);
    return {x: Xt(t), html: `<div class="tt">${fmt(t)}</div>` + (e ? `<div class="tv">${escapeHtml(e[0])}</div><div class="silent"><bdi dir="ltr">${fmt(e[1])} – ${fmt(e[2])}</bdi> · לחיצה מציגה את התרגיל</div>` : `<div class="silent">בין תרגילים</div>`)};
  }, t => { const e = exAt(t); if (e) setRange(e[1], e[2]); else playAt(t); });
}

function render() {
  const [a, b] = range;
  document.getElementById("rangeLabel").textContent =
    (a <= 0 && b >= D.dur) ? "מוצגת כל הגיחה" : "מוצג: \u2066" + fmt(a) + " – " + fmt(b) + "\u2069";
  const st = stats(a, b);
  const pct = Math.round(100 * st.talk / Math.max(b - a, 1e-6));
  document.getElementById("tiles").innerHTML =
    `<div class="tile"><div class="k">זמן דיבור של החניך</div><div class="v">${Math.round(st.talk)} <span class="u">שנ׳ (${pct}% מהטווח, ${st.n} משפטים)</span></div></div>` +
    METRICS.map(M => `<div class="tile"><div class="k">${M.name} — ממוצע בזמן דיבור</div><div class="v">${st[M.key] == null ? "—" : st[M.key].toFixed(M.dec)} <span class="u">${M.unit}</span></div></div>`).join("") +
    (() => {
      const qs = D.resp.filter(r => r.q && r.t >= a && r.t <= b), ans = qs.filter(r => r.lat != null).map(r => r.lat).sort((x, y) => x - y);
      const med = ans.length ? ans[Math.floor((ans.length - 1) / 2)] : null;
      const ev = D.events.filter(e => e.t >= a && e.t <= b), cnt = k => ev.filter(e => e.type === k).length;
      const idx = D.utts.filter(u => u[3] != null && u[1] > a && u[0] < b);
      const tw = idx.reduce((s2, u) => s2 + (u[1] - u[0]), 0), mi = tw ? idx.reduce((s2, u) => s2 + u[3] * (u[1] - u[0]), 0) / tw : null;
      return `<div class="tile"><div class="k">זמן תגובה חציוני לשאלות</div><div class="v">${med == null ? "—" : med.toFixed(1)} <span class="u">שנ׳ (${qs.length - ans.length} מתוך ${qs.length} לא נענו)</span></div></div>` +
        `<div class="tile"><div class="k">העברות שליטה / התערבויות</div><div class="v">${cnt("control")} / ${cnt("interv")} <span class="u">(תקלות/תרחישים: ${cnt("fault")})</span></div></div>` +
        (D.fl ? (() => {
          const st2 = D.fl.step, i0 = Math.floor(a / st2), i1 = Math.ceil(b / st2);
          const g = k => { const v = (D.fl[k] || []).slice(i0, i1 + 1).filter(x => x != null); return v; };
          const al = g("alt"), sp = g("spd"), bk = g("bank");
          const avg = v => v.length ? v.reduce((x, y) => x + y, 0) / v.length : null;
          const sd = v => { const m = avg(v); return v.length ? Math.sqrt(v.reduce((x, y) => x + (y - m) ** 2, 0) / v.length) : null; };
          const e = EX.filter(x => x[2] > a && x[1] < b);
          return `<div class="tile"><div class="k">גובה (ברומטרי) — ממוצע</div><div class="v">${al.length ? Math.round(avg(al)) : "—"} <span class="u">רגל (±${al.length ? Math.round(sd(al)) : "—"})</span></div></div>` +
                 `<div class="tile"><div class="k">מהירות — ממוצע</div><div class="v">${sp.length ? Math.round(avg(sp)) : "—"} <span class="u">קשר (±${sp.length ? sd(sp).toFixed(1) : "—"})</span></div></div>` +
                 `<div class="tile"><div class="k">הטיה מקסימלית</div><div class="v">${bk.length ? Math.round(Math.max(...bk.map(Math.abs))) : "—"}<span class="u">°${e.length === 1 ? " · " + escapeHtml(e[0][0]) : ""}</span></div></div>`;
        })() : "") +
        `<div class="tile"><div class="k">מדד עומס יחסי — ממוצע</div><div class="v">${mi == null ? "—" : (mi > 0 ? "+" : "") + mi.toFixed(2)} <span class="u">(0 = הרגיל שלו)</span></div></div>`;
    })();
  const box = document.getElementById("charts");
  if (!box.children.length)
    box.innerHTML =
      (EX.length ? `<div class="card"><h2>ציר התרגילים <span class="u">(לחיצה על תרגיל מציגה אותו)</span></h2><div class="chart" id="exstrip" style="height:56px"></div></div>` : "") +
      (FLM.length ? `<h2 class="sec">נתוני טיסה מהמאמן</h2>` + (D.sync ? `<p class="u" style="margin:-4px 0 10px">${escapeHtml(D.sync)}</p>` : "") +
        (D.vsync ? `<div class="card"><h2>בדיקת סנכרון <span class="u">גלגול: קו = מה-CSV אחרי ההזזה, נקודות = נמדד מקו האופק בסרטון. חפיפה של הנקודות לקו = סנכרון נכון</span></h2><div class="chart" id="vsync"></div></div>` : "") + FLM.map((F, i) => `<div class="card"><h2>${F.name} <span class="u">${F.unit}</span></h2><div class="chart" id="f${i}"></div></div>`).join("") + `<h2 class="sec">קול החניך</h2>` : "") +
      `<div class="card"><h2>מילים ב-30 שניות <span class="u">(החניך, חלון נע בצעדים של שנייה)</span></h2><div class="chart" id="slide"></div></div>` +
      METRICS.map((M, i) => `<div class="card"><h2>${M.name} <span class="u">${M.unit}</span></h2><div class="chart" id="c${i}"></div></div>`).join("") +
      `<div class="card"><h2>מדד עומס יחסי <span class="u">(שילוב גובה צליל, שונות, עוצמה וקצב ביחס לרגיל של החניך בגיחה; קו = ממוצע 30 שנ׳, נקודה = משפט)</span></h2><div class="chart" id="stress"></div></div>` +
      `<div class="card"><h2>זמן תגובה לשאלות וקריאות <span class="u">(שניות מסוף השאלה של המדריכה/הממיין עד שהחניך מתחיל לענות)</span></h2><div class="chart" id="resp"></div></div>` +
      `<div class="card"><h2>אירועים</h2><div class="legend">${D.evtypes.map(([k, n]) => `<span style="display:inline-flex;align-items:center;gap:4px"><svg width="16" height="16" viewBox="-8 -8 16 16">${marker(EV_STYLE[k].shape, 0, 0, EV_STYLE[k].c)}</svg>${n}</span>`).join("")}</div><div class="chart" id="events" style="height:110px"></div></div>`;
  if (EX.length) drawExStrip(document.getElementById("exstrip"));
  if (D.vsync) drawVsync(document.getElementById("vsync"));
  FLM.forEach((F, i) => drawFlight(F, document.getElementById("f" + i)));
  drawSlide(document.getElementById("slide"));
  METRICS.forEach((M, i) => drawChart(M, document.getElementById("c" + i)));
  drawStress(document.getElementById("stress"));
  drawResp(document.getElementById("resp"));
  drawEvents(document.getElementById("events"));
  document.getElementById("tbody").innerHTML = st.us.map(u => {
    const i0 = lowerBound(D.t, u[0]), i1 = lowerBound(D.t, u[1] + 1e-9);
    const avg = key => { let s = 0, c = 0; for (let i = i0; i < i1; i++) { const v = D[key][i]; if (v != null) { s += v; c++; } } return c ? s / c : null; };
    const v = avg("vol"), f = avg("f0"), r = avg("rate");
    return `<tr><td class="num">${fmt(u[0])}</td><td class="num">${(u[1]-u[0]).toFixed(1)}</td><td class="num">${v==null?"":v.toFixed(1)}</td><td class="num">${f==null?"":f.toFixed(0)}</td><td class="num">${r==null?"":r.toFixed(1)}</td><td>${escapeHtml(u[2])}</td></tr>`;
  }).join("");
}

function setRange(a, b) {
  a = Math.max(0, a); b = Math.min(D.dur, b);
  if (b - a < 3) b = Math.min(D.dur, a + 3);
  range = [a, b];
  document.getElementById("from").value = fmt(a);
  document.getElementById("to").value = fmt(b);
  render();
}
document.getElementById("apply").onclick = () => {
  const err = document.getElementById("err"); err.textContent = "";
  const A0 = parseT(document.getElementById("from").value), B0 = parseT(document.getElementById("to").value);
  if (Number.isNaN(A0) || Number.isNaN(B0)) { err.textContent = "פורמט זמן: דקות:שניות (למשל 12:30) או מספר דקות (12)"; return; }
  const A = A0 ?? 0, B = B0 ?? D.dur;
  if (B <= A) { err.textContent = "'עד דקה' צריך להיות אחרי 'מדקה'"; return; }
  setRange(A, B);
};
["from","to"].forEach(id => document.getElementById(id).addEventListener("keydown", e => { if (e.key === "Enter") document.getElementById("apply").click(); }));
document.getElementById("all").onclick = () => {
  range = [0, D.dur]; document.getElementById("from").value = ""; document.getElementById("to").value = "";
  document.getElementById("err").textContent = ""; render();
};
document.getElementById("smooth").onchange = render;
document.addEventListener("click", e => {
  const l = e.target.closest && e.target.closest("a.exl"); if (!l) return;
  e.preventDefault();
  setRange(parseFloat(l.dataset.a), parseFloat(l.dataset.b));
  document.getElementById("tiles").scrollIntoView({behavior: "smooth", block: "start"});
});
document.querySelectorAll(".summary, .card").forEach(el => el.addEventListener("click", e => {
  const a = e.target.closest("a.ts"); if (!a) return;
  e.preventDefault();
  const t = parseFloat(a.dataset.t);
  setRange(Math.max(0, t - 30), Math.min(D.dur, t + 30));
  document.getElementById("tiles").scrollIntoView({behavior: "smooth", block: "start"});
}));

// ---- זום והזזה ----
const MIN_SPAN = 3;   // הכי קרוב: 3 שניות על כל רוחב הגרף
let raf = 0;
function setRangeSoon(a, b) {             // מקבץ כמה תנועות גלגלת לציור אחד
  cancelAnimationFrame(raf);
  raf = requestAnimationFrame(() => setRange(a, b));
}
function zoomAt(tc, factor) {
  const [a, b] = range, span = b - a;
  const ns = Math.min(D.dur, Math.max(MIN_SPAN, span * factor));
  const f = (tc - a) / span;               // המקום של העכבר נשאר במקום
  let na = tc - f * ns, nb = na + ns;
  if (na < 0) { nb -= na; na = 0; }
  if (nb > D.dur) { na -= nb - D.dur; nb = D.dur; na = Math.max(0, na); }
  if (ns >= D.dur - 0.01) { na = 0; nb = D.dur; }
  range = [na, nb]; setRangeSoon(na, nb);
}
function panBy(frac) {
  const [a, b] = range, span = b - a;
  let na = a + span * frac, nb = b + span * frac;
  if (na < 0) { nb -= na; na = 0; }
  if (nb > D.dur) { na -= nb - D.dur; nb = D.dur; }
  range = [na, nb]; setRangeSoon(na, nb);
}
document.getElementById("charts").addEventListener("wheel", e => {
  const svg = e.target.closest && e.target.closest("svg");
  if (!svg || !(e.ctrlKey || e.metaKey || e.shiftKey)) return;   // גלילה רגילה = גלילת הדף
  e.preventDefault();
  if (e.shiftKey && !(e.ctrlKey || e.metaKey)) { panBy((e.deltaY || e.deltaX) > 0 ? 0.15 : -0.15); return; }
  const r = svg.getBoundingClientRect(), ml = 46, mr = 12;
  const x = Math.min(Math.max(e.clientX - r.left, ml), r.width - mr);
  const [a, b] = range, tc = a + (x - ml) / (r.width - ml - mr) * (b - a);
  zoomAt(tc, e.deltaY < 0 ? 0.7 : 1 / 0.7);
}, {passive: false});
document.getElementById("zin").onclick = () => zoomAt((range[0] + range[1]) / 2, 0.5);
document.getElementById("zout").onclick = () => zoomAt((range[0] + range[1]) / 2, 2);
document.getElementById("pleft").onclick = () => panBy(-0.5);
document.getElementById("pright").onclick = () => panBy(0.5);
window.addEventListener("resize", render);
render();
</script>
</body>
</html>
"""

page = (page.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
            .replace("__TITLE__", html.escape(title_name))
            .replace("__SUMMARY__", summary_html)
            .replace("__LLM__", llm_html)
            .replace("__EXER__", ex_html)
            .replace("__END__", f"{int(dur // 60):02d}:{int(dur % 60):02d}")
            .replace("__AUDIO__", html.escape(audio_src)))

with open(OUT, "w", encoding="utf-8") as f:
    f.write(page)
print(f"נוצר: {OUT}")
webbrowser.open("file:///" + OUT.replace("\\", "/"))

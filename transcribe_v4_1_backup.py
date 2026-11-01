# -*- coding: utf-8 -*-
"""
תמלול גיחות סימולטור + זיהוי דוברים (חניך / מדריכה / ממיין) - גרסה 2

מה השתנה לעומת הגרסה הקודמת:
1. מודל ה-LLM הישן (claude-3-5-sonnet-20241022) כבר לא זמין ב-API, ולכן שלב
   הסיווג הסמנטי נפל בכל ריצה בשקט ועבר ל"הכרעה התנהגותית" (ספירת מקטעים) -
   זו הסיבה העיקרית שמדריכה/ממיין התבלבלו. הוחלף למודל עדכני + בדיקת API key.
2. אמבדינג קולי לפי חלונות של 1.5 שניות (ולא לפי מקטע Whisper שלם) + חותמות
   זמן למילים. מקטע Whisper אחד מכיל לפעמים שני דוברים, ואז הקול "מתערבב".
   עכשיו כל מילה מקבלת דובר בנפרד, ומקטע יכול להתפצל לשני דוברים.
3. אשכול בשני שלבים: (א) 3 אשכולות -> החניך = הכי הרבה זמן דיבור.
   (ב) את כל השאר מאשכלים מחדש ל-2, אחרי "מירכוז" האמבדינגים (הורדת הממוצע
   המשותף - בעיקר מאפייני הערוץ/אינטרקום שזהים לשניהם). כך ההבדל בין שני
   הקולות הדומים בולט יותר.
4. הכרעת תפקידים משלבת שלושה סימנים:
   - טקסט (LLM) על הדיאלוג המלא בסדר כרונולוגי, כולל משפטי החניך כהקשר.
   - גובה צליל אמיתי (F0) - לא Spectral Centroid. אם המדריכה אישה והממיין גבר
     זה סימן חזק מאוד (ניתן לכבות ב-USE_GENDER_PRIOR).
   - התנהגות: מי מדבר יותר פעמים, ומי מדבר יותר בסוף (סיכום/משוב).
5. דגל "לבדיקה ידנית" כשהסימנים סותרים, ומתג FORCE_SWAP_ROLES להחלפה ידנית.
6. תיקון ה-HTML: הצבעים לא עבדו כי השמות בקוד לא תאמו לשמות התפקידים.
"""

SCRIPT_VERSION = "v4.1 - השלמת משפטים שקטים (הגברה + סינון הזיות) (11/10)"
print(f"=== transcribe_v3 | {SCRIPT_VERSION} ===")
print("טוען ספריות (torch / whisper / speechbrain) - זה לוקח חצי דקה עד דקה...", flush=True)

import html
import json
import os
import re
import sys
import warnings
import webbrowser
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torchaudio
from faster_whisper import WhisperModel
from sklearn.cluster import SpectralClustering
# librosa חייב להיטען *לפני* speechbrain: ה-lazy-loading של librosa סורק את
# sys.modules, ונתקע על מודולים "עצלים" של speechbrain (k2) -> ImportError.
try:
    import librosa
    from librosa import pyin as _pyin_preload  # noqa: F401  טעינה מלאה עכשיו

    HAS_LIBROSA = True
except Exception as _e:  # noqa: BLE001
    HAS_LIBROSA = False
    print(f"[אזהרה] librosa לא זמין ({_e}) - סימן גובה הצליל כבוי. התקנה: pip install librosa")

from speechbrain.inference.speaker import EncoderClassifier

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"



# ==========================================
# 1. הגדרות
# ==========================================
AUDIO_PATH = r"d:\Ofir\VoiceSoftware\check 1 tavlan and carmel.mp3"
HTML_OUTPUT = r"d:\Ofir\VoiceSoftware\transcription_result.html"
EXCEL_OUTPUT = r"d:\Ofir\VoiceSoftware\transcription_result.xlsx"
MODEL_SAVE_DIR = r"d:\Ofir\VoiceSoftware\speechbrain_model"

# מנוע LLM לסיווג סמנטי (אופציונלי - יש גם סיווג מקומי לפי מילון ודקדוק):
#   "none"      - בלי LLM. הכל מקומי וחינמי.
#   "ollama"    - מודל שרץ מקומית על המחשב (חינם, בלי אינטרנט). דורש התקנת Ollama.
#   "anthropic" - Claude דרך API (בתשלום, דורש ANTHROPIC_API_KEY, שולח טקסט החוצה).
LLM_BACKEND = "none"
LLM_MODEL = "claude-sonnet-5-5"   # ל-anthropic
OLLAMA_MODEL = "qwen2.5:7b"       # ל-ollama (אפשר גם qwen2.5:14b אם יש GPU חזק)
OLLAMA_URL = "http://localhost:11434/api/generate"

# True = מניחים שהמדריכה בעלת קול גבוה יותר (אישה) מהממיין.
# אם בגיחה מסוימת זה לא נכון (מדריך גבר / ממיינת אישה) - שנה ל-False.
USE_GENDER_PRIOR = True

# אם אחרי בדיקה ידנית התפקידים יצאו הפוכים - שנה ל-True והרץ שוב
FORCE_SWAP_ROLES = False

# משקלות לשילוב הסימנים (מנורמלים אוטומטית לפי מה שזמין)
# W_LEX = מילון ביטויים + דקדוק זכר/נקבה - עובד גם בלי API key.
# W_BEHAVIOR = 0: "מי מדבר יותר" התגלה כלא אמין (בהקלטת tavlan & carmel הממיין
# דיבר פי 6 מהמדריכה) - נשאר רק להדפסה.
W_TEXT, W_LEX, W_PITCH, W_BEHAVIOR = 0.45, 0.35, 0.20, 0.0
REVIEW_MARGIN = 0.15  # אם ההסתברות קרובה ל-0.5 בפחות מזה -> לבדיקה ידנית

# מודל התמלול:
#   "ivrit-ai/whisper-large-v3-turbo-ct2" - מאומן במיוחד על עברית, מהיר יותר (מומלץ)
#   "large-v3"                            - המודל הקודם, למקרה שתרצה להשוות
WHISPER_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"

# השלמת משפטים קצרים ש-Whisper דילג עליהם (למשל "נסק שחרר" בתחילת ההקלטה)
RECOVER_GAPS = True
GAP_TOP_DB = 35          # כמה חזק צריך להיות קטע (ביחס לשיא) כדי להיחשב "יש קול"
GAP_MIN_SEC = 0.4        # קטע קצר מזה לא נבדק
GAP_MAX_COUNT = 60       # מקסימום קטעים להשלמה (זמן ריצה)
GAP_MIN_LOGPROB = -1.0   # תמלול פחות בטוח מזה נזרק (כנראה רעש). להחמיר: -0.8
GAP_MAX_GAIN = 20.0      # הגברה מקסימלית לקטע שקט

# סינון שקט לפני תמלול. אם נראה שמשפטים שקטים (קשר חלש) נעלמים - שנה ל-False
USE_VAD = False  # בהרצה על check 1 עם True נעלמו ~17 דקות (04:54-22:03)

# אשכול שמקבל פחות מ-4% מהמילים נחשב "זבל" (רעש), לא דובר
MIN_SPEAKER_SHARE = 0.04
# פיצול אשכול לשני דוברים מתקבל רק אם החצאים שונים מספיק:
SPLIT_MAX_COS = 0.70      # דמיון קולי נמוך מזה = שני אנשים
SPLIT_MIN_F0_DIFF = 15.0  # או הפרש גובה צליל של 15Hz ומעלה

# פרמטרים לחלונות קוליים
WIN_SEC = 1.5
HOP_SEC = 0.75
MIN_WIN_SEC = 0.4
MIN_RUN_SEC = 0.6  # רצף מילים קצר מזה לא ייחשב כהחלפת דובר (מונע "קפיצות")

ROLE_TRAINEE = "חניך"
ROLE_EVALUATOR = "ממיין"
ROLE_INSTRUCTOR = "מדריכה"
ROLE_SIM = "קולות רקע / סימולטור"

FIX_DICTIONARY = {  # החלפה גם בתוך מילה (התנהגות קודמת)
    "לקוון": "לכוון",
    "הפרוני": "עפרוני",
    "אפרוני": "עפרוני",
    "האוטובוס": "האוטורוטציה",
    "בקיוון": "בכיוון",
    "קיוון": "כיוון",
}

# ---- אותות קריאה בגיחה הזו (לעדכן בכל גיחה!) ----
# משמשים גם ב-prompt של Whisper וגם לתיקון אוטומטי: מילה שדומה מאוד לאות קריאה
# ("שרלו", "שירלוק", "שרלופ") או שבאה לפני "מ<אות קריאה>" ("שירת מתקיף") מתוקנת.
CALLSIGNS = ["שרלוק", "תקיף", "ברזלים"]  # ברירת מחדל - נדרס ע"י SORTIE_SETTINGS
PROMPT_TYPE = "tactical"                   # ברירת מחדל - נדרס ע"י SORTIE_SETTINGS

# הגדרות לפי הקלטה: אם חלק מהשם מופיע בשם הקובץ - משתמשים בהגדרות שלו.
# "tactical" = גיחת תקיפה/משימה, "maneuvers" = שמינייה/תבניות/אוטורוטציה.
# CALLSIGNS ריק = בלי תיקון אוטומטי של אותות קריאה (עדיף מאשר אותות של גיחה אחרת!)
SORTIE_SETTINGS = {
    "check 1": {"prompt": "maneuvers", "callsigns": []},
    "check 5": {"prompt": "tactical", "callsigns": ["שרלוק", "תקיף", "ברזלים"]},
    "check 6": {"prompt": "tactical", "callsigns": ["שרלוק", "תקיף", "ברזלים"]},
}
CALLSIGN_SIMILARITY = 0.75  # כמה דומה מילה צריכה להיות כדי להיות מתוקנת (0-1)
CALLSIGN_STOPWORDS = {"שלוש", "שלום", "שלושה", "שלושים", "ברזל"}  # מילים רגילות שלא לגעת בהן

# החלפה רק של מילה/צירוף שלם (לא בתוך מילה אחרת).
# נבנה מתמלולי check 1 + check 6. אם תיקון מסוים שגוי אצלכם - פשוט מחק את השורה.
FIX_WORDS = {
    # --- אות קריאה / קשר (check 6) - לעדכן לפי הגיחה ---
    "שרלו": "שרלוק", "שלוק": "שרלוק", "שרוק": "שרלוק",
    "מתעקיף": "מתקיף", "מתעקיס": "מתקיף", "מטקיף": "מתקיף", "מתקיך": "מתקיף", "כמתקיך": "מתקיף",
    "קבלתי": "קיבלתי", "קיבל": "קיבלתי",
    "אישה עם ראשי": "האם רשאי", "עם ראשי": "האם רשאי",
    "נון צדיק": 'נ"צ', "נונצלי": 'נ"צ', "בנונצלי": 'בנ"צ',
    "קדמם": 'כטמ"ם', "קטמם": 'כטמ"ם', "הקדמם": 'הכטמ"ם',
    "היום חוסל": "האיום חוסל", "חוסלות": "חוסלה",
    "מתקלנו": "נתקלנו", "הותקלנו": "נתקלנו", "הודקלנו": "נתקלנו", "הוצגלנו": "נתקלנו",
    "התעקלות": "היתקלות",
    "המצנה": "המתנה",
    "נגולת": "נקודת", "הנגודה": "הנקודה",
    "עמדת ריצה": "עמדת ירי", "פרצים": "פרטים",
    "מונושן": "רימון עשן", "להגה": "להגעה", "יחצפו": "ייחשפו",
    "ננקק": "ננקה", "אשכר": "אשחרר", "פיצוע": "ביצוע", "נחלות": "חלות",
    "סרבו": "סרוו", "משבת": "משאבת", "לחבוץ": "לכבות", "לצבח": "לטווח",
    # --- שטח ---
    "ער": "הר", "לער": "להר", "הער": "ההר",
    "הרכז": "הרכס", "מהרכז": "מהרכס", "הרכיב הזה": "הרכס הזה",
    "העיץ": "העץ", "עיץ": "עץ", "העת שלנו": "העץ שלנו",
    # --- הטסה ---
    "פוצח": "פותח", "סקור": "סוגר", "מחניס": "מכניס", "מתקר": "מתקן", "מזויק": "מדויק",
    "הטיר": "הציר", "לצוס": "לטוס", "לתוס": "לטוס", "תס": "טס",
    "מעיץ": "מאיץ", "מעיצ": "מאיץ",
    "מתרקן": "מתארגן", "בחדש": "מחדש", "לצחור": "לצבור",
    "בהנמחה": "בהנמכה", "בהנבחה": "בהנמכה",
    "תא אף": "את האף", "ת'אף": "את האף", "טאף": "את האף", "תאף": "את האף", "מרימאף": "מרים אף",
    "בעילוצים": "באילוצים", "מילוצים": "אילוצים", "מהילוצים": "מהאילוצים",
    "מיירות": "מהירות", "המיירות": "המהירות", "מהירוט": "מהירות", "מהיות": "מהירות",
    "מהירוד": "מהירות", "מיעוט": "מהירות", "גובר": "גובה", "נרוך": "נמוך",
    "בקרא": "בקרה", "תיחסים לב": "תשים לב", "שעקח": "שלקח", "מוודאי": "מוודא",
    # --- הורדת/הרמת אף וכוח (משוב מהרצות) ---
    "מאידך": "מוריד אף", "ומאידך": "ומוריד אף",
    "מאועד": "מוריד", "ראויה כוח": "מוריד כוח",
    "מועדף": "מוריד אף", "מיעד אף": "מוריד אף", "מורד אף": "מוריד אף",
    "מועצת אף": "מוריד אף", "מועצת האף": "מוריד את האף",
    "ראוי כוח": "מוריד כוח",
    "סם כוח": "שם כוח", "סתם כוח": "שם כוח", "סתם-כוח": "שם כוח",
    "מצבה אף": "מצב אף",
    "הטעיה": "הטייה", "הטעייה": "הטייה", "והטעיה": "והטייה", "לטעיה": "להטייה",
    "שירת מתקיף": "שרלוק מתקיף", "שורת מתקיף": "שרלוק מתקיף",
    # --- שחרור הסימולטור ---
    "מתארגן שכם": "נסק שחרר", "מסק שכן": "נסק שחרר", "נסק שכן": "נסק שחרר",
    "מסק שחרר": "נסק שחרר", "נסק שכם": "נסק שחרר",
    # --- check 1 (שמינייה) ---
    "נשיקה": "נסיקה", "נסיכה": "נסיקה",
    "התאיה": "הטייה", "התאייה": "הטייה", "תאיה": "הטייה", "תאייה": "הטייה", "לטיה": "להטייה",
    "מועד": "מוריד",
    "שמיניה": "שמינייה", "השמיניה": "השמינייה", "לשמניה": "לשמינייה",
    "תכיר": "תחקיר", "תחקור": "תחקיר", "קורדינציה": "קואורדינציה",
}

_HEB = "\u05d0-\u05ea"
_FIX_WORDS_RE = [
    (re.compile(rf"(?<![{_HEB}]){re.escape(k)}(?![{_HEB}])"), v)
    for k, v in sorted(FIX_WORDS.items(), key=lambda kv: -len(kv[0]))  # ארוכים קודם
]
# "נסק שחרר" (שחרור הסימולטור) - Whisper שומע את "נסק" בהרבה צורות: "מתארגן", "מסק", "מסך"...
_RELEASE_RE = re.compile(
    rf"(?<![{_HEB}])(?:מתארגן|מתארגנת|מתארגנים|מסק|נסק|נאסק|מסך|נסך|נסט|מסט|נשק)[,.]?\s+(?:שחרר|שחררי|שכם|שכן|שחרור)(?![{_HEB}])"
)
# חזרות של Whisper ("קולקטיב, קולקטיב, קולקטיב..." / "נכון, אני איתך, נכון, אני איתך...")
_REPEAT_RE = re.compile(r"((?:[^\s,.?!]+[,.?!]?\s*){1,5}?)\1{2,}")

ROLE_DESCRIPTION = """
הקלטה מסשן בסימולטור טיסה. בסשן שלושה תפקידים:
- חניך: הטייס שמבצע את התרגיל (כבר זוהה, מסומן "חניך").
- מדריכה (אישה): מפעילת הסימולטור מעמדה. משחקת את גורמי הקשר - מגדל/בקר
  ותנועה אווירית - ולכן מדברת בשפת קשר קצרה ("קיבלתי", "דווח ב...",
  "תנועה מתקרבת מ...", "כנס לאזור", "שחרר"). מזריקה תרחישים ותקלות,
  משחררת/מקפיאה/עוצרת את הסימולטור ("אני עוצרת אתכם").
- ממיין (גבר): יושב ליד החניך ומעריך אותו. שואל שאלות ("למה?", "איזה מהירות
  אתה רוצה?", "איפה אתה מזהה אותו?", "עוד דברים?"), מעיר ונותן משוב - גם תוך
  כדי הטיסה, לא רק בסוף. הוא זה שמבקש מהמדריכה להתחיל/לעצור ("את יכולה
  לשחרר אותנו").
רמז חשוב: בעברית המין הדקדוקי מסגיר את הדובר - "אני שואל" = גבר (ממיין),
"אני עוצרת" = אישה (מדריכה). מי שפונה ל"את" מדבר אל המדריכה, כלומר הוא לא המדריכה.
"""

# ---- מילון ביטויים לסיווג טקסטואלי בלי LLM ----
INSTRUCTOR_PATTERNS = [
    r"קיבלתי", r"\bדווח", r"תנועה", r"שחרר", r"\bכנס\b", r"\bמגדל", r"מאושר",
    r"\bאישור", r"\bעבור\b", r"עוצרת", r"משחררת", r"מקפיא", r"מזריק", r"תקלה",
    r"יכולים לצאת",
]
EVALUATOR_PATTERNS = [
    r"\bלמה\b", r"\bאיזה\b", r"\bאיזו\b", r"\bכמה\b", r"\bאיפה\b", r"אתה זוכר",
    r"אתה רואה", r"אתה רוצה", r"\bתגדיר", r"תחקיר", r"\bתחקר", r"עוד דברים",
    r"\bהבנתי\b", r"סיימנו",
]
# ניסוחים אופייניים לחניך (הטייס) - משמש לבחירת אשכול החניך
TRAINEE_PATTERNS = [
    r"(?:מרים|מוריד|מעלה)\s+(?:את\s+ה|טיפה\s+)?אף", r"(?:סוגר|פותח|מרים|מוריד)\s+(?:את\s+ה|ב)?קולקטיב",
    r"מכניס\s+רגל", r"מתיישר", r"מיישר", r"מנמיך", r"מטפס", r"מאיץ", r"מאט",
    r"\bגובה\s+\d", r"\bמהירות\s+\d", r"\bכיוון\s+\d", r"\bכאן\s+\S+", r"\bאני\s+מ\S+",
]
FEMALE_ADDRESS = r"\bאת\s+(?:יכולה|רוצה|צריכה|מוכנה|שומעת|רואה)\b"  # פונה למדריכה

# בחירת קובץ: (1) נתיב בשורת הפקודה  python transcribe.py "קובץ.mp3"
#              (2) חלון בחירת קובץ (ASK_FOR_FILE = True)
#              (3) AUDIO_PATH שלמעלה
ASK_FOR_FILE = True
if len(sys.argv) > 1:
    AUDIO_PATH = sys.argv[1]
elif ASK_FOR_FILE:
    try:
        import tkinter as tk
        from tkinter import filedialog

        _root = tk.Tk()
        _root.withdraw()
        _root.attributes("-topmost", True)  # שהחלון לא ייפתח מאחורי חלונות אחרים
        print(">>> בחר את קובץ ההקלטה בחלון שנפתח <<<", flush=True)
        _picked = filedialog.askopenfilename(
            title="בחר הקלטת גיחה",
            initialdir=os.path.dirname(AUDIO_PATH),
            filetypes=[("Audio", "*.mp3 *.wav *.m4a *.flac *.ogg"), ("All", "*.*")],
            parent=_root,
        )
        _root.destroy()
        if _picked:
            AUDIO_PATH = _picked
    except Exception as _e:  # noqa: BLE001
        print(f"[אזהרה] חלון בחירת קובץ לא נפתח ({_e}) - משתמש ב-AUDIO_PATH")

if not os.path.exists(AUDIO_PATH):
    print(f"הקובץ לא נמצא בנתיב: {AUDIO_PATH}")
    raise SystemExit(1)
print(f"קובץ: {AUDIO_PATH}")
_fname = os.path.basename(AUDIO_PATH).lower()
for _key, _cfg in SORTIE_SETTINGS.items():
    if _key.lower() in _fname:
        PROMPT_TYPE = _cfg.get("prompt", PROMPT_TYPE)
        CALLSIGNS = list(_cfg.get("callsigns", CALLSIGNS))
        break
print(f"סוג גיחה: {PROMPT_TYPE} | אותות קריאה: {CALLSIGNS or 'ללא'}")

whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if whisper_device == "cuda" else "int8"
sb_device = "cuda:0" if torch.cuda.is_available() else "cpu"


# ==========================================
# פונקציות עזר
# ==========================================
def fmt_time(sec):
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def _similar(a, b):
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def fix_callsigns(text):
    """
    מתקן שיבושים של אותות קריאה:
    1) מילה שדומה מאוד לאות קריאה  ("שרלו", "שירלוק", "שרלופ" -> "שרלוק")
    2) מבנה קשר "<X> מ<אות קריאה>"   ("שירת מתקיף" -> "שרלוק מתקיף"):
       מה שלפני "מתקיף" הוא כמעט תמיד אות קריאה, אז מספיק דמיון חלש + אותה אות ראשונה.
    """
    if not CALLSIGNS:
        return text
    parts = re.split(r"(\s+|[,.?!:;\-])", text)
    heb = [i for i, p in enumerate(parts) if re.search(r"[\u05d0-\u05ea]", p)]
    for k, i in enumerate(heb):
        w = re.sub(r"[^\u05d0-\u05ea]", "", parts[i])
        if len(w) < 3 or w in CALLSIGNS or w in CALLSIGN_STOPWORDS:
            continue
        if w[0] in "מהובלשכ" and w[1:] in CALLSIGNS:  # "משרלוק", "מתקיף", "לברזלים" - כבר תקין
            continue
        best = max(CALLSIGNS, key=lambda c: _similar(w, c))
        sc = _similar(w, best)
        need = CALLSIGN_SIMILARITY if len(best) > 4 else 0.85  # אותות קצרים ("תקיף") - זהירות כפולה
        nxt = re.sub(r"[^\u05d0-\u05ea]", "", parts[heb[k + 1]]) if k + 1 < len(heb) else ""
        radio_ctx = nxt.startswith("מ") and any(
            nxt[1:] == c or _similar(nxt[1:], c) >= CALLSIGN_SIMILARITY for c in CALLSIGNS if c != best
        )
        if sc >= need or (radio_ctx and w[0] == best[0] and sc >= 0.35):
            parts[i] = parts[i].replace(w, best)
    return "".join(parts)


def fix_text(text):
    for wrong, right in FIX_DICTIONARY.items():
        text = text.replace(wrong, right)
    for rx, right in _FIX_WORDS_RE:
        text = rx.sub(right, text)
    text = _RELEASE_RE.sub("נסק שחרר", text)
    text = fix_callsigns(text)
    text = re.sub(r"\b(\d)\.(\d)\.(\d)\b", r"\1\2\3", text)  # "2.7.0" -> "270"
    text = _REPEAT_RE.sub(r"\1", text)  # מוחק חזרות של 3 פעמים ומעלה
    return text.strip()


def spectral(X, k):
    return SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=max(2, min(10, len(X) - 1)),
        assign_labels="kmeans",
        random_state=42,
    ).fit_predict(X)


def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


class _Word:
    def __init__(self, start, end, word):
        self.start, self.end, self.word = start, end, word


def median_f0(wav, sr, spans, max_sec=90.0):
    """גובה צליל חציוני (Hz) על מקטעי הדיבור של דובר. NaN אם לא ניתן לחשב."""
    pieces, total = [], 0.0
    for a, b in sorted(spans, key=lambda x: x[1] - x[0], reverse=True):
        if total >= max_sec:
            break
        pieces.append(wav[0, int(a * sr):int(b * sr)].cpu().numpy())
        total += b - a
    if not pieces:
        return float("nan")
    y = np.concatenate(pieces).astype(np.float32)
    # רק librosa.pyin. החלופה של torchaudio נתנה ערכים "מדורגים" (125/200/250Hz)
    # ולא אמינים - עדיף בלי סימן גובה צליל מאשר סימן שגוי.
    if not HAS_LIBROSA:
        return float("nan")
    try:
        f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr, frame_length=1024)
    except Exception as e:  # noqa: BLE001
        print(f"[אזהרה] חישוב גובה צליל נכשל: {e}")
        return float("nan")
    vals = f0[voiced & ~np.isnan(f0)]
    return float(np.median(vals)) if len(vals) > 20 else float("nan")


def _speaker_gender_hits(text):
    """(נקבה, זכר) לפי 'אני + פועל בהווה' - 'אני עוצרת' מול 'אני שואל'."""
    fem = male = 0
    for w in re.findall(r"\bאני\s+(\S+)", text):
        w = re.sub(r"[^א-ת]", "", w)
        if len(w) < 3:
            continue
        if (w.endswith("ת") and not w.endswith("ות")) or re.fullmatch(r"מ\S*י\S?ה", w):
            fem += 1
        elif re.fullmatch(r"\Sו\S\S", w) and not w.endswith(("ה", "ת")):  # שואל, עוצר, שומע
            male += 1
        elif re.fullmatch(r"מ\S{2,}", w) and not w.endswith(("ה", "ת")):  # מבין, מסביר
            male += 1
    return fem, male


def lexical_role_score(texts, use_gender=True):
    """מחזיר (instructor_hits, evaluator_hits) לרשימת משפטים של דובר אחד."""
    inst = ev = 0
    for t in texts:
        inst += sum(1 for p in INSTRUCTOR_PATTERNS if re.search(p, t))
        ev += sum(1 for p in EVALUATOR_PATTERNS if re.search(p, t))
        if use_gender:
            fem, male = _speaker_gender_hits(t)
            inst += 2 * fem
            ev += 2 * male
            ev += 2 * len(re.findall(FEMALE_ADDRESS, t))
    return inst, ev


def classify_roles_by_lexicon(texts_c1, texts_c2, use_gender=True, min_hits=3):
    """P(c1 = מדריכה) לפי מילון + דקדוק. None אם אין מספיק ראיות."""
    i1, e1 = lexical_role_score(texts_c1, use_gender)
    i2, e2 = lexical_role_score(texts_c2, use_gender)
    if i1 + e1 + i2 + e2 < min_hits:
        return None, f"מעט מדי ביטויים מזהים ({i1 + e1 + i2 + e2})"
    r1 = (i1 + 1) / (i1 + e1 + 2)  # "כמה מדריכה" אשכול 1
    r2 = (i2 + 1) / (i2 + e2 + 2)
    p = 1.0 / (1.0 + np.exp(-6.0 * (r1 - r2)))
    return float(p), f"אשכול1: מדריכה={i1}, ממיין={e1} | אשכול2: מדריכה={i2}, ממיין={e2}"


# ---------- LLM ----------
def make_ollama_llm(model=OLLAMA_MODEL, url=OLLAMA_URL):
    """קריאה למודל מקומי דרך Ollama - בלי ספריות נוספות ובלי אינטרנט."""
    import urllib.request

    def call(prompt):
        body = json.dumps(
            {"model": model, "prompt": prompt, "stream": False, "format": "json",
             "options": {"temperature": 0}}
        ).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode("utf-8"))["response"]

    return call


def make_llm():
    if LLM_BACKEND == "ollama":
        return make_ollama_llm()
    if LLM_BACKEND == "anthropic":
        return make_anthropic_llm()
    return None


def make_anthropic_llm(model=LLM_MODEL, max_tokens=600):
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("חסר משתנה סביבה ANTHROPIC_API_KEY")
    client = anthropic.Anthropic()

    def call(prompt):
        r = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text

    return call


def build_dialogue(utterances, c_a, c_b, max_lines=220):
    """דיאלוג כרונולוגי: חניך / A / B - כדי שה-LLM יראה הקשר (מי מדבר אל מי, מתי)."""
    names = {0: "חניך", c_a: "A", c_b: "B"}
    lines = [
        (u["start"], names[u["label"]], u["text"])
        for u in utterances
        if u["label"] in names and u["text"]
    ]
    if len(lines) > max_lines:
        head, tail = lines[:60], lines[-60:]
        middle = lines[60:-60]
        pick = np.linspace(0, len(middle) - 1, max_lines - 120).astype(int)
        lines = head + [middle[i] for i in sorted(set(pick))] + tail
    return "\n".join(f"[{fmt_time(t)}] {who}: {txt[:200]}" for t, who, txt in lines)


def _parse_probability(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"לא נמצא JSON בתשובה: {raw[:200]}")
    data = json.loads(m.group(0))
    p = float(data["a_is_instructor"])
    return min(max(p, 0.0), 1.0), str(data.get("evidence", ""))


def classify_roles_by_text(utterances, c1, c2, llm_call, retries=2):
    """מחזיר הסתברות ש-c1 היא המדריכה. שואל פעמיים בסדר הפוך כדי לבטל הטיית מיקום."""

    def ask(a, b):
        prompt = f"""{ROLE_DESCRIPTION}

להלן קטעים מתוך תמלול הסשן בסדר כרונולוגי. "חניך" מזוהה. A ו-B הם שני
דוברים לא מזוהים - אחד המדריכה והשני הממיין. הזיהוי הקולי לא מושלם, אז ייתכן
שחלק קטן מהמשפטים משויך לא נכון; הסתכל על התמונה הכללית.
שים לב: מי מדבר אל החניך בזמן התרגיל, מי נותן הוראות/תרחישים/תקלות, ומי
שואל שאלות הערכה ונותן משוב וסיכום.

{build_dialogue(utterances, a, b)}

החזר JSON בלבד, בלי טקסט נוסף, בפורמט:
{{"a_is_instructor": <מספר בין 0 ל-1, ההסתברות ש-A היא המדריכה>,
  "evidence": "<משפט או שניים: על מה התבססת>"}}
"""
        last_err = None
        for _ in range(retries + 1):
            try:
                return _parse_probability(llm_call(prompt))
            except Exception as e:  # noqa: BLE001
                last_err = e
        raise RuntimeError(f"הסיווג נכשל: {last_err}")

    p_a, ev_a = ask(c1, c2)  # A = c1
    p_b, ev_b = ask(c2, c1)  # A = c2
    p_c1 = (p_a + (1.0 - p_b)) / 2.0
    return p_c1, f"[סדר רגיל] {ev_a} | [סדר הפוך] {ev_b}"


# ==========================================
# 2. תמלול
# ==========================================
print(f"1/5 טוען את מודל התמלול ({WHISPER_MODEL})...")


def _load_whisper(name):
    kw = dict(device=whisper_device, device_index=0, compute_type=compute_type)
    try:
        return WhisperModel(name, local_files_only=True, **kw)  # מהמחשב, בלי אינטרנט
    except Exception:  # noqa: BLE001
        print("   המודל לא נמצא במחשב - מוריד פעם אחת (כמה GB, רק בהרצה הראשונה)...")
        return WhisperModel(name, local_files_only=False, **kw)


model = _load_whisper(WHISPER_MODEL)

# ---- Prompt לתמלול ----
# שים לב: Whisper משתמש רק ב-~220 הטוקנים *האחרונים* של ה-prompt (בעברית ~350 תווים).
# לכן יש שני prompts קצרים לפי סוג הגיחה, ולא אחד ארוך. כל השאר מתוקן ב-FIX_WORDS.
PROMPTS = {
    # גיחת תקיפה/טקטית (כמו check 6)
    "tactical": (
        'נסק שחרר. קיבלתי, האם רשאי לחצות קו? חיובי. נ"צ מדויק, כטמ"ם, האיום חוסל, נתקלנו מימין, '
        "נקודת מעבר חובה, מעגל המתנה, עמדת ירי, רימון עשן אדום, ביצוע חוזר, יעף, רכס, נחל, "
        "טווח, משאבת דלק, סרוו, זווית הטייה, באילוצים, סוגר קולקטיב, מרים את האף, מכניס רגל, "
        "300 רגל, 40 קשר, כיוון 270, תחקיר, אתה לוקח, אני לוקח."
    ),
    # תרגילי הטסה בסיסיים - שמינייה, תבניות, אוטורוטציה (כמו check 1)
    "maneuvers": (
        "נסק שחרר. דיווח קשר טיסה. נוודא תקרת אזור, מיישר, ננמיך, תחילת מתקן, נקודה בדיוק, "
        "פנייה ימנית, מכניס רגל, פותח קולקטיב, מרים אף, מוריד אף, מוריד כוח, הטייה 20 מעלות, "
        "שיעור נסיקה 600, שמינייה, בנייה, תחקיר, קואורדינציה, זווית הטייה, סוף תרגיל, "
        "מהירות 70 קשר, גובה 1500, ציר מקביל, אוטורוטציה, עפרוני."
    ),
}

aviation_prompt = (", ".join(CALLSIGNS) + ". " if CALLSIGNS else "") + PROMPTS[PROMPT_TYPE]
try:
    _n_tok = len(model.hf_tokenizer.encode(" " + aviation_prompt, add_special_tokens=False).ids)
    print(f"   prompt: {_n_tok} טוקנים" + ("  ⚠ ארוך מדי - ההתחלה תיחתך!" if _n_tok > 220 else ""))
except Exception:  # noqa: BLE001
    pass

print("2/5 מנתח את השמע ומבצע תמלול...")
segments_generator, _whisper_info = model.transcribe(
    AUDIO_PATH,
    language="he",
    initial_prompt=aviation_prompt,
    beam_size=5,
    temperature=0.0,
    condition_on_previous_text=False,
    no_speech_threshold=0.6,
    log_prob_threshold=-1.0,
    compression_ratio_threshold=2.2,
    repetition_penalty=1.2,
    word_timestamps=True,  # חדש: זמן לכל מילה, כדי לפצל מקטע בין דוברים
    vad_filter=USE_VAD,  # מדלג על שקט/רעש - מפחית "הזיות" וחזרות של Whisper
    vad_parameters={"min_silence_duration_ms": 500},
)
segments = [s for s in segments_generator if (s.text or "").strip()]


# ---- 2b. השלמת קטעים ש-Whisper דילג עליהם ----
# Whisper "מוותר" על משפט קצר ושקט כשהוא לא בטוח בו (כך נעלם "נסק שחרר" בתחילת check 1).
# כאן מוצאים קטעים שיש בהם קול אבל אף מילה לא תומללה בהם, מגבירים כל קטע בנפרד,
# ומתמללים אותו שוב. תוצאות שנראות כמו "הזיות" טיפוסיות של Whisper נזרקות.
class _Seg:
    def __init__(self, start, end, text, words, avg_logprob, no_speech_prob):
        self.start, self.end, self.text, self.words = start, end, text, words
        self.avg_logprob, self.no_speech_prob = avg_logprob, no_speech_prob


_HALLUCINATIONS = {"תודה רבה", "תודה", "תודה רבה לך", "בבקשה", "אה", "אהה", "אממ", "טוב", "כן", "לא",
                   "אוקיי", "וזה", "הכל", "עוד פעם"}
_JUNK_PARTS = ("כתוביות", "תורגם", "subtitles", "amara", "תודה שצפיתם", "תודה רבה")


def _is_junk(text):
    t = re.sub(r"[^\u05d0-\u05ea\w\s]", " ", text).strip()
    t = re.sub(r"\s+", " ", t)
    if not t or t in _HALLUCINATIONS:
        return True
    if any(j in text.lower() for j in _JUNK_PARTS):
        return True
    if all(w in _HALLUCINATIONS or len(w) <= 1 for w in t.split()):  # "אה... כן..." / "אה... ו..."
        return True
    return False


def _recover_skipped(segments):
    if not RECOVER_GAPS or not HAS_LIBROSA:
        return segments
    from faster_whisper import decode_audio

    y = decode_audio(AUDIO_PATH, sampling_rate=16000)
    regions = librosa.effects.split(y, top_db=GAP_TOP_DB, frame_length=1024, hop_length=256)

    # כיסוי לפי *מילים* (גבולות מקטע של Whisper נמתחים ו"מכסים" שקט שאין בו כלום)
    covered = []
    for sg in segments:
        ws = list(sg.words or [])
        covered += [(float(w.start), float(w.end)) for w in ws] if ws else [(float(sg.start), float(sg.end))]

    def coverage(a, b):
        ov = sum(max(0.0, min(b, e) - max(a, st)) for st, e in covered if e > a and st < b)
        return ov / max(b - a, 1e-6)

    gaps = []
    for r0, r1 in regions:
        a, b = r0 / 16000, r1 / 16000
        if b - a >= GAP_MIN_SEC and coverage(a, b) < 0.2:
            if gaps and a - gaps[-1][1] < 0.4:
                gaps[-1][1] = b
            else:
                gaps.append([a, b])
    gaps = [g for g in gaps if g[1] - g[0] <= 15.0]
    gaps = sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[:GAP_MAX_COUNT]
    gaps.sort()
    if not gaps:
        return segments
    print(f"   בודק {len(gaps)} קטעים עם קול שלא תומללו...")

    added = []
    for a, b in gaps:
        s0 = max(0.0, a - 0.3)
        clip = y[int(s0 * 16000):int((b + 0.3) * 16000)].copy()
        peak = float(np.max(np.abs(clip))) or 1.0
        clip *= min(0.5 / peak, GAP_MAX_GAIN)  # הגברה של הקטע השקט
        gen, _ = model.transcribe(
            clip,
            language="he",
            initial_prompt=aviation_prompt,
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=None,
            log_prob_threshold=None,
            word_timestamps=True,
            vad_filter=False,
        )
        for sg in gen:
            txt = (sg.text or "").strip()
            if not txt or _is_junk(txt):
                continue
            if sg.avg_logprob < GAP_MIN_LOGPROB or sg.no_speech_prob > 0.7:
                continue
            words = [_Word(w.start + s0, w.end + s0, w.word) for w in (sg.words or [])]
            st, en = sg.start + s0, sg.end + s0
            if coverage(st, en) > 0.5:
                continue
            added.append(_Seg(st, en, sg.text, words, sg.avg_logprob, sg.no_speech_prob))
    print(f"   נוספו {len(added)} משפטים:")
    for sg in added:
        print(f"     + {fmt_time(sg.start)} {sg.text.strip()[:60]}")
    return sorted(list(segments) + added, key=lambda sg: sg.start)


segments = _recover_skipped(segments)

# ==========================================
# 3. זיהוי דוברים
# ==========================================
print("3/5 מזהה דוברים...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=MODEL_SAVE_DIR,
    run_opts={"device": sb_device},
)

waveform, sample_rate = torchaudio.load(AUDIO_PATH)
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)
if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    sample_rate = 16000
SR = sample_rate

# --- 3.1 אמבדינג לכל חלון של 1.5 שניות ---
windows = []  # {"seg", "start", "end", "emb"}
for si, seg in enumerate(segments):
    s, e = float(seg.start), float(seg.end)
    dur = e - s
    if dur < MIN_WIN_SEC:
        continue
    if dur <= WIN_SEC:
        spans = [(s, e)]
    else:
        starts = list(np.arange(s, e - WIN_SEC, HOP_SEC)) + [e - WIN_SEC]
        spans = [(float(t), float(t) + WIN_SEC) for t in starts]
    for a, b in spans:
        wav = waveform[:, int(a * SR):int(b * SR)]
        if wav.shape[1] < int(MIN_WIN_SEC * SR):
            continue
        with torch.no_grad():
            emb = classifier.encode_batch(wav).squeeze().cpu().numpy()
        n = np.linalg.norm(emb)
        if n > 0:
            windows.append({"seg": si, "start": a, "end": b, "emb": emb / n})

print(f"   {len(segments)} מקטעי תמלול, {len(windows)} חלונות קוליים")

# --- 3.2 אשכול דו-שלבי ---
_seg_win_tmp = defaultdict(list)
for _wi, _w in enumerate(windows):
    _seg_win_tmp[_w["seg"]].append(_wi)


def _find_speech_windows():
    """חלונות שהם הקרובים ביותר לפחות למילה אחת = חלונות עם דיבור."""
    out = set()
    for si, seg in enumerate(segments):
        wis = _seg_win_tmp.get(si)
        if not wis:
            continue
        for w in (seg.words or []):
            t = (w.start + w.end) / 2
            out.add(min(wis, key=lambda wi: abs((windows[wi]["start"] + windows[wi]["end"]) / 2 - t)))
    return out


_speech_windows = _find_speech_windows()


def words_by_cluster(win_labels):
    """כמה שניות של מילים (ואיזה טקסט) נפלו בכל אשכול."""
    word_sec = Counter()
    texts = defaultdict(list)
    for si, seg in enumerate(segments):
        wis = _seg_win_tmp.get(si)
        if not wis:
            continue
        for w in (seg.words or []):
            t = (w.start + w.end) / 2
            best = min(wis, key=lambda wi: abs((windows[wi]["start"] + windows[wi]["end"]) / 2 - t))
            c = int(win_labels[best])
            word_sec[c] += max(w.end - w.start, 0.0)
            texts[c].append(w.word)
    return word_sec, texts


def fix_degenerate_clusters(X, lab, max_iter=1):
    """
    תיקון למקרה הנפוץ: אחד מ-3 האשכולות הוא "זבל" (רעש/קטעים קצרים, כמעט בלי מילים),
    ובגלל זה החניך והממיין (שני קולות גבריים באותו חדר) נדחסים לאשכול אחד.
    הפתרון: לבטל את אשכול הזבל, ולפצל לשניים את האשכול הכי "מדבר" - אבל רק אם
    שני החצאים באמת נשמעים כמו שני אנשים (קול שונה או גובה צליל שונה).
    """
    lab = np.array(lab).copy()
    for _ in range(max_iter):
        word_sec, _ = words_by_cluster(lab)
        tot = sum(word_sec.values()) or 1.0
        clusters = sorted(set(int(c) for c in lab))
        small = [c for c in clusters if word_sec[c] / tot < MIN_SPEAKER_SHARE]
        big = [c for c in clusters if c not in small]
        if not small or not big:
            break
        cents = {c: l2norm(X[lab == c].mean(0, keepdims=True))[0] for c in big}
        for c in small:
            for i in np.where(lab == c)[0]:
                lab[i] = max(big, key=lambda b: float(X[i] @ cents[b]))
        target = max(big, key=lambda c: word_sec[c])
        tgt = np.where(lab == target)[0]
        idx = np.array([i for i in tgt if i in _speech_windows])  # רק חלונות עם דיבור
        if len(idx) < 8:
            break
        sub = spectral(l2norm(X[idx] - X[idx].mean(0)), 2)
        a, b = idx[sub == 0], idx[sub == 1]
        # חלונות בלי מילים באשכול הזה -> לחצי הקרוב יותר
        ca0 = l2norm(X[a].mean(0, keepdims=True))[0]
        cb0 = l2norm(X[b].mean(0, keepdims=True))[0]
        rest_w = np.array([i for i in tgt if i not in _speech_windows], dtype=int)
        b = np.concatenate([b, [i for i in rest_w if X[i] @ cb0 > X[i] @ ca0]]).astype(int)
        ca = l2norm(X[a].mean(0, keepdims=True))[0]
        cb = l2norm(X[b].mean(0, keepdims=True))[0]
        cos = float(ca @ cb)
        fa = median_f0(waveform, SR, [(windows[i]["start"], windows[i]["end"]) for i in a], max_sec=40)
        fb = median_f0(waveform, SR, [(windows[i]["start"], windows[i]["end"]) for i in b], max_sec=40)
        f0diff = abs(fa - fb) if not (np.isnan(fa) or np.isnan(fb)) else 0.0
        ok = cos < SPLIT_MAX_COS or f0diff >= SPLIT_MIN_F0_DIFF
        print(
            f"   [תיקון אשכולות] אשכול 'זבל' {small} בוטל; פיצול אשכול {target}: "
            f"דמיון={cos:.2f}, הפרש גובה צליל={f0diff:.0f}Hz -> {'מפוצל' if ok else 'לא מפוצל (אותו דובר)'}"
        )
        if not ok:
            break
        lab[b] = max(clusters) + 1
    uniq = sorted(set(int(c) for c in lab))
    remap = {c: i for i, c in enumerate(uniq)}
    return np.array([remap[int(c)] for c in lab])


def pick_trainee_cluster(win_labels):
    """
    החניך = האשכול עם הכי הרבה *מילים מדוברות* + הכי הרבה ניסוחי טייס
    ("מרים אף", "סוגר קולקטיב", "גובה 60", "כאן שרלוק"...).
    בעבר נבחר לפי זמן החלונות - ואשכול של רעש/שקט יכול "לנצח" ככה.
    """
    word_sec, texts = words_by_cluster(win_labels)
    clusters = sorted(set(int(x) for x in win_labels))
    tot_sec = sum(word_sec.values()) or 1.0
    pilot_hits = {c: sum(len(re.findall(p, " ".join(texts[c]))) for p in TRAINEE_PATTERNS)
                  for c in clusters}
    tot_hits = sum(pilot_hits.values()) or 1
    score = {c: word_sec[c] / tot_sec + pilot_hits[c] / tot_hits for c in clusters}
    best = max(score, key=score.get)
    info = {c: f"דיבור={word_sec[c]:.0f}s, ניסוחי טייס={pilot_hits[c]}, ציון={score[c]:.2f}"
            for c in clusters}
    return best, info


# תוויות: 0 = חניך, 1/2 = שני הדוברים האחרים
labels = np.zeros(len(windows), dtype=int)
diag = {}

if len(windows) >= 6:
    X = np.stack([w["emb"] for w in windows])
    lab1 = spectral(X, 3)
    lab1 = fix_degenerate_clusters(X, lab1)

    trainee_c, trainee_diag = pick_trainee_cluster(lab1)
    diag["trainee_pick"] = trainee_diag

    rest = np.where(lab1 != trainee_c)[0]
    if len(rest) >= 4:
        Xr = X[rest] - X[rest].mean(axis=0)  # מירכוז: מסיר את מה ששני הקולות חולקים
        Xr = l2norm(Xr)
        lab2 = spectral(Xr, 2)
        labels[rest] = lab2 + 1
    else:
        labels[rest] = 1

    # אבחון: דמיון קוסינוס בין מרכזי הדוברים (קרוב ל-1 = הקולות כמעט זהים)
    cents = {c: l2norm(X[labels == c].mean(0, keepdims=True))[0] for c in (0, 1, 2) if (labels == c).any()}
    for a in cents:
        for b in cents:
            if a < b:
                diag[f"cos({a},{b})"] = round(float(cents[a] @ cents[b]), 3)

# --- 3.3 החלקה: כל חלון מקבל את תווית הרוב שלו ושל שכניו באותו מקטע ---
seg_windows = defaultdict(list)
for wi, w in enumerate(windows):
    seg_windows[w["seg"]].append(wi)

smoothed = labels.copy()
for si, wis in seg_windows.items():
    for k, wi in enumerate(wis):
        neigh = [labels[x] for x in wis[max(0, k - 1):k + 2]]
        top, cnt = Counter(neigh).most_common(1)[0]
        if cnt >= 2:
            smoothed[wi] = top
labels = smoothed


def label_at(si, t):
    wis = seg_windows.get(si)
    if not wis:
        return None
    best = min(wis, key=lambda wi: abs((windows[wi]["start"] + windows[wi]["end"]) / 2 - t))
    return int(labels[best])


# --- 3.4 תווית לכל מילה -> פיצול מקטעים לפי דובר ---
utterances = []  # {"start","end","label","text"}
prev_label = 0
for si, seg in enumerate(segments):
    words = list(seg.words or []) or [_Word(seg.start, seg.end, seg.text)]
    runs = []  # [label, [words]]
    for w in words:
        lb = label_at(si, (w.start + w.end) / 2)
        if lb is None:
            lb = prev_label
        if runs and runs[-1][0] == lb:
            runs[-1][1].append(w)
        else:
            runs.append([lb, [w]])
        prev_label = lb

    # רצפים קצרים מדי מצורפים לשכן (מונע החלפות דובר באמצע מילה/צירוף)
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (lb, ws) in enumerate(runs):
            if ws[-1].end - ws[0].start < MIN_RUN_SEC:
                j = i - 1 if i > 0 else i + 1
                if j < i:
                    runs[j][1].extend(ws)
                else:
                    runs[j][1][:0] = ws
                del runs[i]
                merged = []
                for r in runs:
                    if merged and merged[-1][0] == r[0]:
                        merged[-1][1].extend(r[1])
                    else:
                        merged.append(r)
                runs = merged
                changed = True
                break

    for lb, ws in runs:
        text = fix_text("".join(w.word for w in ws).strip())
        if text:
            utterances.append(
                {"start": float(ws[0].start), "end": float(ws[-1].end), "label": lb, "text": text}
            )

# --- 3.5 הכרעה: מי מבין 1/2 היא המדריכה ---
c1, c2 = 1, 2
stats = {}
total_end = max((u["end"] for u in utterances), default=1.0)
late_cut = 0.85 * total_end
for c in (c1, c2):
    us = [u for u in utterances if u["label"] == c]
    dur = sum(u["end"] - u["start"] for u in us)
    late = sum(u["end"] - u["start"] for u in us if u["start"] >= late_cut)
    stats[c] = {
        "turns": len(us),
        "duration": dur,
        "late_share": late / dur if dur > 0 else 0.0,
        "f0": median_f0(waveform, SR, [(u["start"], u["end"]) for u in us]),
    }

signals = {}

# (א) טקסט - LLM
evidence_text = ""
try:
    llm = make_llm()
    if llm is not None:
        p_text, evidence_text = classify_roles_by_text(utterances, c1, c2, llm)
        signals["text"] = (W_TEXT, p_text)
except Exception as e:  # noqa: BLE001
    print(f"[אזהרה] הסיווג הטקסטי (LLM={LLM_BACKEND}) נכשל: {e}")

# (א2) מילון ביטויים + דקדוק זכר/נקבה - לא דורש API
p_lex, lex_info = classify_roles_by_lexicon(
    [u["text"] for u in utterances if u["label"] == c1],
    [u["text"] for u in utterances if u["label"] == c2],
    use_gender=USE_GENDER_PRIOR,
)
if p_lex is not None:
    signals["lexicon"] = (W_LEX, p_lex)

# (ב) גובה צליל - מדריכה (קול נשי) גבוהה יותר מהממיין
f1, f2 = stats[c1]["f0"], stats[c2]["f0"]
if USE_GENDER_PRIOR and not (np.isnan(f1) or np.isnan(f2)):
    diff = f1 - f2
    if abs(diff) >= 40:
        signals["pitch"] = (W_PITCH, 0.9 if diff > 0 else 0.1)
    elif abs(diff) >= 20:
        signals["pitch"] = (W_PITCH, 0.7 if diff > 0 else 0.3)

# (ג) התנהגות - המדריכה מדברת יותר פעמים, הממיין יותר בסוף (משוב/סיכום)
t1, t2 = stats[c1]["turns"], stats[c2]["turns"]
if t1 + t2 > 0:
    p_turns = t1 / (t1 + t2)
    p_late = 0.5 + 0.5 * (stats[c2]["late_share"] - stats[c1]["late_share"])
    p_beh = 0.5 * p_turns + 0.5 * p_late
    if W_BEHAVIOR > 0:
        signals["behavior"] = (W_BEHAVIOR, p_beh)

if signals:
    wsum = sum(w for w, _ in signals.values())
    p_c1 = sum(w * p for w, p in signals.values()) / wsum
else:
    p_c1 = 0.5

instructor_c, evaluator_c = (c1, c2) if p_c1 >= 0.5 else (c2, c1)
if FORCE_SWAP_ROLES:
    instructor_c, evaluator_c = evaluator_c, instructor_c

needs_review = (not signals) or abs(p_c1 - 0.5) < REVIEW_MARGIN
role_of = {0: ROLE_TRAINEE, instructor_c: ROLE_INSTRUCTOR, evaluator_c: ROLE_EVALUATOR}

print("\n" + "=" * 60)
print("ניתוח והכרעה בין הדוברים:")
for c in (c1, c2):
    st = stats[c]
    print(
        f"  • אשכול {c}: מקטעים={st['turns']}, זמן={st['duration']:.1f}s, "
        f"F0≈{st['f0']:.0f}Hz, חלק בסוף ההקלטה={st['late_share']:.0%}  -> {role_of[c]}"
    )
for name, (w, p) in signals.items():
    print(f"  סימן {name}: P(אשכול {c1} = מדריכה) = {p:.2f}  (משקל {w})")
print(f"  משולב: {p_c1:.2f}   ביטחון: {abs(p_c1 - 0.5) * 2:.0%}")
if "trainee_pick" in diag:
    print("  בחירת חניך (שלב 1):")
    for c, txt in diag["trainee_pick"].items():
        print(f"     אשכול {c}: {txt}")
if diag:
    print(f"  דמיון קולי בין אשכולות: { {k: v for k, v in diag.items() if k != 'trainee_pick'} }")
print(f"  [מילון ביטויים]: {lex_info}")
if evidence_text:
    print(f"  [ראיות ה-LLM]: {evidence_text}")
if not signals:
    print("  ⚠ אין אף סימן אמין - ההכרעה אקראית! התקן librosa / הפעל LLM_BACKEND")
if needs_review:
    print("  ⚠ הסימנים לא חד-משמעיים - כדאי לבדוק ידנית (ואם הפוך: FORCE_SWAP_ROLES = True)")
print("=" * 60 + "\n")

# ==========================================
# 4. עיבוד נתונים לטבלה
# ==========================================
print("4/5 מעבד נתונים ומכין טבלאות...")
processed, excel_rows = [], []
for u in utterances:
    text = u["text"]
    speaker = role_of.get(u["label"], ROLE_TRAINEE)

    duration = max(u["end"] - u["start"], 0.1)
    speech_rate = round(len(text.split()) / duration, 2)
    seg_wav = waveform[:, int(u["start"] * SR):int(u["end"] * SR)]
    rms_volume = round(float(torch.sqrt(torch.mean(seg_wav ** 2)).item()) * 100, 2) if seg_wav.numel() else 0.0
    f0 = median_f0(waveform, SR, [(u["start"], u["end"])]) if duration >= 0.5 else float("nan")

    processed.append({**u, "speaker": speaker})
    excel_rows.append(
        {
            "הערות / תמלול": text,
            "דובר": speaker,
            "עוצמה": rms_volume,
            "גובה צליל (F0)": "" if np.isnan(f0) else f"{f0:.0f} Hz",
            "קצב": speech_rate,
            "זמן": fmt_time(u["start"]),
        }
    )

# ==========================================
# 5. ייצוא לאקסל
# ==========================================
print(f"5/5 מייצא לאקסל: {EXCEL_OUTPUT}")
summary_rows = [
    {"פרמטר": "P(אשכול 1 = מדריכה)", "ערך": round(p_c1, 3)},
    {"פרמטר": "ביטחון", "ערך": f"{abs(p_c1 - 0.5) * 2:.0%}"},
    {"פרמטר": "לבדיקה ידנית", "ערך": "כן" if needs_review else "לא"},
    {"פרמטר": "סימנים", "ערך": ", ".join(f"{k}={p:.2f}" for k, (_, p) in signals.items())},
    {"פרמטר": "מילון ביטויים", "ערך": lex_info},
    {"פרמטר": "ראיות LLM", "ערך": evidence_text},
]
for c in (c1, c2):
    st = stats[c]
    summary_rows.append(
        {
            "פרמטר": f"אשכול {c} -> {role_of[c]}",
            "ערך": f"מקטעים={st['turns']}, זמן={st['duration']:.0f}s, F0={st['f0']:.0f}Hz",
        }
    )

with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:
    pd.DataFrame(excel_rows).to_excel(writer, index=False, sheet_name="תמלול קשר")
    pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="זיהוי דוברים")
    for ws in writer.sheets.values():
        ws.sheet_view.rightToLeft = True

# ==========================================
# 6. HTML
# ==========================================
audio_src_path = AUDIO_PATH.replace("\\", "/")
STYLE_BY_ROLE = {
    ROLE_TRAINEE: "speaker-1",
    ROLE_EVALUATOR: "speaker-2",
    ROLE_INSTRUCTOR: "instructor",
}
review_banner = (
    '<div class="review">⚠ זיהוי מדריכה/ממיין לא חד-משמעי בגיחה הזו - כדאי לבדוק ידנית</div>'
    if needs_review
    else ""
)

html_header = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<title>תמלול גיחה</title>
<style>
 body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background:#f0f4f8; margin:0; padding:20px; color:#333; }}
 .wrapper {{ max-width:900px; margin:0 auto; }}
 .player-panel {{ position:sticky; top:20px; background:#2c3e50; color:#fff; padding:20px; border-radius:12px; margin-bottom:25px; z-index:100; }}
 .player-panel h2 {{ margin:0 0 15px; font-size:1.4rem; }}
 .legend span {{ margin-left:14px; font-size:.9rem; }}
 audio {{ width:100%; }}
 .review {{ background:#fff3cd; color:#7a5b00; padding:10px 14px; border-radius:8px; margin-bottom:15px; }}
 .transcript-container {{ background:#fff; padding:30px; border-radius:12px; }}
 .segment-block {{ padding:15px; margin-bottom:15px; border-radius:8px; border-right:4px solid #bdc3c7; background:#fdfdfd; cursor:pointer; transition:all .3s; }}
 .segment-block:hover {{ background:#f5f7fa; }}
 .segment-block.active {{ border-right-color:#2ecc71 !important; background:#e8f8f0 !important; }}
 .meta-info {{ display:flex; justify-content:space-between; font-size:.85rem; color:#7f8c8d; margin-bottom:8px; font-weight:bold; }}
 .speaker-1 {{ border-right-color:#3498db; }} .speaker-1 .speaker-name {{ color:#3498db; }}
 .speaker-2 {{ border-right-color:#9b59b6; }} .speaker-2 .speaker-name {{ color:#9b59b6; }}
 .instructor {{ border-right-color:#e74c3c; background:#fdf2f2; }} .instructor .speaker-name {{ color:#e74c3c; }}
 .sim-noise {{ border-right-color:#7f8c8d; background:#f2f4f4; }} .sim-noise .speaker-name {{ color:#7f8c8d; font-style:italic; }}
 .text-content {{ font-size:1.15rem; line-height:1.6; }}
</style>
</head>
<body>
<div class="wrapper">
 <div class="player-panel">
  <h2>נגן קשר טיסה ✈️</h2>
  <div class="legend"><span style="color:#5dade2">■ חניך</span><span style="color:#ec7063">■ מדריכה</span><span style="color:#bb8fce">■ ממיין</span></div>
  <audio id="audioPlayer" controls><source src="file:///{audio_src_path}" type="audio/mp3"></audio>
 </div>
 {review_banner}
 <div class="transcript-container"><div id="timeline">
"""

html_segments = ""
for idx, seg in enumerate(processed):
    cls = STYLE_BY_ROLE.get(seg["speaker"], "sim-noise")
    html_segments += f"""
  <div class="segment-block {cls}" id="seg-{idx}" data-start="{seg['start']}" data-end="{seg['end']}" onclick="seekTo({seg['start']})">
   <div class="meta-info"><span class="speaker-name">{html.escape(seg['speaker'])}</span><span>{fmt_time(seg['start'])}</span></div>
   <div class="text-content">{html.escape(seg['text'])}</div>
  </div>"""

html_footer = """
 </div></div>
</div>
<script>
 const player = document.getElementById('audioPlayer');
 const blocks = document.querySelectorAll('.segment-block');
 function seekTo(s) { player.currentTime = s; player.play(); }
 player.addEventListener('timeupdate', () => {
   const t = player.currentTime;
   blocks.forEach((b) => {
     const s = parseFloat(b.dataset.start), e = parseFloat(b.dataset.end);
     if (t >= s && t <= e && !b.classList.contains('active')) {
       blocks.forEach(x => x.classList.remove('active'));
       b.classList.add('active');
       b.scrollIntoView({ behavior: 'smooth', block: 'center' });
     }
   });
 });
</script>
</body>
</html>
"""

with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
    f.write(html_header + html_segments + html_footer)

print("\nהתהליך הושלם בהצלחה!")
print(f"אקסל: {EXCEL_OUTPUT}")
webbrowser.open(HTML_OUTPUT)

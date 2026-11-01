# -*- coding: utf-8 -*-
"""
הכנה חד-פעמית להעברה למחשב מנותק מרשת.

מעתיק את שני המודלים (תמלול Whisper + זיהוי קולות speechbrain) מתיקיית המטמון של
Hugging Face (C:\\Users\\<שם>\\.cache\\huggingface) לתוך התיקייה models שליד הסקריפט,
כקבצים אמיתיים (לא קיצורי דרך).

אחרי זה transcribe_v3.py טוען את המודלים רק מ-models ולא פונה לרשת בכלל,
ואפשר להעתיק את כל התיקייה VoiceSoftware (כולל .venv ו-models) למחשב מנותק.

הרצה: פעם אחת, על המחשב הזה (שבו המודלים כבר ירדו).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

MODELS = [
    # (מזהה ב-Hugging Face, תיקייה מקומית, חובה?)
    ("ivrit-ai/whisper-large-v3-turbo-ct2", "whisper-ivrit-turbo", True),
    ("speechbrain/spkrec-ecapa-voxceleb", "spkrec-ecapa-voxceleb", True),
    ("Systran/faster-whisper-large-v3", "whisper-large-v3", False),  # המודל הישן, רק אם קיים
]

from huggingface_hub import snapshot_download  # noqa: E402

import shutil  # noqa: E402

ok = True
for repo, folder, required in MODELS:
    dest = os.path.join(MODELS_DIR, folder)
    print(f"\n{repo}  ->  {dest}")
    try:
        src = snapshot_download(repo, local_files_only=True)      # מהמטמון בלבד, בלי רשת
        print("   נמצא במטמון")
    except Exception:  # noqa: BLE001
        if not required:
            print("   לא נמצא במטמון - מדלג (לא חובה)")
            continue
        try:
            print("   לא נמצא במטמון - מוריד מהאינטרנט (חד-פעמי)...")
            src = snapshot_download(repo)
        except Exception as e:  # noqa: BLE001
            print(f"   נכשל: {e}")
            ok = False
            continue
    # העתקה כקבצים אמיתיים (קיצורי הדרך של המטמון מוחלפים בקבצים עצמם)
    shutil.copytree(src, dest, symlinks=False, dirs_exist_ok=True)
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(dest) for f in fs)
    print(f"   הועתק: {total / 1e9:.2f} GB")

# ---- מודל שפה מקומי (DictaLM 2.0, עברית) - קובץ אחד של ~4.4GB ----
# משמש ל"מה המודל המקומי לומד על החניך" בדף הגרפים. לא חובה - אפשר לכבות כאן.
DOWNLOAD_LLM = True
if DOWNLOAD_LLM:
    from huggingface_hub import hf_hub_download

    repo, fname = "dicta-il/dictalm2.0-instruct-GGUF", "dictalm2.0-instruct.Q4_K_M.gguf"
    dest_dir = os.path.join(MODELS_DIR, "llm")
    dest = os.path.join(dest_dir, fname)
    print(f"\n{repo}/{fname}  ->  {dest}")
    if os.path.exists(dest) and os.path.getsize(dest) > 1e9:
        print("   כבר קיים")
    else:
        try:
            try:
                src = hf_hub_download(repo, fname, local_files_only=True)
                print("   נמצא במטמון")
            except Exception:  # noqa: BLE001
                print("   מוריד מהאינטרנט (חד-פעמי, ~4.4GB - יכול לקחת זמן)...")
                src = hf_hub_download(repo, fname)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(os.path.realpath(src), dest)
            print(f"   הועתק: {os.path.getsize(dest) / 1e9:.2f} GB")
        except Exception as e:  # noqa: BLE001
            print(f"   נכשל (לא חובה - הגרפים יעבדו גם בלי): {e}")
    # התקנת llama-cpp-python מהקבצים שבתיקייה wheels (בלי אינטרנט)
    try:
        import llama_cpp  # noqa: F401

        print("   llama-cpp-python כבר מותקן")
    except ImportError:
        import subprocess

        wheels = os.path.join(HERE, "wheels")
        print(f"   מתקין llama-cpp-python מתוך {wheels} ...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", wheels,
                            "llama-cpp-python"], check=False)
        if r.returncode == 0:
            print("   llama-cpp-python הותקן")
        else:
            print("   ההתקנה נכשלה - ראה את השגיאה למעלה")

print()
if ok:
    print("מוכן. מעכשיו transcribe_v3.py עובד בלי רשת.")
    print("להעברה למחשב מנותק: מעתיקים את כל התיקייה VoiceSoftware (כולל .venv ו-models).")
    print("במחשב היעד צריך Windows ו-Python 3.11 מותקן באותו נתיב כמו כאן.")
else:
    print("חלק מהמודלים לא הוכנו - ראה הודעות למעלה.")
    sys.exit(1)

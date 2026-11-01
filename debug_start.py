# -*- coding: utf-8 -*-
"""
בדיקה קצרה: למה "נסק שחרר" (בערך שנייה 8-10) לא מתומלל.
מתמלל רק את 20 השניות הראשונות, בכמה הגדרות ו-2 מודלים, וכותב הכל ל-debug_start.txt
(הקובץ הזה נפרד מ-transcribe_v3.py ולא משנה אותו)
"""
import os
import sys

import numpy as np
from faster_whisper import WhisperModel, decode_audio

AUDIO = r"d:\Ofir\VoiceSoftware\check 1 tavlan and carmel.mp3"
START, END = 0.0, 20.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_start.txt")
MODELS = ["ivrit-ai/whisper-large-v3-turbo-ct2", "large-v3"]

try:
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:  # noqa: BLE001
    dev = "cpu"
ctype = "float16" if dev == "cuda" else "int8"

lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


y = decode_audio(AUDIO, sampling_rate=16000)[int(START * 16000):int(END * 16000)]
log(f"audio {START}-{END}s, device={dev}")

# 1) איפה יש קול (עוצמה)
try:
    import librosa

    for db in (30, 35, 40):
        r = librosa.effects.split(y, top_db=db, frame_length=1024, hop_length=256)
        log(f"regions top_db={db}: " + ", ".join(f"{a/16000+START:.1f}-{b/16000+START:.1f}" for a, b in r))
except Exception as e:  # noqa: BLE001
    log(f"librosa error: {e}")

# עוצמה לכל חצי שנייה
rms = [float(np.sqrt(np.mean(y[i:i + 8000] ** 2))) for i in range(0, len(y), 8000)]
log("RMS per 0.5s: " + " ".join(f"{START + i*0.5:.1f}:{v*100:.2f}" for i, v in enumerate(rms)))

SETTINGS = {
    "default": dict(no_speech_threshold=0.6, log_prob_threshold=-1.0),
    "no_skip": dict(no_speech_threshold=None, log_prob_threshold=None),
}
CLIPS = {
    "first20s": y,
    "only_7-12s_boosted": None,
}
c = y[int(7 * 16000):int(12 * 16000)].copy()
c *= min(0.5 / (float(np.max(np.abs(c))) or 1.0), 20.0)
CLIPS["only_7-12s_boosted"] = c

for name in MODELS:
    log("\n" + "=" * 70)
    log(f"MODEL: {name}")
    try:
        try:
            m = WhisperModel(name, device=dev, compute_type=ctype, local_files_only=True)
        except Exception:  # noqa: BLE001
            m = WhisperModel(name, device=dev, compute_type=ctype)
    except Exception as e:  # noqa: BLE001
        log(f"  cannot load: {e}")
        continue
    for cname, audio in CLIPS.items():
        off = START if cname == "first20s" else 7.0
        for sname, kw in SETTINGS.items():
            for prompt in (None, "נסק שחרר. דיווח קשר טיסה."):
                log(f"\n--- clip={cname} settings={sname} prompt={'yes' if prompt else 'no'}")
                try:
                    segs, _ = m.transcribe(
                        audio, language="he", beam_size=5, temperature=0.0,
                        condition_on_previous_text=False, word_timestamps=True,
                        initial_prompt=prompt, vad_filter=False, **kw,
                    )
                    for s in segs:
                        log(f"  [{s.start+off:6.2f}-{s.end+off:6.2f}] lp={s.avg_logprob:.2f} "
                            f"nosp={s.no_speech_prob:.2f} | {s.text.strip()}")
                        log("      words: " + " ".join(
                            f"{w.word.strip()}({w.start+off:.1f}-{w.end+off:.1f})" for w in (s.words or [])))
                except Exception as e:  # noqa: BLE001
                    log(f"  error: {e}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
log(f"\nנשמר: {OUT}")

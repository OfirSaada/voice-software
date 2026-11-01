---
license: apache-2.0
datasets:
- ivrit-ai/crowd-transcribe-v5
language:
- he
base_model:
- openai/whisper-large-v3-turbo
tags:
- audio
- automatic-speech-recognition
library_name: ctranslate2
---

This is ivrit.ai's faster-whisper model, based on the ivrit-ai/whisper-large-v3-turbo Whisper model.

Training data includes 295 hours of volunteer-transcribed speech from the ivrit-ai/crowd-transcribe-v5 dataset, as well as 93 hours of professional transcribed speech from other sources.

Release date: TBD

# Prerequisites

pip3 install faster_whisper

# Usage

```
import faster_whisper
model = faster_whisper.WhisperModel('ivrit-ai/whisper-large-v3-turbo-ct2')

segs, _ = model.transcribe('media-file', language='he')

texts = [s.text for s in segs]

transcribed_text = ' '.join(texts)
print(f'Transcribed text: {transcribed_text}')
```
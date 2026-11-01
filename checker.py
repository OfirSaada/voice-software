import os
from faster_whisper import WhisperModel
from pydub import AudioSegment

# Step 1: Clean radio audio locally
def clean_radio_audio(input_file, output_file):
    sound = AudioSegment.from_file(input_file)
    
    # Force mono and 16kHz sample rate required by local models
    sound = sound.set_channels(1).set_frame_rate(16000)
    
    # Boost lower vocal frequencies and drop extreme high-frequency radio static
    sound = sound.low_pass_filter(3000).high_pass_filter(300)
    
    sound.export(output_file, format="wav")
    print("Audio optimized for local AI model processing.")

# Step 2: Offline transcription 
def transcribe_offline(audio_path):
    print("Loading local model into memory...")
    
    # Use "cpu" if you don't have an Nvidia GPU. Use "cuda" if you do.
    # "base" model is small (~140MB), fast, and very accurate for clear speech.
    # Change to "small" or "medium" if your radio static is severe.
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    print("Transcribing file locally...")
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    print(f"Detected language: '{info.language}' with probability {info.language_probability:.2f}")
    
    full_text = []
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        full_text.append(segment.text)
        
    return " ".join(full_text)

# Execution
input_file = "radio_recording.mp3"
clean_file = "local_clean.wav"

clean_radio_audio(input_file, clean_file)
final_text = transcribe_offline(clean_file)
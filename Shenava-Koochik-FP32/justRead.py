from pathlib import Path
import re
import time
import numpy as np
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig

BASE = Path(__file__).resolve().parent

MODEL = BASE / "models/fa_IR-ganji-medium.onnx"
TEXT = BASE / "read.txt"

SPEED = 1.20       # Higher = slower
PAUSE = 0.8        # Pause between paragraphs

# Load offline voice
voice = PiperVoice.load(str(MODEL))

# Slower speech
config = SynthesisConfig(length_scale=SPEED)

# Read and split paragraphs
content = TEXT.read_text(encoding="utf-8")
paragraphs = [
    p.strip().replace("\n", " ")
    for p in re.split(r"\n\s*\n", content)
    if p.strip()
]

# Read paragraphs one by one
for paragraph in paragraphs:
    chunks = list(voice.synthesize(paragraph, syn_config=config))

    if not chunks:
        continue

    audio = np.concatenate(
        [chunk.audio_float_array for chunk in chunks]
    ).astype(np.float32)

    rate = chunks[0].sample_rate

    # Stereo playback
    stereo = np.column_stack((audio, audio))
    sd.play(stereo, rate)
    sd.wait()

    time.sleep(PAUSE)

print("Reading completed.")
import os
from pathlib import Path
import librosa
import soundfile as sf

INPUT_DIR  = r"D:\spoken_rec_wav"
OUTPUT_DIR = r"D:\spoken_rec_clean"

TOP_DB = 25  # silence threshold 

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

processed = 0

for command in os.listdir(INPUT_DIR):
    in_cmd = os.path.join(INPUT_DIR, command)
    out_cmd = os.path.join(OUTPUT_DIR, command)

    if not os.path.isdir(in_cmd):
        continue

    Path(out_cmd).mkdir(parents=True, exist_ok=True)

    for file in os.listdir(in_cmd):
        if file.lower().endswith(".wav"):
            in_path = os.path.join(in_cmd, file)

            # load (already 16k mono, but keep sr fixed)
            y, sr = librosa.load(in_path, sr=16000, mono=True)

            # remove silence
            y_trim, _ = librosa.effects.trim(y, top_db=TOP_DB)

            # normalize amplitude
            y_norm = librosa.util.normalize(y_trim)

            out_path = os.path.join(out_cmd, file)
            sf.write(out_path, y_norm, 16000)

            processed += 1

print(f"Done. Cleaned {processed} files into: {OUTPUT_DIR}")

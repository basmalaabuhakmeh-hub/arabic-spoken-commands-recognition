import os
import subprocess
from pathlib import Path

# ======================================================
# 1. FULL PATH TO ffmpeg.exe  
# ======================================================
FFMPEG = r"C:\Users\Layal\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"

# ======================================================
# 2. INPUT & OUTPUT FOLDERS
# ======================================================
INPUT_DIR = r"D:\spoken_rec_raw"   # raw phone recordings
OUTPUT_DIR = r"D:\spoken_rec_wav"  # converted WAV files

# Audio formats to convert
AUDIO_EXTS = (".mp4", ".m4a", ".mp3", ".aac")

# ======================================================
# 3. Conversion function
# ======================================================
def convert_to_wav_16k_mono(in_path, out_path):
    cmd = [
        FFMPEG,
        "-y",                 # overwrite output if exists
        "-i", in_path,        # input file
        "-ac", "1",           # mono
        "-ar", "16000",       # 16 kHz
        "-vn",                # no video
        out_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error converting:", in_path)
        print(result.stderr)
    else:
        print(f"Converted: {in_path} ->  {out_path}")

# ======================================================
# 4. Walk through dataset & convert
# ======================================================
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

converted_count = 0

for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if file.lower().endswith(AUDIO_EXTS):
            in_path = os.path.join(root, file)

            # Keep same folder structure
            relative_path = os.path.relpath(root, INPUT_DIR)
            out_dir = os.path.join(OUTPUT_DIR, relative_path)
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            out_file = Path(file).stem + ".wav"
            out_path = os.path.join(out_dir, out_file)

            convert_to_wav_16k_mono(in_path, out_path)
            converted_count += 1

print("\n==============================")
print(f"DONE! Converted {converted_count} files to WAV (16 kHz, mono)")
print("==============================")

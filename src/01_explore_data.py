"""
Phase 1: Explore ALFFA Swahili Dataset
Reads the Kaldi-style text and wav.scp files.
"""

import os
import sys
from collections import Counter

DATA_DIR = "data/raw/alffa_sw"

if not os.path.exists(DATA_DIR):
    print(f"Directory not found: {DATA_DIR}")
    print("Please run the extraction script first.")
    sys.exit(1)

text_path = os.path.join(DATA_DIR, "text")
wav_scp_path = os.path.join(DATA_DIR, "wav.scp")

if not os.path.exists(text_path):
    print(f"text file not found: {text_path}")
    sys.exit(1)

if not os.path.exists(wav_scp_path):
    print(f"wav.scp file not found: {wav_scp_path}")
    sys.exit(1)

print("Loading ALFFA dataset...")

with open(text_path, 'r', encoding='utf-8') as f:
    text_lines = f.readlines()

print(f"Loaded {len(text_lines)} utterances from text file.")

with open(wav_scp_path, 'r', encoding='utf-8') as f:
    wav_lines = f.readlines()

print(f"Loaded {len(wav_lines)} audio references from wav.scp.")

text_dict = {}
for line in text_lines:
    parts = line.strip().split(maxsplit=1)
    if len(parts) == 2:
        text_dict[parts[0]] = parts[1]

wav_dict = {}
for line in wav_lines:
    parts = line.strip().split(maxsplit=1)
    if len(parts) == 2:
        wav_dict[parts[0]] = parts[1]

common_ids = set(text_dict.keys()) & set(wav_dict.keys())
print(f"Common utterance IDs between text and audio: {len(common_ids)}")

if len(common_ids) == 0:
    print("Error: No matching utterance IDs found.")
    sys.exit(1)

ids = list(common_ids)
NUM_SAMPLES = min(10, len(ids))

print("\n" + "=" * 50)
print("SAMPLE INSPECTION (First 10 utterances)")
print("=" * 50)

for i in range(NUM_SAMPLES):
    utt_id = ids[i]
    text = text_dict[utt_id]
    audio_path = wav_dict[utt_id]
    print(f"\n--- Sample {i+1} ---")
    print(f"Utterance ID: {utt_id}")
    print(f"Text: {text}")
    print(f"Audio file: {audio_path}")

print("\n" + "=" * 50)
print("TEXT ANALYSIS")
print("=" * 50)

texts = list(text_dict.values())
all_chars = "".join(texts)
unique_chars = sorted(set(all_chars))

print(f"Total utterances: {len(texts)}")
print(f"Unique characters found: {len(unique_chars)}")
print(f"Character set: {''.join(unique_chars)}")

problematic = [c for c in unique_chars if not c.isalnum() and c not in " .,!?'\""]
if problematic:
    print(f"Non-standard punctuation found: {problematic}")

avg_len = sum(len(t) for t in texts) / len(texts)
print(f"Average transcript length: {avg_len:.2f} characters")

print("\n" + "=" * 50)
print("AUDIO FILE CHECK")
print("=" * 50)

audio_files = list(wav_dict.values())
valid_audio = []
for path in audio_files:
    if os.path.exists(path):
        valid_audio.append(path)
    else:
        # Try relative path from DATA_DIR
        alt_path = os.path.join(DATA_DIR, path)
        if os.path.exists(alt_path):
            valid_audio.append(alt_path)

print(f"Audio references in wav.scp: {len(audio_files)}")
print(f"Audio files found on disk: {len(valid_audio)}")

if len(valid_audio) < len(audio_files):
    print(f"Missing {len(audio_files) - len(valid_audio)} audio files.")
    print("This might be due to relative paths. The exploration script will work anyway.")

os.makedirs("data/raw", exist_ok=True)
report_path = "data/raw/phase1_alffa_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("PHASE 1 ALFFA EXPLORATION REPORT\n")
    f.write("=" * 40 + "\n")
    f.write(f"Data directory: {DATA_DIR}\n")
    f.write(f"Total utterances: {len(texts)}\n")
    f.write(f"Unique characters: {len(unique_chars)}\n")
    f.write(f"Character set: {''.join(unique_chars)}\n")
    f.write(f"Average transcript length: {avg_len:.2f}\n")

print(f"\nReport saved to: {report_path}")

print("\n" + "=" * 50)
print("PHASE 1 COMPLETE")
print("=" * 50)
print(f"Summary:")
print(f"  - Dataset: ALFFA Swahili")
print(f"  - {len(texts)} utterances found")
print(f"  - {len(unique_chars)} unique characters detected")
print("\nNext: Phase 2 - Preprocessing on Google Colab")
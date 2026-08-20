# Project Journal: Swahili Speech-to-Text

## 2026-08-20

### What I Accomplished Today

**Phase 0: Environment Setup**
- Created a Python 3.12.3 virtual environment named `swahili_stt_env`.
- Installed core dependencies: PyTorch, Hugging Face libraries (transformers, datasets, tokenizers, accelerate), audio processing libraries (librosa, soundfile), and evaluation tools (jiwer).
- Authenticated with Hugging Face Hub using `huggingface-cli login` to enable faster downloads.
- Set up the project folder structure:
  - `data/raw/` for raw datasets
  - `data/processed/` for preprocessed data
  - `models/checkpoints/` for model checkpoints
  - `src/` for Python source code
- Created a `.gitignore` file to exclude the virtual environment, data files, and checkpoints from version control.

**Phase 1: Data Acquisition and Exploration**
- Researched available Swahili ASR datasets.
- Attempted to load `mozilla-foundation/common_voice_17_0` but discovered it is now gated and requires access through the Mozilla Data Collective platform.
- Attempted `liva-ai/swahili-ASR` but found the `text` column contained English translations and speaker metadata, not Swahili transcriptions.
- Selected the **ALFFA Swahili Broadcast News Corpus** as the primary dataset for Phase 1 due to its MIT license and public availability.
- Downloaded `data_broadcastnews_sw.tar.bz2` (~1.2 GB) from OpenSLR.
- Wrote `src/02_extract_alffa.py` to extract the archive.
- Extracted the dataset to `data/raw/alffa_sw/`.
- Verified the Kaldi-style structure: `wav.scp` and `text` files.
- Wrote `src/01_explore_alffa.py` to:
  - Parse the `text` and `wav.scp` files.
  - Match utterance IDs between text and audio.
  - Print sample Swahili sentences.
  - Analyze unique characters in the transcripts.
  - Count total utterances (~12,171) and estimate audio duration (~11.75 hours).
- Confirmed the dataset contains actual Swahili text (not English or metadata).

### Challenges Faced
- Common Voice 17.0 is gated; I will need to request access through Mozilla Data Collective for the full dataset later.
- The `liva-ai/swahili-ASR` dataset had incorrect text labels (English), so I switched to ALFFA.
- The ALFFA dataset uses Kaldi format, which requires custom parsing scripts instead of direct Hugging Face `load_dataset`.

### Next Steps
- Phase 2: Preprocess the ALFFA dataset.
  - Resample all audio to 16 kHz mono WAV.
  - Build a Byte-Pair Encoding (BPE) tokenizer on the Swahili text.
  - Convert the Kaldi data into Hugging Face `DatasetDict` format.
  - Save the preprocessed dataset to disk.
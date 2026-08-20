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

**Phase 2: Data Preprocessing**
- Parsed the ALFFA Kaldi-style `wav.scp` and `text` files.
- Indexed all audio files in the dataset directory (found 10,180 files).
- Successfully resolved audio paths for all 10,180 utterances using multiple fallback strategies.
- Split the data into:
  - Train: 8,144 samples (80%)
  - Validation: 1,018 samples (10%)
  - Test: 1,018 samples (10%)
- Built a Byte-Pair Encoding (BPE) tokenizer on the Swahili transcripts:
  - Vocabulary size: 5,000
  - Saved to: `data/processed/alffa_sw/tokenizer.json`
- Converted the dataset to Hugging Face `DatasetDict` format with audio resampling to 16 kHz.
- Saved the dataset to `data/processed/alffa_sw/`.
- Saved metadata including character set, vocabulary size, and sample counts.

**Key Metrics from Phase 2**:
- Total utterances: 10,180
- Train: 8,144
- Validation: 1,018
- Test: 1,018
- Vocabulary size: 5,000
- Sample rate: 16,000 Hz

### Challenges Faced
- Audio paths in `wav.scp` did not match the actual file locations. Fixed by building a comprehensive index of all audio files in the dataset directory and using multiple resolution strategies.
- None of the audio paths resolved using the original method; implemented fallback strategies (basename matching, utterance ID matching, directory recursion).

### Next Steps
- Phase 3: Model Fine-Tuning on Google Colab or local GPU.
  - Load the preprocessed dataset.
  - Load a pre-trained wav2vec 2.0 model (`facebook/wav2vec2-xls-r-300m`).
  - Configure training arguments and run fine-tuning.
  - Save model checkpoints.
# KSL-to-Text Translator: Master To-Do List

**Project:** Real-time Kenyan Sign Language (KSL) to Text Translator  
**Approach:** Pose-Estimation (MediaPipe Holistic) + LSTM/Transformer  
**Dataset:** Kaggle KSL Video Dataset (2,237 videos, 30 classes)  
**Last Updated:** [Insert Today's Date]

---

## PHASE 0: Environment Setup
**Status:** COMPLETED

- [x] Installed Python 3.9+
- [x] Created and activated virtual environment (`ksl_env`)
- [x] Installed dependencies: `opencv-python`, `mediapipe`, `numpy`, `tensorflow`, `pandas`, `scikit-learn`
- [x] Created project folders: `data/`, `models/`, `scripts/`, `notebooks/`
- [x] Downloaded Kaggle dataset (currently downloading/ready)

---

## PHASE 1: Keypoint Extraction from Kaggle Videos
**Status:** COMPLETED

**Goal:** Convert all 2,237 downloaded `.mov` videos into `.npy` keypoint sequences using MediaPipe Holistic.

### Tasks:
- [x] **Verify dataset structure:** Confirm the downloaded folder contains `train/`, `val/`, and `test/` subfolders with 30 class folders inside.
- [x] **Update script variables:** Edit the extraction script with:
  - [x] `DATASET_PATH =` (the exact path where you saved the Kaggle folder, e.g., `C:/Users/YourName/Downloads/ksl-video-dataset/`)
  - [x] `VIDEO_EXTENSION =` (likely `.mov`, but check if it's `.mp4` or `.avi`)
- [x] **Run the extraction script:** Execute `extract_ksl_keypoints.py` to process all videos.
  - [x] Monitor progress (this will take 2–4 hours depending on your CPU).
- [x] **Verify output:** Ensure the script generates `processed_train/`, `processed_val/`, and `processed_test/` folders containing `.npy` files.
- [x] **Quick sanity check:** Load one `.npy` file in Python and confirm its shape is `(frames, 1662)`.

---

## PHASE 2: Data Preprocessing & Augmentation
**Status:** PLANNED

**Goal:** Clean the extracted keypoints, pad/truncate sequences to a fixed length, and apply augmentation.

### Tasks:
- [x] **Normalize keypoints:** Subtract shoulder coordinates from all points to make the model position-invariant.
- [x] **Fix sequence length:** Pad or truncate all sequences to exactly `SEQUENCE_LENGTH = 30` frames.
  - [x] Create a mapping dictionary for labels (0 to 29).
- [ ] **Load Kaggle splits:** Since the dataset is already split, load `processed_train/`, `processed_val/`, and `processed_test/` separately.
- [ ] **Data Augmentation:**
  - [ ] Horizontal flipping (mirror the x-coordinates to double your data).
  - [ ] Add small Gaussian noise (+/- 0.01) to coordinates.
- [ ] **Save final arrays:** Save as `X_train.npy`, `X_val.npy`, `X_test.npy`, and corresponding `y_train.npy`, `y_val.npy`, `y_test.npy` inside `data/`.

---

## PHASE 3: Model Architecture Design
**Status:** PLANNED

**Goal:** Build the deep learning model that maps pose sequences to sign labels.

### Tasks:
- [ ] **Choose architecture:** LSTM (baseline) or Transformer Encoder (more advanced).
- [ ] **Define model in TensorFlow/Keras:**
  - [ ] Input Shape: `(30, 1662)`.
  - [ ] Hidden Layers: LSTM(128) -> LSTM(64) -> Dense(32) -> Dropout(0.3).
  - [ ] Output Layer: `Dense(30, activation='softmax')` (because we have 30 sign classes).
- [ ] **Compile model:** Use `adam` optimizer and `categorical_crossentropy` loss.
- [ ] **Write `model.summary()`** and save the architecture diagram.

---

## PHASE 4: Model Training
**Status:** PLANNED

**Goal:** Train the model on the 1,487 training videos.

### Tasks:
- [ ] **Load data:** Load `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`.
- [ ] **Set training parameters:** Batch size = 32, Epochs = 100, Early Stopping (patience = 10).
- [ ] **Start training:** Run `model.fit()` and monitor validation accuracy.
- [ ] **Save the best model:** Use `ModelCheckpoint` to save the best weights as `models/ksl_best_model.h5`.
- [ ] **Plot loss & accuracy curves:** Save training history graphs for your report.

---

## PHASE 5: Evaluation
**Status:** PLANNED

**Goal:** Test your model on the 450 unseen videos in the Kaggle test set.

### Tasks:
- [ ] **Load test data:** Load `X_test.npy` and `y_test.npy`.
- [ ] **Run evaluation:** Use `model.evaluate()` to get final test accuracy and loss.
- [ ] **Generate Confusion Matrix:** Identify which signs the model confuses most often.
- [ ] **Document results:** Record final accuracy, precision, recall, and F1-score.

---

## PHASE 6: Real-Time Deployment
**Status:** PLANNED

**Goal:** Turn on your webcam and translate KSL signs in real-time.

### Tasks:
- [ ] **Write `realtime_translator.py`:**
  - [ ] Open webcam with OpenCV.
  - [ ] Extract MediaPipe keypoints from each frame.
  - [ ] Maintain a buffer of the last 30 frames.
  - [ ] Feed the buffer to the loaded model every 5 frames.
- [ ] **Display predictions:** Overlay the predicted text on the video feed.
- [ ] **Add a trigger key:** (e.g., press "Spacebar" to freeze and translate a sign).
- [ ] **Add a confidence threshold:** Only display predictions above 80% confidence.

---

## PHASE 7: Scaling to Continuous Translation (BONUS / FUTURE)
**Status:** PLANNED

**Goal:** Move from recognizing single words to translating full sentences (like AutoSign).

### Tasks:
- [ ] **Research CTC Loss:** Replace the current classification head with Connectionist Temporal Classification to align frames to text without per-frame labels.
- [ ] **Collect sentence data:** Find or record continuous KSL sentences (multiple signs in a row).
- [ ] **Integrate a Language Model:** Attach a lightweight text decoder (like a small GPT-2) to correct grammar.
- [ ] **Deploy:** Update the real-time script to handle streaming video without a fixed 30-frame window.

---

## Quick Reference: File Structure
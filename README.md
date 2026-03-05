# Arabic Spoken Command Recognition — SLP Project Fall 2025

**Spoken Language Processing — Fall 2025**  
Group: 1220871, 1220184, 1220031

Isolated **Arabic spoken command recognition** using MFCC features and machine learning (KNN, Random Forest, SVM, GMM). Commands: افتح، اغلق، ابدأ، توقف، يمين، يسار (open, close, start, stop, right, left).

## Repository structure

```
1220871_1220184_1220031_project/
├── README.md
├── Project/
│   ├── Dataset_Link.txt       # Link to dataset (Google Drive)
│   └── Code/
│       ├── preprocessing.py   # Audio conversion & resampling to 16 kHz
│       ├── clean_audio.py     # Silence removal & normalization
│       ├── compare_models.py           # LOSO (leave-one-speaker-out) evaluation
│       └── compare_models_randomsplit  # Random train/test split evaluation
```

## Requirements

- Python 3.x  
- **librosa**, **numpy**, **scikit-learn**, **matplotlib**  
- **FFmpeg** (for preprocessing; path may be set in code/config)

```bash
pip install librosa numpy scikit-learn matplotlib
```

## Dataset

- Arabic spoken commands, 6 classes, 16 kHz WAV.  
- Link: see `Project/Dataset_Link.txt`.

## How to run

1. Set `DATASET_DIR` (and FFmpeg path if needed) in the scripts.  
2. **LOSO evaluation:** `python compare_models.py`  
3. **Random split evaluation:** `python compare_models_randomsplit` (or `.py` if it’s a script)

Run from `Project/Code/` or adjust paths in the scripts.

## Models

- KNN, Random Forest, SVM, GMM  
- Features: 13 MFCC → mean + std → 26‑dim vector per utterance  

## Deliverables (course)

- Dataset link, source code, 2–4 page report (IEEE style).  
- Full project description: **Spoken_lang_Proc_Project_Fall2025.pdf**

## Authors

1220871, 1220184, 1220031

## License

For academic/educational use.

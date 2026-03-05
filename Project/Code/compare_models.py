# ============================================================
# compare_models_loso.py
# Train + evaluate ALL models using LOSO (leave-one-speaker-out)
# Models: KNN, Random Forest, SVM, GMM
# Features: MFCC (13) + mean/std  -> 26-dim vector 
#
# ============================================================

import os
import re
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture


# =========================
# SETTINGS
# =========================
DATASET_DIR = r"D:\spoken_rec_clean"
COMMANDS = ["aghliq", "ibda", "iftah", "tawaqaf", "yameen", "yasar"]

SR = 16000
N_MFCC = 13

RANDOM_STATE = 42


# =========================
# HELPERS
# =========================
def get_speaker_id(filename: str) -> str:
    """
    Extract speaker id from filename like: Aghliq10_h.wav, ibda3_m.wav, yasar7_b.wav
    Returns: 'h', 'm', 'b', ... (any id after last underscore)
    """
    base = os.path.basename(filename).lower()
    m = re.search(r"_([a-z0-9]+)\.wav$", base)   # accepts letters or digits
    if not m:
        raise ValueError(f"Cannot detect speaker in filename: {filename} (expected ..._speakerid.wav)")
    return m.group(1)



def extract_features(filepath: str) -> np.ndarray:
    """
    MFCC mean + std (fixed-length)
    Output size = 2*N_MFCC = 26 when N_MFCC=13
    """
    y, sr = librosa.load(filepath, sr=SR, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    return np.concatenate([mean, std], axis=0)


def gmm_predict_classwise(X_train, y_train, X_test, n_components=4):
    """
    Train one GMM per class on training data, then predict by max log-likelihood.
    """
    classes = np.unique(y_train)
    gmms = {}

    for c in classes:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            random_state=RANDOM_STATE
        )
        gmm.fit(X_train[y_train == c])
        gmms[c] = gmm

    preds = []
    for x in X_test:
        scores = {c: gmms[c].score(x.reshape(1, -1)) for c in classes}
        preds.append(max(scores, key=scores.get))

    return np.array(preds)


# =========================
# LOAD DATASET
# =========================
X, y, spk = [], [], []

for label, cmd in enumerate(COMMANDS):
    cmd_dir = os.path.join(DATASET_DIR, cmd)
    if not os.path.isdir(cmd_dir):
        raise FileNotFoundError(f"Folder not found: {cmd_dir}")

    for f in os.listdir(cmd_dir):
        if f.lower().endswith(".wav"):
            path = os.path.join(cmd_dir, f)
            X.append(extract_features(path))
            y.append(label)
            spk.append(get_speaker_id(f))

X = np.array(X)
y = np.array(y)
spk = np.array(spk)

print("Loaded dataset")
print("X shape:", X.shape)  # expected (180, 26)
print("y shape:", y.shape)
print("speakers:", sorted(set(spk)))


# =========================
# DEFINE MODELS
# =========================
models = {
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE
    ),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale"))
    ]),
    "GMM": None  # handled separately
}


# =========================
# LOSO EVALUATION
# =========================
speakers_unique = sorted(set(spk))

# store per-model accuracies
accs = {name: [] for name in models.keys()}

# store confusion matrices summed across folds
cms = {name: np.zeros((len(COMMANDS), len(COMMANDS)), dtype=int) for name in models.keys()}

for test_speaker in speakers_unique:
    train_idx = spk != test_speaker
    test_idx = spk == test_speaker

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("\n==============================")
    print(f"LOSO Fold - Test speaker: {test_speaker}")
    print("==============================")

    # ----- KNN / RF / SVM -----
    for name, model in models.items():
        if name == "GMM":
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_acc = accuracy_score(y_test, y_pred)
        accs[name].append(fold_acc)

        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(COMMANDS))))
        cms[name] += cm

        print(f"{name:13s} Accuracy: {fold_acc*100:.2f}%")

    # ----- GMM -----
    gmm_pred = gmm_predict_classwise(X_train, y_train, X_test, n_components=4)
    gmm_acc = accuracy_score(y_test, gmm_pred)
    accs["GMM"].append(gmm_acc)

    cm = confusion_matrix(y_test, gmm_pred, labels=list(range(len(COMMANDS))))
    cms["GMM"] += cm

    print(f"{'GMM':13s} Accuracy: {gmm_acc*100:.2f}%")


# =========================
# SUMMARY (Mean ± Std)
# =========================
print("\n==============================")
print("LOSO MODEL COMPARISON SUMMARY")
print("==============================")

summary = []
for name in models.keys():
    mean_acc = float(np.mean(accs[name]))
    std_acc = float(np.std(accs[name]))
    summary.append((name, mean_acc, std_acc))
    print(f"{name:13s}: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

best = max(summary, key=lambda t: t[1])
print("\n Best LOSO model (by mean accuracy):", best[0])


# =========================
# SAVE + PLOT CONFUSION MATRICES FOR ALL MODELS
# =========================
OUT_DIR = "cm_results"
os.makedirs(OUT_DIR, exist_ok=True)

for name in models.keys():
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cms[name],
        display_labels=COMMANDS
    )
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix (LOSO) - {name} (summed across folds)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, f"confusion_matrix_{name.replace(' ', '_').lower()}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


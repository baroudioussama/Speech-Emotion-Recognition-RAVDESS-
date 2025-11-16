# --------------------------------------------------------------
#  FULL SER (Speech Emotion Recognition) PIPELINE – RAVDESS
# --------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ----- imbalanced-learn (SMOTE) --------------------------------
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except Exception as e:                     # pip install imbalanced-learn
    print("imbalanced-learn not found – will use class-weights instead.")
    SMOTE_AVAILABLE = False

# ----- TensorFlow / Keras --------------------------------------
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy

# --------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------
DATA_PATH = r"D:\codes\audio"          # <-- change if needed
EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
DURATION = 3.0                         # seconds (fixed)
SR = None                              # will be inferred from file

# --------------------------------------------------------------
# 2. FEATURE EXTRACTION
# --------------------------------------------------------------
def extract_features(file_path):
    """Return a 147-dim feature vector (mean over time)."""
    y, sr = librosa.load(file_path, sr=SR, duration=DURATION, offset=0.5)
    # Pad / truncate to exactly DURATION seconds
    y = librosa.util.fix_length(y, size=int(sr * DURATION))

    # ----- MFCC + deltas (40 + 40 + 40 = 120) -----
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)                # (40, n_frames)
    delta = librosa.feature.delta(mfcc)                               # (40, n_frames)
    delta2 = librosa.feature.delta(mfcc, order=2)                     # (40, n_frames)
    mfcc_all = np.concatenate((mfcc, delta, delta2), axis=0)          # (120, n_frames)

    # ----- Extra acoustic features -----
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)                  # (12, n_frames)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)          # (7,  n_frames)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)                     # (6,  n_frames)
    zcr = librosa.feature.zero_crossing_rate(y)[0].reshape(1, -1)     # (1,  n_frames)
    rms = librosa.feature.rms(y=y)[0].reshape(1, -1)                  # (1,  n_frames)

    extra = np.concatenate([chroma, contrast, tonnetz, zcr, rms], axis=0)  # (27, n_frames)

    # ----- Combine & mean-pool over time -----
    combined = np.concatenate([mfcc_all, extra], axis=0)               # (147, n_frames)
    return np.mean(combined, axis=1)                                   # (147,)


# --------------------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------------------
features, labels = [], []

for fname in os.listdir(DATA_PATH):
    if not fname.endswith(".wav"):
        continue
    fp = os.path.join(DATA_PATH, fname)

    # extract emotion id from filename (e.g. 03-01-05-01-02-01-01.wav)
    emo_id = fname.split("-")[2]
    if emo_id not in EMOTION_MAP:
        continue

    try:
        feat = extract_features(fp)
        features.append(feat)
        labels.append(EMOTION_MAP[emo_id])
    except Exception as e:
        print(f"Warning: Skipping {fname} – {e}")

X = np.array(features)                     # (n_samples, 147)
y = np.array(labels)                       # (n_samples,)

print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features each.")
print("Class distribution:")
print(np.unique(y, return_counts=True))

# --------------------------------------------------------------
# 4. LABEL ENCODING & (optional) SMOTE
# --------------------------------------------------------------
le = LabelEncoder()
y_int = le.fit_transform(y)                # 0 … 7
y_cat = to_categorical(y_int)              # (n_samples, 8)

if SMOTE_AVAILABLE:
    sm = SMOTE(random_state=42)
    X_res, y_res_int = sm.fit_resample(X, y_int)
    y_res = to_categorical(y_res_int)
    print(f"After SMOTE: {X_res.shape[0]} samples")
else:
    X_res, y_res_int, y_res = X, y_int, y_cat
    print("SMOTE skipped – will use class-weights.")

# --------------------------------------------------------------
# 5. TRAIN / VAL SPLIT
# --------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_res, y_res, test_size=0.20, stratify=y_res_int, random_state=42
)

# --------------------------------------------------------------
# 6. CLASS WEIGHTS (always useful, even with SMOTE)
# --------------------------------------------------------------
cw = compute_class_weight('balanced', classes=np.unique(y_res_int), y=y_res_int)
class_weight_dict = dict(enumerate(cw))

# --------------------------------------------------------------
# 7. MODEL
# --------------------------------------------------------------
input_dim = X_train.shape[1]

model = Sequential([
    Dense(512, input_shape=(input_dim,), kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.4),

    Dense(256, kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.4),

    Dense(128, kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Dense(y_res.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.summary()

# --------------------------------------------------------------
# 8. CALLBACKS
# --------------------------------------------------------------
es = EarlyStopping(monitor='val_accuracy', patience=15,
                   restore_best_weights=True, verbose=1)
rl = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                       patience=7, min_lr=1e-7, verbose=1)

# --------------------------------------------------------------
# 9. TRAIN
# --------------------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=[es, rl],
    verbose=2
)

# --------------------------------------------------------------
# 10. PLOT TRAINING CURVES
# --------------------------------------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 11. FINAL EVALUATION
# --------------------------------------------------------------
y_pred = model.predict(X_val).argmax(axis=1)
y_true = y_val.argmax(axis=1)

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
# %% [markdown]
# # Theme 5 – Edge AI & TinyML: Keyword Spotting on a Microcontroller
#
# **Task:** recognise spoken keywords (the kit's classes, e.g. yes / no / up / … / silence / unknown)
# from short audio clips, shrink the model to 8-bit, and run it live on a microcontroller.
#
# | Tier | Target | Where in this notebook |
# |---|---|---|
# | Foundation | accuracy ≥ `TARGETS['accuracy']` (85 %) | Sections 1–5 |
# | Advanced | int8 model ≤ `TARGETS['max_kb']` (250 KB), drop ≤ `TARGETS['max_drop_points']` (2) | Section 6 |
# | Hard path | live on the board, 8/10 correct within 200 ms | Section 7 + `kws_live.ino` |
#
# **Network:** DS-CNN (depthwise-separable CNN, "Hello Edge", Zhang et al. 2017), width set by `CHANNELS` (64 → about 23 k parameters).
#
# Run all cells top to bottom (Runtime → Run all). On a Colab GPU the whole notebook takes about 3–5 minutes.

# %% [markdown]
# ## 0. Settings – the only cell you should need to edit

# %%
# Nothing about the data is hard-coded: the kit folder, class names, feature shape and MFCC settings are all
# found or worked out from what's in the kit. Override any of these with an environment variable or by editing here.
import os, glob, re, json, hashlib, time

def _env(name, default, cast=str):
    v = os.environ.get(name); return cast(v) if v not in (None, "") else default

KIT_DIR = _env("KIT_DIR", None)          # None = search for the kit automatically
FEATURE_SOURCE = _env("FEATURE_SOURCE", "auto")   # "kit" | "wav" | "auto"
CHANNELS = _env("CHANNELS", 64, int)     # DS-CNN width (64 ≈ 23k params; 32 if the board is too slow)
EPOCHS = _env("EPOCHS", 40, int)
BATCH = _env("BATCH", 128, int)
SEED = _env("SEED", 42, int)
# Pass marks from the hackathon outline. Change them here if the organisers announce different targets.
TARGETS = dict(accuracy=_env("TARGET_ACC", 0.85, float), max_kb=_env("TARGET_KB", 250.0, float),
               max_drop_points=_env("TARGET_DROP", 2.0, float))

def find_kit_dir():
    """Look in the usual places for a folder holding feature files (.npz/.npy/.csv) or .wav files."""
    roots = [os.getcwd(), "/content", "/content/drive/MyDrive", "/kaggle/input", os.path.expanduser("~")]
    best = None
    for r in roots:
        if not os.path.isdir(r): continue
        for dirpath, dirnames, files in os.walk(r):
            if dirpath.count(os.sep) - r.count(os.sep) > 3 or "sample_data" in dirpath or "/." in dirpath:
                dirnames[:] = []; continue
            if any(f.endswith((".npz", ".npy")) for f in files) or \
               any(f.endswith(".csv") and re.search(r"train|mfcc|feat", f, re.I) for f in files):
                return dirpath
            if best is None and sum(f.endswith(".wav") for f in files) > 20:
                best = os.path.dirname(dirpath)          # .../audio/<class>/*.wav -> .../audio
    return best

# If the kit was uploaded as a .zip (e.g. to /content), unzip it next to itself first
import zipfile
for _root in (os.getcwd(), "/content", "/content/drive/MyDrive"):
    for _z in glob.glob(os.path.join(_root, "*.zip")):
        _dst = os.path.splitext(_z)[0]
        if not os.path.isdir(_dst):
            print("Unzipping", _z); zipfile.ZipFile(_z).extractall(_dst)
if KIT_DIR is None: KIT_DIR = find_kit_dir()
print("Kit folder:", KIT_DIR)

# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score

np.random.seed(SEED); tf.random.set_seed(SEED)
print("TensorFlow", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))
os.makedirs("out", exist_ok=True)

# %% [markdown]
# ## 1. Feature extractor (MFCC)
#
# We use one MFCC definition for both training and the microcontroller. The C++ code in `kws_live.ino`
# runs exactly these steps with the same numbers, so the board sees the same kind of input the model was trained on.
#
# clip → overlapping frames → Hann window → FFT → power → mel filterbank → log → DCT → keep the first
# N_MFCC coefficients → a **frames × coefficients** "image" (49 × 10 with the usual settings).

# %%
# Starting values are the common TinyML keyword-spotting settings. If the kit includes a config/README file
# (json, yaml or txt) giving its MFCC settings, those replace the starting values, so our features and the
# board's features match the kit's.
FEAT = dict(sample_rate=16000, clip_seconds=1.0, frame_ms=30.0, hop_ms=20.0, n_fft=512, n_mels=40,
            f_lo=20.0, f_hi=4000.0, n_mfcc=10)
_ALIASES = {"sample_rate": ["sample_rate", "sr", "sampling_rate", "fs"],
            "frame_ms": ["frame_ms", "window_ms", "win_ms", "window_size_ms", "frame_length_ms"],
            "hop_ms": ["hop_ms", "stride_ms", "window_stride_ms", "frame_stride_ms", "hop_length_ms"],
            "n_fft": ["n_fft", "nfft", "fft_length", "fft_size"],
            "n_mels": ["n_mels", "num_mel_bins", "mel_bins", "num_filters", "nfilt"],
            "f_lo": ["f_lo", "fmin", "lower_edge_hertz", "low_freq"],
            "f_hi": ["f_hi", "fmax", "upper_edge_hertz", "high_freq"],
            "n_mfcc": ["n_mfcc", "num_mfcc", "dct_coefficient_count", "num_cepstral", "numcep"]}

def read_kit_feature_config(kit_dir):
    found = {}
    if not kit_dir: return found
    files = [f for ext in ("json", "yaml", "yml", "txt", "md", "cfg") for f in
             glob.glob(os.path.join(kit_dir, "**", f"*.{ext}"), recursive=True) if os.path.getsize(f) < 200_000]
    for f in files:
        text = open(f, errors="ignore").read()
        for key, names in _ALIASES.items():
            for nm in names:
                m = re.search(rf"[\"']?{nm}[\"']?\s*[:=]\s*([0-9.]+)", text, re.I)
                if m and key not in found: found[key] = float(m.group(1)); break
        # settings given in samples instead of ms
        for nm, key in (("frame_length", "frame_ms"), ("win_length", "frame_ms"), ("hop_length", "hop_ms")):
            m = re.search(rf"[\"']?{nm}[\"']?\s*[:=]\s*([0-9]+)\b(?!\s*ms)", text, re.I)
            if m and key not in found: found[key + "_samples"] = int(m.group(1))
    sr = found.get("sample_rate", FEAT["sample_rate"])
    for key in ("frame_ms", "hop_ms"):
        if key + "_samples" in found: found[key] = found.pop(key + "_samples") * 1000.0 / sr
    return {k: v for k, v in found.items() if k in FEAT}

_kit_cfg = read_kit_feature_config(KIT_DIR)
FEAT.update(_kit_cfg)
print("MFCC settings read from kit:", _kit_cfg or "none found – using starting values")

SR = int(FEAT["sample_rate"]); CLIP = int(round(SR * FEAT["clip_seconds"]))
FRAME_LEN = int(round(SR * FEAT["frame_ms"] / 1000)); HOP = int(round(SR * FEAT["hop_ms"] / 1000))
NFFT = int(max(FEAT["n_fft"], 2 ** int(np.ceil(np.log2(FRAME_LEN)))))     # FFT must cover the frame
N_MEL, N_MFCC = int(FEAT["n_mels"]), int(FEAT["n_mfcc"])
F_LO, F_HI = float(FEAT["f_lo"]), float(min(FEAT["f_hi"], SR / 2))
N_FRAMES = 1 + (CLIP - FRAME_LEN) // HOP
print(f"→ {SR} Hz, frame {FRAME_LEN} / hop {HOP} samples, FFT {NFFT}, {N_MEL} mels, "
      f"{N_MFCC} MFCCs, {N_FRAMES} frames per clip")

def _hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
def _mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def mel_filterbank():
    mels = np.linspace(_hz2mel(F_LO), _hz2mel(F_HI), N_MEL + 2)
    hz = _mel2hz(mels)
    bin_hz = np.arange(NFFT // 2 + 1) * SR / NFFT
    fb = np.zeros((N_MEL, NFFT // 2 + 1), np.float32)
    for m in range(N_MEL):
        lo, c, hi = hz[m], hz[m + 1], hz[m + 2]
        up = (bin_hz - lo) / (c - lo)
        down = (hi - bin_hz) / (hi - c)
        fb[m] = np.maximum(0, np.minimum(up, down))
    return fb

WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME_LEN) / FRAME_LEN)).astype(np.float32)
MEL_FB = mel_filterbank()
k = np.arange(N_MFCC)[:, None]; n = np.arange(N_MEL)[None, :]
DCT = (np.cos(np.pi * k * (2 * n + 1) / (2 * N_MEL)) * np.sqrt(2.0 / N_MEL)).astype(np.float32)
DCT[0] *= np.sqrt(0.5)          # orthonormal DCT-II

def mfcc(audio):
    """audio: float array in [-1, 1] at SR. Returns (N_FRAMES, N_MFCC)."""
    audio = np.asarray(audio, np.float32)
    if len(audio) < CLIP: audio = np.pad(audio, (0, CLIP - len(audio)))
    audio = audio[:CLIP]
    frames = np.stack([audio[i * HOP: i * HOP + FRAME_LEN] for i in range(N_FRAMES)]) * WINDOW
    power = np.abs(np.fft.rfft(frames, NFFT)) ** 2
    logmel = np.log(power @ MEL_FB.T + 1e-6)
    return (logmel @ DCT.T).astype(np.float32)

# %% [markdown]
# ## 2. Load the data
#
# The loader looks through the kit folder and works with whatever layout it finds: `.npz`, separate `.npy` files,
# or `.csv`. Labels can be strings or integers. If the kit has no validation split, we split one off the training
# set. If the test labels are held back by the organisers, we still write predictions for them.

# %%
def _split_of(name):
    n = os.path.basename(str(name)).lower()
    if re.search(r"test", n): return "test"
    if re.search(r"val|valid|dev", n): return "val"
    if re.search(r"train", n): return "train"
    return None

def _is_label(name):
    return bool(re.search(r"(^|[_\-/])(y|label|labels|target|targets|class|classes|word|words)([_\-.]|$)",
                          os.path.basename(str(name)).lower()))

def _looks_like_names(arr):
    a = np.asarray(arr)
    return a.ndim == 1 and a.dtype.kind in "US O" and len(a) <= 1000 and len(set(map(str, a))) == len(a)

CLASS_NAMES_FROM_KIT = None

def read_class_names(kit_dir):
    """Class-name list from a labels/classes file in the kit (txt one-per-line, or json list/dict)."""
    if not kit_dir: return None
    for f in glob.glob(os.path.join(kit_dir, "**", "*"), recursive=True):
        b = os.path.basename(f).lower()
        if not re.search(r"(label|class)", b) or not b.endswith((".txt", ".json")): continue
        try:
            if b.endswith(".json"):
                j = json.load(open(f))
                if isinstance(j, list): return [str(x) for x in j]
                if isinstance(j, dict):     # {"yes": 0, ...} or {"0": "yes", ...}
                    if all(isinstance(v, int) for v in j.values()): return [k for k, _ in sorted(j.items(), key=lambda kv: kv[1])]
                    return [str(j[k]) for k in sorted(j, key=lambda k: int(k))]
            else:
                lines = [l.strip() for l in open(f) if l.strip()]
                if 2 <= len(lines) <= 1000 and all(len(l) < 40 for l in lines): return lines
        except Exception:
            pass
    return None

def load_kit(kit_dir):
    """Returns dict split -> (X, y_or_None). Supports .npz, .npy pairs and .csv, any naming."""
    global CLASS_NAMES_FROM_KIT
    out = {}
    def put(sp, is_y, arr, key):
        global CLASS_NAMES_FROM_KIT
        if is_y and _looks_like_names(arr) and not any(len(arr) == len(v[0]) for v in out.values() if v[0] is not None):
            CLASS_NAMES_FROM_KIT = [str(x) for x in arr]; return          # it's a class-name list, not labels
        out.setdefault(sp, [None, None])[1 if is_y else 0] = arr
    for f in sorted(glob.glob(os.path.join(kit_dir, "**", "*.npz"), recursive=True)):
        d = np.load(f, allow_pickle=True)
        for key in sorted(d.files, key=_is_label):                        # features first
            put(_split_of(key) or _split_of(f) or "train", _is_label(key), d[key], key)
    for f in sorted(glob.glob(os.path.join(kit_dir, "**", "*.npy"), recursive=True), key=_is_label):
        put(_split_of(f) or "train", _is_label(f), np.load(f, allow_pickle=True), f)
    for f in sorted(glob.glob(os.path.join(kit_dir, "**", "*.csv"), recursive=True)):
        import pandas as pd
        df = pd.read_csv(f)
        lab_cols = [c for c in df.columns if _is_label(c)] or [c for c in df.columns if df[c].dtype == object]
        num = df.drop(columns=lab_cols).select_dtypes("number")
        if num.shape[1] < 10: continue                                   # not a feature table
        out[_split_of(f) or "train"] = [num.values, df[lab_cols[0]].values if lab_cols else None]
    if CLASS_NAMES_FROM_KIT is None: CLASS_NAMES_FROM_KIT = read_class_names(kit_dir)
    return {k: (v[0], v[1]) for k, v in out.items() if v[0] is not None}

# Optional: map a raw Speech Commands folder layout onto the task's classes. Set TARGET_WORDS (comma-separated),
# e.g. "yes,no,up,down,left,right,on,off,stop,go": other words -> "unknown", noise folders -> "silence".
TARGET_WORDS = [w for w in _env("TARGET_WORDS", "").split(",") if w]

def find_wav_root(kit_dir):
    if not kit_dir: return None
    for dirpath, dirnames, _ in os.walk(kit_dir):
        n_cls = sum(1 for d in dirnames if glob.glob(os.path.join(dirpath, d, "*.wav")))
        if n_cls >= 2: return dirpath
    return None

def load_wavs(wav_dir):
    """Folder-per-class wav loader. Uses the dataset's validation_list.txt / testing_list.txt if present,
    otherwise splits by speaker (hash of the speaker id, as Google does), so a speaker is never in two splits."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    lists = {}
    for sp, nm in (("val", "validation_list.txt"), ("test", "testing_list.txt")):
        f = os.path.join(wav_dir, nm)
        if os.path.exists(f): lists.update({l.strip(): sp for l in open(f) if l.strip()})
    X, y, split = [], [], []
    for cls in sorted(d for d in os.listdir(wav_dir) if os.path.isdir(os.path.join(wav_dir, d))):
        is_noise = cls.startswith("_")
        label = cls.strip("_")
        if TARGET_WORDS: label = "silence" if is_noise else (cls if cls in TARGET_WORDS else "unknown")
        for f in sorted(glob.glob(os.path.join(wav_dir, cls, "*.wav"))):
            sr, a = wavfile.read(f)
            a = a.astype(np.float32) / (np.iinfo(a.dtype).max + 1.0 if a.dtype.kind == "i" else 1.0)
            if a.ndim > 1: a = a.mean(1)
            if sr != SR: a = resample_poly(a, SR, sr).astype(np.float32)
            rel = f"{cls}/{os.path.basename(f)}"
            speaker = os.path.basename(f).split("_nohash_")[0]
            h = int(hashlib.sha1(speaker.encode()).hexdigest(), 16) % 100
            sp = lists.get(rel) or ("val" if h < 10 else "test" if h < 20 else "train")
            # long recordings (e.g. background noise) become many clip-length chunks
            chunks = [a[i:i + CLIP] for i in range(0, len(a) - CLIP + 1, CLIP)] if len(a) >= 2 * CLIP else [a]
            for j, c in enumerate(chunks):
                X.append(mfcc(c)); y.append(label)
                split.append(sp if len(chunks) == 1 else ("val", "test", *["train"] * 8)[j % 10])
    X, y, split = np.array(X), np.array(y), np.array(split)
    return {s: (X[split == s], y[split == s]) for s in ("train", "val", "test") if (split == s).any()}

def synthetic_dry_run(n_classes=12, n_per_class=300):
    """Fake but learnable data so the whole notebook can be tested before the kit is available."""
    rng = np.random.default_rng(SEED)
    protos = rng.normal(0, 1, (n_classes, N_FRAMES, N_MFCC))
    max_shift = max(1, N_FRAMES // 10)
    def make(n):
        X, y = [], []
        for c in range(n_classes):
            for _ in range(n):
                X.append(np.roll(protos[c], rng.integers(-max_shift, max_shift + 1), 0) + rng.normal(0, 1.4, protos[c].shape))
                y.append(f"class_{c}")
        return np.array(X, np.float32), np.array(y)
    return {"train": make(n_per_class), "val": make(n_per_class // 6), "test": make(n_per_class // 6)}

src, data = FEATURE_SOURCE, {}
if src in ("auto", "kit") and KIT_DIR and os.path.isdir(KIT_DIR):
    data = load_kit(KIT_DIR); src = "kit" if data else src
WAV_DIR = find_wav_root(KIT_DIR)
if not data and src in ("auto", "wav") and WAV_DIR:
    print("Computing MFCCs from wav files in", WAV_DIR); data = load_wavs(WAV_DIR); src = "wav"
if not data:
    print("⚠️  No kit found – using SYNTHETIC data (dry run only, scores mean nothing).")
    data = synthetic_dry_run(); src = "synthetic"
print("Feature source:", src)
for k_, (X_, y_) in data.items():
    print(f"  {k_:5s} X{tuple(np.shape(X_))}  labels: {'present' if y_ is not None else 'HELD BY ORGANISERS'}")

# %% [markdown]
# ### Tidy up: reshape to (N, frames, coefficients, 1), encode labels, make sure we have a validation split
# The shape comes from the data itself, so the number of frames and coefficients is never assumed.

# %%
def to_4d(X):
    X = np.asarray(X, np.float32)
    if X.ndim == 4 and X.shape[-1] != 1: X = X.mean(-1, keepdims=True)
    if X.ndim == 2:                                           # flattened rows -> use the known MFCC count
        if X.shape[1] % N_MFCC: raise ValueError(f"{X.shape[1]} values per row isn't a multiple of n_mfcc={N_MFCC}; "
                                                 "set N_MFCC via the kit config or the n_mfcc entry in FEAT")
        X = X.reshape(len(X), -1, N_MFCC)
    if X.ndim == 3:
        if X.shape[2] != N_MFCC and X.shape[1] == N_MFCC: X = X.transpose(0, 2, 1)   # (N, coeff, frames)
        elif X.shape[2] != N_MFCC and X.shape[1] < X.shape[2]: X = X.transpose(0, 2, 1)
        X = X[..., None]
    return X

def to_labels(y):
    y = np.asarray(y)
    if y.ndim == 2: y = y.argmax(1)                           # one-hot
    return y

X_train, y_train_raw = to_4d(data["train"][0]), to_labels(data["train"][1])
if "val" in data and data["val"][1] is not None:
    X_val, y_val_raw = to_4d(data["val"][0]), to_labels(data["val"][1])
else:
    X_train, X_val, y_train_raw, y_val_raw = train_test_split(
        X_train, y_train_raw, test_size=0.1, stratify=y_train_raw, random_state=SEED)
X_test = to_4d(data["test"][0]) if "test" in data else None
y_test_raw = to_labels(data["test"][1]) if "test" in data and data["test"][1] is not None else None

DATA_FRAMES, DATA_MFCC = X_train.shape[1], X_train.shape[2]
if (DATA_FRAMES, DATA_MFCC) != (N_FRAMES, N_MFCC):
    print(f"⚠️  Data is {DATA_FRAMES}×{DATA_MFCC} but the MFCC settings give {N_FRAMES}×{N_MFCC}. Training uses the data's "
          "shape; for the live demo, put the kit's real MFCC settings in FEAT (Section 1) so the board matches.")
N_FRAMES, N_MFCC = DATA_FRAMES, DATA_MFCC

# Class names: string labels name themselves; integer labels use the kit's class list if it has one
if y_train_raw.dtype.kind in "iu":
    n = int(max(y_train_raw.max(), y_val_raw.max())) + 1
    CLASSES = CLASS_NAMES_FROM_KIT if CLASS_NAMES_FROM_KIT and len(CLASS_NAMES_FROM_KIT) >= n else [f"class_{i}" for i in range(n)]
    enc = lambda y: np.asarray(y, np.int64)
else:
    seen = sorted(set(map(str, y_train_raw)))
    CLASSES = [c for c in (CLASS_NAMES_FROM_KIT or []) if c in seen] + [c for c in seen if c not in (CLASS_NAMES_FROM_KIT or [])]
    idx = {c: i for i, c in enumerate(CLASSES)}
    enc = lambda y: np.array([idx[str(v)] for v in y], np.int64)
NUM_CLASSES = len(CLASSES)
y_train, y_val = enc(y_train_raw), enc(y_val_raw)
y_test = enc(y_test_raw) if y_test_raw is not None else None
print("Classes:", CLASSES)
print("Train", X_train.shape, "Val", X_val.shape, "Test", None if X_test is None else X_test.shape)
print("Train class counts:", dict(zip(CLASSES, np.bincount(y_train, minlength=NUM_CLASSES).tolist())))

# %% [markdown]
# ### Normalise each MFCC coefficient using **training-set statistics only**
# Taking the mean and std from the training set alone means no test information leaks in. The same per-coefficient
# means and stds get copied to the microcontroller.

# %%
MEAN = X_train.mean(axis=(0, 1, 3)).astype(np.float32)   # shape (N_MFCC,)
STD = (X_train.std(axis=(0, 1, 3)) + 1e-6).astype(np.float32)
norm = lambda X: ((X - MEAN[None, None, :, None]) / STD[None, None, :, None]).astype(np.float32)
Xtr, Xva = norm(X_train), norm(X_val)
Xte = norm(X_test) if X_test is not None else None

fig, axs = plt.subplots(1, 4, figsize=(14, 2.6))
for ax, c in zip(axs, range(4)):
    i = np.where(y_train == c)[0][0]
    ax.imshow(Xtr[i, :, :, 0].T, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(CLASSES[c]); ax.set_xlabel("frame"); ax.set_ylabel("MFCC")
plt.tight_layout(); plt.savefig("out/mfcc_examples.png", dpi=120); plt.show()

# %% [markdown]
# ## 3. Baseline vs our model
#
# **Baseline (for comparison):** a plain dense network on the flattened features. Expect roughly 70–80 %.
#
# **Our model, DS-CNN:** a normal conv followed by 4 *depthwise-separable* blocks. A depthwise-separable conv splits
# a 3×3 conv into a per-channel 3×3 filter and a 1×1 mixing step. That uses about 8× fewer parameters and multiplies
# for almost the same accuracy, which is why it's the standard network for microcontrollers.
#
# What we changed from the baseline, and why:
# 1. **Convolutions** – a spoken word is a pattern in time and frequency, and convs pick it up wherever it falls in the clip.
# 2. **BatchNorm** – faster, more stable training. It folds into the conv weights at export, so it costs nothing on the chip.
# 3. **Dropout 0.3 + SpecAugment** (random time/frequency masks and time shifts) – reduces overfitting to the training speakers.
# 4. **Early stopping on validation** – keeps the best epoch. The test set is never looked at during training.

# %%
def baseline_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input((N_FRAMES, N_MFCC, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")], name="baseline_dnn")

def ds_cnn(ch=CHANNELS, blocks=4, batch=None):
    L = tf.keras.layers
    inp = L.Input((N_FRAMES, N_MFCC, 1), batch_size=batch)
    # first-layer kernel scales with the input: ~1/5 of the frames (≈200 ms of audio) × ~2/5 of the coefficients
    k_t, k_f = max(3, round(N_FRAMES / 5)), max(2, round(N_MFCC * 0.4))
    x = L.Conv2D(ch, (k_t, k_f), strides=(2, 2), padding="same", use_bias=False)(inp)
    x = L.BatchNormalization(momentum=0.9)(x); x = L.ReLU()(x)
    for _ in range(blocks):
        x = L.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization(momentum=0.9)(x); x = L.ReLU()(x)
        x = L.Conv2D(ch, 1, use_bias=False)(x)
        x = L.BatchNormalization(momentum=0.9)(x); x = L.ReLU()(x)
    x = L.Dropout(0.3)(x)
    x = L.AveragePooling2D(pool_size=(x.shape[1], x.shape[2]))(x)   # global average pool, MCU-friendly op
    x = L.Reshape((ch,))(x)          # fixed-size reshape (Flatten adds extra SHAPE/PACK ops on the MCU)
    out = L.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="ds_cnn")

model = ds_cnn()
model.summary()

# %% [markdown]
# ### Data augmentation (training only)

# %%
SHIFT, TMASK, FMASK = max(1, N_FRAMES // 10), max(1, N_FRAMES // 8), max(1, N_MFCC // 5)

def augment(x, y):
    # random time shift of up to ±10 % of the clip
    s = tf.random.uniform([], -SHIFT, SHIFT + 1, tf.int32)
    x = tf.roll(x, s, axis=0)
    # time mask: blank out up to ~12 % of the frames
    t0 = tf.random.uniform([], 0, N_FRAMES, tf.int32); tw = tf.random.uniform([], 0, TMASK + 1, tf.int32)
    tmask = tf.logical_or(tf.range(N_FRAMES) < t0, tf.range(N_FRAMES) >= t0 + tw)
    x = x * tf.cast(tmask, x.dtype)[:, None, None]
    # frequency mask: blank out up to ~20 % of the coefficients
    f0 = tf.random.uniform([], 0, N_MFCC, tf.int32); fw = tf.random.uniform([], 0, FMASK + 1, tf.int32)
    fmask = tf.logical_or(tf.range(N_MFCC) < f0, tf.range(N_MFCC) >= f0 + fw)
    x = x * tf.cast(fmask, x.dtype)[None, :, None]
    x = x + tf.random.normal(tf.shape(x), stddev=0.05)
    return x, y

AUTOTUNE = tf.data.AUTOTUNE
ds_train = (tf.data.Dataset.from_tensor_slices((Xtr, y_train)).shuffle(len(Xtr), seed=SEED)
            .map(augment, num_parallel_calls=AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE))
ds_val = tf.data.Dataset.from_tensor_slices((Xva, y_val)).batch(512)

# %% [markdown]
# ## 4. Train

# %%
# Baseline: a few quick epochs so we have a number to beat
base = baseline_model()
base.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
base.fit(Xtr, y_train, validation_data=(Xva, y_val), epochs=10, batch_size=BATCH, verbose=0)
base_val_acc = base.evaluate(Xva, y_val, verbose=0)[1]
print(f"Baseline DNN val accuracy: {base_val_acc:.3f}")

steps = int(np.ceil(len(Xtr) / BATCH)) * EPOCHS
lr = tf.keras.optimizers.schedules.CosineDecay(3e-3, steps, alpha=0.02)
model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cb = [tf.keras.callbacks.EarlyStopping("val_accuracy", patience=8, restore_best_weights=True)]
t0 = time.time()
hist = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=cb, verbose=2)
print(f"Training took {time.time() - t0:.0f} s")

plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1); plt.plot(hist.history["accuracy"], label="train"); plt.plot(hist.history["val_accuracy"], label="val")
plt.title("Accuracy"); plt.legend(); plt.xlabel("epoch")
plt.subplot(1, 2, 2); plt.plot(hist.history["loss"], label="train"); plt.plot(hist.history["val_loss"], label="val")
plt.title("Loss"); plt.legend(); plt.xlabel("epoch")
plt.tight_layout(); plt.savefig("out/training_curves.png", dpi=120); plt.show()
model.save("out/ds_cnn_float.keras")

# %% [markdown]
# ## 5. Foundation result – held-out test set

# %%
EVAL_X, EVAL_y, EVAL_NAME = (Xte, y_test, "TEST") if y_test is not None else (Xva, y_val, "VALIDATION (test labels held by organisers)")
probs = model.predict(EVAL_X, batch_size=512, verbose=0)
pred = probs.argmax(1)
float_acc = (pred == EVAL_y).mean()
print(f"=== {EVAL_NAME} accuracy (float32): {float_acc*100:.2f} %   target ≥ {TARGETS['accuracy']*100:.0f} %   "
      f"{'✅ PASS' if float_acc >= TARGETS['accuracy'] else '❌'}")
print(f"Macro F1: {f1_score(EVAL_y, pred, average='macro'):.3f}   | baseline DNN (val): {base_val_acc*100:.1f} %")
print(classification_report(EVAL_y, pred, labels=range(NUM_CLASSES), target_names=CLASSES, digits=3, zero_division=0))

cm = confusion_matrix(EVAL_y, pred, labels=range(NUM_CLASSES))
cmn = cm / cm.sum(1, keepdims=True).clip(1)
plt.figure(figsize=(7.5, 6.5))
plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
plt.xticks(range(NUM_CLASSES), CLASSES, rotation=45, ha="right"); plt.yticks(range(NUM_CLASSES), CLASSES)
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if cm[i, j]: plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                              color="white" if cmn[i, j] > 0.5 else "black")
plt.xlabel("predicted"); plt.ylabel("true"); plt.title(f"Confusion matrix – {EVAL_NAME.split()[0]} acc {float_acc*100:.1f}%")
plt.colorbar(fraction=0.046); plt.tight_layout(); plt.savefig("out/confusion_matrix.png", dpi=150); plt.show()

# If the test labels are held back, save predictions for the organisers
if X_test is not None and y_test is None:
    tp = model.predict(Xte, verbose=0).argmax(1)
    np.savetxt("out/test_predictions.csv", np.array(CLASSES)[tp], fmt="%s")
    print("Saved out/test_predictions.csv")

# %% [markdown]
# ### Failure-case analysis (worth 20 marks for "Insight")
# Find the pairs of classes the model mixes up most often, and show the most confident mistakes.

# %%
off = cm.copy(); np.fill_diagonal(off, 0)
pairs = sorted([(off[i, j], CLASSES[i], CLASSES[j]) for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if off[i, j]], reverse=True)[:5]
print("Most common confusions (count, true → predicted):")
for c, a, b in pairs: print(f"  {c:4d}  {a:>8s} → {b}")

wrong = np.where(pred != EVAL_y)[0]
worst = wrong[np.argsort(-probs[wrong].max(1))][:4]
fig, axs = plt.subplots(1, len(worst), figsize=(14, 2.8))
for ax, i in zip(np.atleast_1d(axs), worst):
    ax.imshow(EVAL_X[i, :, :, 0].T, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(f"true: {CLASSES[EVAL_y[i]]}\npred: {CLASSES[pred[i]]} ({probs[i].max():.2f})", fontsize=9)
    ax.set_xlabel("frame")
plt.suptitle("Most confident mistakes"); plt.tight_layout(); plt.savefig("out/failure_cases.png", dpi=120); plt.show()

# %% [markdown]
# ### Failure-case summary, written from *your* results
# This cell writes the story for slide 3 using the numbers above. Nothing in it is pre-written.

# %%
recall = cmn.diagonal()
worst_cls = np.argsort(recall)[:3]
conf_right, conf_wrong = probs[pred == EVAL_y].max(1).mean(), probs[pred != EVAL_y].max(1).mean() if len(wrong) else float("nan")
lines = [f"Overall {EVAL_NAME.split()[0].lower()} accuracy is {float_acc*100:.1f}% on {len(EVAL_y)} clips "
         f"({len(wrong)} mistakes)."]
lines.append("Weakest classes (recall): " + ", ".join(f"'{CLASSES[i]}' {recall[i]*100:.0f}%" for i in worst_cls) + ".")
for c, a_, b_ in pairs[:3]:
    tot = cm[CLASSES.index(a_)].sum()
    lines.append(f"'{a_}' was predicted as '{b_}' {c} times ({c/tot*100:.1f}% of all '{a_}' clips).")
catch_all = [c for c in CLASSES if c.lower() in ("unknown", "_unknown_", "other")]
if catch_all:
    u = CLASSES.index(catch_all[0])
    into_u = off[:, u].sum(); from_u = off[u].sum()
    lines.append(f"{into_u} keyword clips were sent to '{catch_all[0]}' and {from_u} '{catch_all[0]}' clips were mistaken "
                 f"for keywords. A catch-all class covers many different words, so its boundary is the hardest to learn.")
lines.append(f"On average the model is {conf_right*100:.0f}% confident when it's right and {conf_wrong*100:.0f}% when "
             f"it's wrong, so a confidence threshold on the board will filter out some errors.")
lines.append("Why these pairs: listen to a few of the clips in the failure plot. Confused words usually share their "
             "vowel, and the sound that tells them apart is short and quiet, so it gets lost after MFCC compression and background noise.")
FAILURE_STORY = "\n".join("• " + l for l in lines)
print(FAILURE_STORY)
open("out/failure_story.txt", "w").write(FAILURE_STORY)

# %% [markdown]
# ## 6. Advanced – full 8-bit (int8) quantisation
#
# Post-training quantisation turns every weight and activation from 32-bit float into 8-bit integers, using
# about 300 training clips as a *representative dataset* to measure the range of each activation. That makes the
# model about 4× smaller and much faster on a chip with no FPU for int maths (CMSIS-NN kernels).

# %%
def rep_data():
    idx = np.random.default_rng(0).choice(len(Xtr), min(500, len(Xtr)), replace=False)
    for i in idx: yield [Xtr[i:i + 1]]

# Same network with the batch size fixed at 1 (what the board runs) and the trained weights copied in,
# so the converter emits only the core ops
model_b1 = ds_cnn(batch=1); model_b1.set_weights(model.get_weights())
conv = tf.lite.TFLiteConverter.from_keras_model(model_b1)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_data
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tflite_int8 = conv.convert()
open("out/kws_int8.tflite", "wb").write(tflite_int8)

conv_f = tf.lite.TFLiteConverter.from_keras_model(model_b1)
tflite_f32 = conv_f.convert()
open("out/kws_float.tflite", "wb").write(tflite_f32)

def tflite_eval(tfl, X):
    it = tf.lite.Interpreter(model_content=tfl); it.allocate_tensors()
    i_d, o_d = it.get_input_details()[0], it.get_output_details()[0]
    preds = []
    for x in X:
        x = x[None]
        if i_d["dtype"] == np.int8:
            s, z = i_d["quantization"]; x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        it.set_tensor(i_d["index"], x); it.invoke()
        preds.append(it.get_tensor(o_d["index"])[0].argmax())
    return np.array(preds)

t0 = time.time()
pred_q = tflite_eval(tflite_int8, EVAL_X)
q_acc = (pred_q == EVAL_y).mean()
drop = (float_acc - q_acc) * 100
size_f, size_q = len(tflite_f32) / 1024, len(tflite_int8) / 1024
print(f"float32 model: {size_f:7.1f} KB   accuracy {float_acc*100:.2f} %")
print(f"int8    model: {size_q:7.1f} KB   accuracy {q_acc*100:.2f} %   ({time.time()-t0:.0f} s to evaluate)")
print(f"Size ≤ {TARGETS['max_kb']:.0f} KB: {'✅' if size_q <= TARGETS['max_kb'] else '❌'}   |   "
      f"accuracy drop {drop:.2f} points ≤ {TARGETS['max_drop_points']}: {'✅' if drop <= TARGETS['max_drop_points'] else '❌'}")
print(f"Parameters: {model.count_params():,}")

plt.figure(figsize=(6, 3))
b = plt.bar(["float32", "int8"], [size_f, size_q], color=["#888", "#2a7"])
plt.bar_label(b, [f"{size_f:.0f} KB\n{float_acc*100:.1f}%", f"{size_q:.0f} KB\n{q_acc*100:.1f}%"])
plt.axhline(TARGETS['max_kb'], ls="--", c="r")
plt.text(1.45, TARGETS['max_kb'], f"{TARGETS['max_kb']:.0f} KB limit", va="bottom", ha="right", color="r")
plt.ylabel("KB"); plt.title("Quantisation: size and accuracy"); plt.ylim(0, max(TARGETS['max_kb'] * 1.1, size_f * 1.3))
plt.tight_layout(); plt.savefig("out/quantisation.png", dpi=120); plt.show()

# %% [markdown]
# ## 7. Hard path – export for the microcontroller
#
# Writes `model_data.h` (the int8 model as a C array) and `kws_config.h` (labels, normalisation stats,
# input/output quantisation). Copy both into the `kws_live` sketch folder next to `kws_live.ino`.
#
# ⚠️ **The board computes MFCCs with the code from Section 1.** If you trained on the kit's own MFCCs
# (`FEATURE_SOURCE == "kit"`), the live demo only works when the kit used the same settings
# (the values printed in Section 1). If you're unsure, retrain with `FEATURE_SOURCE = "wav"`
# (you need the kit's .wav files for that), or use the kit's sample-project feature extractor instead.

# %%
it = tf.lite.Interpreter(model_content=tflite_int8); it.allocate_tensors()
in_s, in_z = it.get_input_details()[0]["quantization"]
out_s, out_z = it.get_output_details()[0]["quantization"]
ops = sorted({d["op_name"] for d in it._get_ops_details()} - {"DELEGATE"})
print("TFLite ops used (all must be registered in the sketch):", ops)
if src == "kit": print("⚠️  Trained on KIT MFCCs – check the kit's MFCC settings match Section 1 before the live demo.")

# Self-check signal: a 1 s chirp sweeping across the mel range. The board makes the same chirp at boot, runs its
# own MFCC code on it and compares against these values, which shows whether the C++ features match Python's.
def chirp():
    t = np.arange(CLIP, dtype=np.float64) / SR
    f0, f1, T = F_LO * 5, F_HI * 0.95, CLIP / SR
    return (0.3 * np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * T) * t ** 2))).astype(np.float32)
GOLD_FRAMES = [0, N_FRAMES // 2, N_FRAMES - 1]
golden = mfcc(chirp())
GOLD = golden[GOLD_FRAMES]

def fmt(arr, per=12, f="{:.6f}f"):
    return ",\n  ".join(", ".join(f.format(v) for v in arr[i:i + per]) for i in range(0, len(arr), per))

with open("out/model_data.h", "w") as fh:
    fh.write("// Auto-generated by kws_notebook – int8 DS-CNN keyword spotter\n#pragma once\n#include <stdint.h>\n")
    fh.write(f"alignas(16) const unsigned char g_kws_model[] = {{\n  ")
    fh.write(",\n  ".join(", ".join(f"0x{b:02x}" for b in tflite_int8[i:i + 16]) for i in range(0, len(tflite_int8), 16)))
    fh.write(f"\n}};\nconst unsigned int g_kws_model_len = {len(tflite_int8)};\n")

with open("out/kws_config.h", "w") as fh:
    fh.write("// Auto-generated by kws_notebook – must match the trained model\n#pragma once\n")
    fh.write(f"#define KWS_NUM_CLASSES {NUM_CLASSES}\n#define KWS_N_FRAMES {N_FRAMES}\n#define KWS_N_MFCC {N_MFCC}\n")
    fh.write(f"#define KWS_CLIP {CLIP}\n#define KWS_SR {SR}\n#define KWS_FRAME_LEN {FRAME_LEN}\n#define KWS_HOP {HOP}\n#define KWS_NFFT {NFFT}\n")
    fh.write(f"#define KWS_N_MEL {N_MEL}\n#define KWS_F_LO {F_LO}f\n#define KWS_F_HI {F_HI}f\n")
    fh.write(f"const float KWS_IN_SCALE = {in_s:.9g}f;\nconst int KWS_IN_ZP = {in_z};\n")
    fh.write(f"const float KWS_OUT_SCALE = {out_s:.9g}f;\nconst int KWS_OUT_ZP = {out_z};\n")
    fh.write(f"const float KWS_MEAN[KWS_N_MFCC] = {{ {fmt(MEAN)} }};\n")
    fh.write(f"const float KWS_STD[KWS_N_MFCC]  = {{ {fmt(STD)} }};\n")
    fh.write("const char* const KWS_LABELS[KWS_NUM_CLASSES] = { " + ", ".join(f'"{c}"' for c in CLASSES) + " };\n")
    fh.write(f"// TFLite ops: {', '.join(ops)}\n")
    fh.write(f"#define KWS_HAS_GOLDEN 1\n#define KWS_CHIRP_F0 {F_LO*5}f\n#define KWS_CHIRP_F1 {F_HI*0.95}f\n")
    fh.write("const int KWS_GOLD_FRAMES[3] = { " + ", ".join(map(str, GOLD_FRAMES)) + " };\n")
    fh.write(f"const float KWS_GOLD[3][KWS_N_MFCC] = {{ " + ", ".join("{ " + fmt(g) + " }" for g in GOLD) + " };\n")
print("Wrote out/model_data.h and out/kws_config.h")
print("Self-check MFCC (frame 0):", np.round(GOLD[0], 3))

# %% [markdown]
# ## 8. Live-proof fallback (if the board misbehaves): run a prediction on a laptop recording
# Upload any .wav (any sample rate – it's resampled; or record one on your phone) and set the path below.

# %%
def predict_wav(path):
    from scipy.io import wavfile
    sr, a = wavfile.read(path)
    a = a.astype(np.float32) / (32768.0 if a.dtype == np.int16 else 1.0)
    if a.ndim > 1: a = a.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        a = resample_poly(a, SR, sr).astype(np.float32)
    # take the loudest 1 s window
    if len(a) > CLIP:
        e = np.convolve(a ** 2, np.ones(CLIP), "valid"); st = int(e.argmax()); a = a[st:st + CLIP]
    p = model.predict(norm(mfcc(a)[None, :, :, None]), verbose=0)[0]
    top = p.argsort()[::-1][:3]
    print(" | ".join(f"{CLASSES[i]} {p[i]*100:.0f}%" for i in top))

# predict_wav("/content/my_yes.wav")

# %% [markdown]
# ## 9. Summary for the slides

# %%
summary = {
    "feature_source": src, "classes": NUM_CLASSES, "params": int(model.count_params()),
    "baseline_val_acc": round(float(base_val_acc) * 100, 2),
    "eval_set": EVAL_NAME.split()[0], "float_acc": round(float(float_acc) * 100, 2),
    "int8_acc": round(float(q_acc) * 100, 2), "acc_drop_points": round(float(drop), 2),
    "float_kb": round(size_f, 1), "int8_kb": round(size_q, 1),
    "top_confusions": [f"{a}->{b} ({c})" for c, a, b in pairs[:3]],
    "targets": TARGETS, "mfcc_settings": FEAT,
}
json.dump(summary, open("out/summary.json", "w"), indent=2)
for k_, v_ in summary.items(): print(f"{k_:18s} {v_}")

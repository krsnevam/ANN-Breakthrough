"""
Dataset preparation script for Theme 3: Predictive Maintenance
Downloads CWRU Bearing data (12 kHz drive-end) for 0 hp and 3 hp,
slices into 2,048-sample windows, and prepares C-MAPSS FD001 and FD004 with RUL labels.
All data saved locally inside /Applications/annhack/ANN-Breakthrough/data/
"""

import os
import urllib.request
import numpy as np
import pandas as pd
from scipy.io import loadmat

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CWRU_DIR = os.path.join(DATA_DIR, "cwru")
CMAPSS_DIR = os.path.join(DATA_DIR, "cmapss")

os.makedirs(CWRU_DIR, exist_ok=True)
os.makedirs(CMAPSS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. CWRU Bearing Data Preparation
# -------------------------------------------------------------
CWRU_CLASSES = [
    "Normal",
    "Ball_007", "InnerRace_007", "OuterRace_007",
    "Ball_014", "InnerRace_014", "OuterRace_014",
    "Ball_021", "InnerRace_021", "OuterRace_021"
]

# File IDs mapped to (0 hp, 3 hp)
CWRU_FILE_MAP = {
    "Normal":         (97,  100),
    "Ball_007":       (118, 121),
    "InnerRace_007":  (105, 108),
    "OuterRace_007":  (130, 133),
    "Ball_014":       (185, 188),
    "InnerRace_014":  (169, 172),
    "OuterRace_014":  (197, 200),
    "Ball_021":       (222, 225),
    "InnerRace_021":  (209, 212),
    "OuterRace_021":  (234, 237)
}

CWRU_BASE_URL = "https://engineering.case.edu/sites/default/files/"

def download_file(url, target_path):
    if not os.path.exists(target_path):
        print(f"Downloading {url} -> {target_path}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(target_path, 'wb') as out_f:
            out_f.write(resp.read())
    else:
        print(f"Cached: {target_path}")

def extract_de_signal(mat_path):
    mat_data = loadmat(mat_path)
    # Search for Drive End vibration variable: key containing 'DE_time'
    for k in mat_data.keys():
        if "DE_time" in k:
            return mat_data[k].squeeze()
    raise KeyError(f"No DE_time key found in {mat_path}. Keys: {list(mat_data.keys())}")

def process_cwru(window_size=2048, step=1024, max_windows_per_class=120):
    for load_idx, load_hp in enumerate(["0hp", "3hp"]):
        X_list = []
        y_list = []
        print(f"\n--- Processing CWRU {load_hp} (12 kHz Drive End) ---")
        for class_idx, class_name in enumerate(CWRU_CLASSES):
            mat_id = CWRU_FILE_MAP[class_name][load_idx]
            mat_filename = f"{mat_id}.mat"
            mat_path = os.path.join(CWRU_DIR, mat_filename)
            download_file(f"{CWRU_BASE_URL}{mat_filename}", mat_path)

            signal = extract_de_signal(mat_path)
            # Slice into windows of window_size
            windows = []
            for start in range(0, len(signal) - window_size + 1, step):
                windows.append(signal[start : start + window_size])
                if len(windows) >= max_windows_per_class:
                    break

            windows = np.array(windows, dtype=np.float32)
            labels = np.full(len(windows), class_idx, dtype=np.int64)
            X_list.append(windows)
            y_list.append(labels)
            print(f"Class {class_idx} ({class_name}): {len(windows)} windows of length {window_size}")

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        out_npz = os.path.join(CWRU_DIR, f"cwru_{load_hp}_12k_w2048.npz")
        np.savez_compressed(out_npz, X=X, y=y, classes=CWRU_CLASSES)
        print(f"Saved {load_hp} dataset to {out_npz}: shape X={X.shape}, y={y.shape}")

# -------------------------------------------------------------
# 2. NASA C-MAPSS Preparation
# -------------------------------------------------------------
CMAPSS_BASE_URL = "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/"
CMAPSS_COLS = ["unit_id", "time_cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"s_{i}" for i in range(1, 22)]

def process_cmapss():
    for sub in ["FD001", "FD004"]:
        print(f"\n--- Processing C-MAPSS {sub} ---")
        train_raw = os.path.join(CMAPSS_DIR, f"train_{sub}.txt")
        test_raw = os.path.join(CMAPSS_DIR, f"test_{sub}.txt")
        rul_raw = os.path.join(CMAPSS_DIR, f"RUL_{sub}.txt")

        download_file(f"{CMAPSS_BASE_URL}train_{sub}.txt", train_raw)
        download_file(f"{CMAPSS_BASE_URL}test_{sub}.txt", test_raw)
        download_file(f"{CMAPSS_BASE_URL}RUL_{sub}.txt", rul_raw)

        # 1. Process Train Data: calculate RUL = max_cycle - current_cycle
        df_train = pd.read_csv(train_raw, sep=r"\s+", header=None, names=CMAPSS_COLS)
        max_cycle = df_train.groupby("unit_id")["time_cycle"].transform("max")
        df_train["RUL"] = max_cycle - df_train["time_cycle"]
        train_csv = os.path.join(CMAPSS_DIR, f"train_{sub}.csv")
        df_train.to_csv(train_csv, index=False)
        print(f"Saved {train_csv}: {len(df_train)} rows across {df_train['unit_id'].nunique()} units.")

        # 2. Process Test Data & RUL:
        df_test = pd.read_csv(test_raw, sep=r"\s+", header=None, names=CMAPSS_COLS)
        rul_df = pd.read_csv(rul_raw, sep=r"\s+", header=None, names=["true_RUL"])
        rul_df["unit_id"] = np.arange(1, len(rul_df) + 1)

        test_max_cycle = df_test.groupby("unit_id")["time_cycle"].transform("max")
        df_test = df_test.merge(rul_df, on="unit_id", how="left")
        # RUL at any test cycle = true_RUL + (max_cycle - current_cycle)
        df_test["RUL"] = df_test["true_RUL"] + (test_max_cycle - df_test["time_cycle"])
        test_csv = os.path.join(CMAPSS_DIR, f"test_{sub}.csv")
        df_test.to_csv(test_csv, index=False)
        print(f"Saved {test_csv}: {len(df_test)} rows across {df_test['unit_id'].nunique()} test units.")

if __name__ == "__main__":
    process_cwru()
    process_cmapss()
    print("\nAll datasets downloaded, processed, and cached locally successfully!")

import json

with open('ANN_Hackathon_Predictive_Maintenance.ipynb') as f:
    nb = json.load(f)

# --- Change 1: Update preprocessing cell (cell 9) ---
cell9_code = '''# CWRU Preprocessing
def normalize_vibration(X):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

def add_sensor_noise(X, snr_db=20, seed=42):
    # Add realistic Gaussian sensor noise at a given SNR (Signal-to-Noise Ratio in dB).
    # Real accelerometers always introduce some measurement noise.
    # SNR=20 dB is typical for a good industrial vibration sensor.
    rng = np.random.RandomState(seed)
    noisy = np.empty_like(X)
    for i in range(len(X)):
        sig_power = np.mean(X[i] ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = rng.normal(0, np.sqrt(noise_power), size=X[i].shape)
        noisy[i] = X[i] + noise
    return noisy

X0_norm = normalize_vibration(X0)
X3_norm = normalize_vibration(X3)

# C-MAPSS Preprocessing
active_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
cmapss_features = ['op_setting_1', 'op_setting_2'] + active_sensors

train_df = cmapss_train.copy()
test_df = cmapss_test.copy()

train_df['RUL_clipped'] = train_df['RUL'].clip(upper=CONFIG['cmapss_rul_max'])
test_df['RUL_clipped']  = test_df['RUL'].clip(upper=CONFIG['cmapss_rul_max'])

cmapss_scaler = StandardScaler()
train_df[cmapss_features] = cmapss_scaler.fit_transform(train_df[cmapss_features])
test_df[cmapss_features]  = cmapss_scaler.transform(test_df[cmapss_features])

W = CONFIG['cmapss_window_len']
def create_cmapss_windows(df, is_test=False):
    X_seq, y_target = [], []
    for unit_id, group in df.groupby('unit_id'):
        vals = group[cmapss_features].values
        ruls = group['RUL_clipped'].values
        if len(vals) < W:
            pad = np.repeat(vals[:1], W - len(vals), axis=0)
            vals = np.vstack([pad, vals])
            ruls = np.repeat(ruls[-1], len(vals))
        if is_test:
            X_seq.append(vals[-W:])
            y_target.append(ruls[-1])
        else:
            for i in range(len(vals) - W + 1):
                X_seq.append(vals[i : i + W])
                y_target.append(ruls[i + W - 1])
    return np.array(X_seq, dtype=np.float32), np.array(y_target, dtype=np.float32)

X_cmapss_tr, y_cmapss_tr = create_cmapss_windows(train_df, is_test=False)
X_cmapss_te, y_cmapss_te = create_cmapss_windows(test_df, is_test=True)

print(f"C-MAPSS Sequences | Train: {X_cmapss_tr.shape} | Test: {X_cmapss_te.shape}")'''

nb['cells'][9]['source'] = [line + "\n" for line in cell9_code.split("\n")]

# --- Change 2: Update split cell (cell 11) to add noise ---
cell11_code = '''# CWRU 0hp split
X_tr, X_temp, y_tr, y_temp = train_test_split(X0_norm, y0, test_size=0.30, random_state=CONFIG['seed'], stratify=y0)
X_val, X_te, y_val, y_te   = train_test_split(X_temp, y_temp, test_size=0.50, random_state=CONFIG['seed'], stratify=y_temp)

# Add realistic sensor noise to validation and test sets
# This simulates real-world conditions where sensors always have some measurement noise
X_val = normalize_vibration(add_sensor_noise(X_val, snr_db=18, seed=100))
X_te  = normalize_vibration(add_sensor_noise(X_te,  snr_db=15, seed=200))

print(f"CWRU 0hp Splits -> Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_te)}")
print(f"Test set has realistic sensor noise added (SNR=15 dB) to simulate field conditions")'''

nb['cells'][11]['source'] = [line + "\n" for line in cell11_code.split("\n")]

# --- Change 3: Update scorecard markdown (cell 24) ---
cell24_md = '''## Step 12 — Summary Scorecard & Presentation Checklist (3 minutes)

### Final Results Scorecard

| Milestone | Task | Dataset & Condition | Target | Status |
|---|---|---|---|---|
| **Foundation** | 10-Class Bearing Fault Diagnosis | CWRU (0 hp, 12 kHz) with sensor noise | Accuracy >= 95% | See Step 9 |
| **Advanced** | Remaining Useful Life (RUL) | NASA C-MAPSS (FD001) | RMSE <= 25 cycles | See Step 9 |
| **Hard Path** | Cross-Condition Domain Generalization | Train 0 hp -> Test 3 hp | Report drop & fix | See Step 11 |

---

### Presentation Pitch Checklist:
1. **The Core Insight**: Machine fault frequencies are tied to operational speed. Load changes cause rotor slip, causing raw time-domain classifiers to misalign unless wide temporal kernels and domain adaptation are applied.
2. **Why 1D-CNN over standard MLP/RF**: The wide first kernel (k=64, s=8) acts as a data-driven filter bank that captures impulse shocks directly from raw 12 kHz sensor signals without requiring manual feature engineering.
3. **Turbofan Lifecycle Framing**: C-MAPSS engine degradation follows a piecewise-linear regime. Capping RUL at 125 cycles allows temporal convolutions to focus on the degradation phase.'''

nb['cells'][24]['source'] = [line + "\n" for line in cell24_md.split("\n")]

# --- Change 4: Append a live demo markdown + code cell ---
demo_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [line + "\n" for line in """## Step 13 — Live Demo: Test the Model on Any Sample
Pick any random vibration sample (even with added noise) and see the model diagnose it in real time.
You can change the `demo_index` to test different samples, or adjust `noise_snr_db` to simulate noisier/cleaner sensors.""".split("\n")]
}

demo_code_str = '''# ===== LIVE DEMO: Change these values to test different inputs =====
demo_index = np.random.randint(0, len(X0))   # random sample from the full 0hp dataset
add_noise_demo = True                         # True = simulate real sensor noise
noise_snr_db = 15                             # lower = noisier (10=very noisy, 20=clean, 15=realistic)

# --- Prepare the sample ---
raw_sample = X0[demo_index:demo_index+1].copy()
true_label = y0[demo_index]

if add_noise_demo:
    raw_sample = add_sensor_noise(raw_sample, snr_db=noise_snr_db, seed=None)

sample_norm = normalize_vibration(raw_sample)
sample_tensor = torch.tensor(sample_norm[:, None, :], dtype=torch.float32)

# --- Run through the trained model ---
cwru_model.eval()
with torch.no_grad():
    logits = cwru_model(sample_tensor)
    probs = torch.softmax(logits, dim=-1).numpy()[0]
    predicted_class = int(logits.argmax(dim=-1).item())

# --- Display results ---
fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# Left: the vibration waveform
axes[0].plot(raw_sample[0][:500], lw=1, color='steelblue')
axes[0].set_title(f"Input Vibration Waveform (sample #{demo_index})", fontweight='bold')
axes[0].set_xlabel("Time (sample points)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, alpha=0.3)

# Right: confidence bar chart
colors = ['green' if i == true_label else 'salmon' if i == predicted_class else 'lightgray' for i in range(10)]
bars = axes[1].barh(range(10), probs * 100, color=colors)
axes[1].set_yticks(range(10))
axes[1].set_yticklabels(classes, fontsize=9)
axes[1].set_xlabel("Confidence (%)")
axes[1].set_title("Model Prediction Confidence", fontweight='bold')
axes[1].set_xlim(0, 105)
for i, v in enumerate(probs):
    if v > 0.05:
        axes[1].text(v * 100 + 1, i, f"{v*100:.1f}%", va='center', fontsize=9)

result_text = "CORRECT" if predicted_class == true_label else "WRONG"
result_color = "green" if predicted_class == true_label else "red"
plt.suptitle(
    f"TRUE: {classes[true_label]}  |  PREDICTED: {classes[predicted_class]}  |  {result_text}",
    fontsize=14, fontweight='bold', color=result_color, y=1.02
)
plt.tight_layout()
plt.show()

print(f"\\nTrue Label:      {classes[true_label]}")
print(f"Predicted Label: {classes[predicted_class]}")
print(f"Confidence:      {probs[predicted_class]*100:.1f}%")
if add_noise_demo:
    print(f"Noise Added:     Yes (SNR={noise_snr_db} dB)")
else:
    print(f"Noise Added:     No (clean lab data)")'''

demo_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in demo_code_str.split("\n")]
}

nb['cells'].append(demo_md)
nb['cells'].append(demo_code)

with open('ANN_Hackathon_Predictive_Maintenance.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Notebook updated successfully!")

# %%
from pathlib import Path

import os
import argparse
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# %%
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Getting params from command line
def parse_args():
    parser = argparse.ArgumentParser(description="LSTM autoencoder training for anomalies")
    parser.add_argument("output_dir", type=str, help="Path to output directory for model and results")
    parser.add_argument("seq_length", type=int, help="Sequence length")
    parser.add_argument("stride", type=int, help="Stride of windows")
    parser.add_argument("units", type=int, help="Main LSTM Units")
    parser.add_argument("middle_units", type=int, help="Middle LSTM Units")
    parser.add_argument("batch", type=int, help="Batch size")
    parser.add_argument("epoch", type=int, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda_l2", type=float, default=1e-5, help="L2 regularization")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout LSTM")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for K-fold")
    return parser.parse_args()


args = parse_args()
OUTPUT_DIR = Path(args.output_dir)

# %% [markdown]
# ## Load JSON and prepare dataframe

# %%
with open('data/rpi1_all.json') as f1, open('data/rpi2_all.json') as f2, open('data/rpi3_all.json') as f3, open('data/rpi4_all.json') as f4:
  pcks1 = json.load(f1)
  pcks2 = json.load(f2)
  pcks3 = json.load(f3)
  pcks4 = json.load(f4)


print(len(pcks1))
print(len(pcks2))
print(len(pcks3))
print(len(pcks4))

# %%
columns = [
    "packet_idx", "src16", "dst16", "length", "time_rel_us",
    "rssi", "epoch_us", "attack", "dev", "rpi"
]

frames = [pd.DataFrame(p, columns=columns) for p in (pcks1, pcks2, pcks3, pcks4) if len(p) > 0]
dataframe = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)

numeric_cols = ["packet_idx", "src16", "dst16", "length", "time_rel_us", "rssi", "epoch_us", "rpi"]
dataframe[numeric_cols] = dataframe[numeric_cols].apply(pd.to_numeric, errors="coerce")

dataframe = dataframe.sort_values(by="epoch_us", ignore_index=True)

# %% [markdown]
# ## Preprocessing

# %%
df = dataframe
df = df.drop(['packet_idx', 'time_rel_us'], axis=1)

src_vals = [8957, 43231, 31646, 24250, 3078, 31863, 41643, 2578, 18154, 35010]

df = df[df['src16'].isin(src_vals)].reset_index(drop=True)
df = df[df['dst16'].isin(src_vals)].reset_index(drop=True)

df = df.drop(['dst16', 'dev'], axis=1)
df.head()

# %%
mapped = {
    'NO_ATTACK': 0,
    'A': 1,
    'B': 1,
    'C': 1,
    'D': 1,
    'E': 1,
}

df['attack'] = df['attack'].map(mapped)
df.head()

# %%
df = df.sort_values(["epoch_us"], ignore_index=True)
df["delta_t"] = df.groupby(["src16", "rpi"])["epoch_us"].diff().fillna(0).astype(int)
df["delta_l"] = df.groupby(["src16", "rpi"])["length"].diff().fillna(0).astype(int)

df.head()


# %%
df['date'] = pd.to_datetime(df['epoch_us'], unit='us', utc=True)
df['date'] = df['date'].dt.tz_convert('Europe/Paris')

print("First packet is at: \t", df['date'].min())
print("Last packet is at: \t", df['date'].max())

# %% [markdown]
# Divide by time and scale with fit only in train

# %%
start_attack = pd.Timestamp('2022-07-03 07:00:00', tz='Europe/Paris')
end_attack = pd.Timestamp('2022-07-09 00:00:00', tz='Europe/Paris')

# Normal period before attack
df_pre_attack = df[df['date'] < start_attack].copy()

# Period of attack
df_attack = df[df['date'].between(start_attack, end_attack)].copy()

# Temporary split of the normal pre-attack period
split_normal = df_pre_attack['date'].quantile(0.8)
df_train = df_pre_attack[df_pre_attack['date'] <= split_normal].copy()
df_val_normal = df_pre_attack[df_pre_attack['date'] > split_normal].copy()

# Temporary split of the attack period
split_attack = df_attack['date'].quantile(0.5)
df_val_mixed = df_attack[df_attack['date'] <= split_attack].copy()
df_test = df_attack[df_attack['date'] > split_attack].copy()

df_train = df_train.drop(['date', 'epoch_us'], axis=1)
df_val_normal = df_val_normal.drop(['date', 'epoch_us'], axis=1)
df_val_mixed = df_val_mixed.drop(['date', 'epoch_us'], axis=1)
df_test = df_test.drop(['date', 'epoch_us'], axis=1)

print('Rows train normal:', len(df_train))
print('Rows val normal:', len(df_val_normal))
print('Rows val mixed:', len(df_val_mixed))
print('Rows test total:', len(df_test))
print('\nTrain class balance:')
print(df_train['attack'].value_counts())
print('\nValidation class balance:')
print(df_val_normal['attack'].value_counts())
print('\nTest class balance:')
print(df_test['attack'].value_counts())


# %% [markdown]
# Scaling of variables

# %%
from sklearn.preprocessing import RobustScaler, StandardScaler


scRSSI = StandardScaler()
df_train["rssi"] = scRSSI.fit_transform(df_train[["rssi"]])
df_val_normal["rssi"] = scRSSI.transform(df_val_normal[["rssi"]])
df_val_mixed["rssi"] = scRSSI.transform(df_val_mixed[["rssi"]])
df_test["rssi"] = scRSSI.transform(df_test[["rssi"]])

cols_scale = ["length", "delta_t", "delta_l"]
scaler = RobustScaler()

df_train[cols_scale] = scaler.fit_transform(df_train[cols_scale])
df_val_normal[cols_scale] = scaler.transform(df_val_normal[cols_scale])
df_val_mixed[cols_scale] = scaler.transform(df_val_mixed[cols_scale])
df_test[cols_scale] = scaler.transform(df_test[cols_scale])

print('Shapes tras split+scale:')
print('train:', df_train.shape)
print('val normal:', df_val_normal.shape)
print('val mixed:', df_val_mixed.shape)
print('test:', df_test.shape)

# %%
print(df_train.head())

# %% [markdown]
# ## LSTM

# %%
from time import time

from keras import Input
from keras.models import Sequential
from keras.layers import Dense, LSTM, LayerNormalization,  RepeatVector, TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TerminateOnNaN, ModelCheckpoint
from keras.regularizers import l2

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    classification_report
)

# %% [markdown]
# ### Build windows

# %% [markdown]
# Group by (src16,rpi)

# %%
X_train_groups = df_train.drop(columns=["attack"]).groupby(["src16", "rpi"])
X_val_normal_groups = df_val_normal.drop(columns=["attack"]).groupby(["src16", "rpi"])
X_val_mixed_groups = df_val_mixed.groupby(["src16", "rpi"])
X_test_groups = df_test.groupby(["src16", "rpi"])

# %%
seq_length = args.seq_length
stride = args.stride
anomaly_fraction_threshold = 0.2

feature_cols = ["length", "rssi", "delta_t", "delta_l"]

n_features = len(feature_cols)
print('Feature columns:', feature_cols)

def build_windows(array, seq_len, step):
    n_rows = len(array)
    if n_rows < seq_len:
        return np.empty((0, seq_len, array.shape[1]), dtype=array.dtype), None

    starts = np.arange(0, n_rows - seq_len + 1, step)
    idx = starts[:, None] + np.arange(seq_len)
    return array[idx], idx

def build_windows_from_groups(groups, seq_len, step, with_labels=False):
    windows_list = []
    labels_list = []
    short_groups = 0
    total_groups = 0

    for (src16, rpi), group in groups:
        total_groups += 1

        features = group[feature_cols].to_numpy(dtype=np.float32, copy=False)
        windows, idx = build_windows(features, seq_len, step)

        if len(windows) == 0:
            short_groups += 1
            continue

        windows_list.append(windows)

        if with_labels:
            labels = group["attack"].to_numpy(copy=False)
            
            # A window is labeled as anomalous if it contains a relevant fraction of anomaly
            y_group =  (labels[idx].mean(axis=1) >= anomaly_fraction_threshold).astype(np.int8)
            labels_list.append(y_group)

    X = np.concatenate(windows_list, axis=0) if windows_list else np.empty((0, seq_len, n_features), dtype=np.float32)

    if with_labels:
        y = np.concatenate(labels_list, axis=0) if labels_list else np.empty((0,), dtype=np.int8)
        return X, y, short_groups, total_groups

    return X, short_groups, total_groups

# %%
# short: groups shorter than seq_length that are lost
# total: total number of generated windows
X_train, short_train, total_train = build_windows_from_groups(
    X_train_groups, seq_length, stride, with_labels=False)

X_val_normal, short_val_normal, total_val_normal = build_windows_from_groups(
    X_val_normal_groups, seq_length, stride, with_labels=False)

X_val_mixed, y_true_seq_val, short_val_mixed, total_val_mixed = build_windows_from_groups(
    X_val_mixed_groups, seq_length, stride, with_labels=True)

X_test, y_true_seq_test, short_test, total_test = build_windows_from_groups(
    X_test_groups, seq_length, stride, with_labels=True)

print('Shapes windows:', X_train.shape, X_val_normal.shape, X_test.shape)
print(f'Groups train short (<{seq_length}): {short_train}/{total_train}')
print(f'Groups val normal short (<{seq_length}): {short_val_normal}/{total_val_normal}')
print(f'Groups val mixed short (<{seq_length}): {short_val_mixed}/{total_val_mixed}')
print(f'Groups test short (<{seq_length}): {short_test}/{total_test}')
print('Positives in val (windows):', int(y_true_seq_val.sum()), 'of', len(y_true_seq_val))
print('Positives in test (windows):', int(y_true_seq_test.sum()), 'of', len(y_true_seq_test))

# %% [markdown]
# ### Modelo

# %%
# Hiperparametros base
UNITS = args.units
MIDDLE_UNITS = args.middle_units
M_TRAIN = X_train.shape[0]
M_TEST = X_test.shape[0]
T = X_train.shape[1]
N = X_train.shape[2]

BATCH = args.batch
EPOCH = args.epoch
LR = args.learning_rate
LAMBD = args.lambda_l2
DP = args.dropout
RDP = 0.0

print(f'seq_length={seq_length}, stride={stride}')
print(f'units={UNITS}/{MIDDLE_UNITS}, train_examples={M_TRAIN}, test_examples={M_TEST}')
print(f'batch={BATCH}, timesteps={T}, features={N}, epochs={EPOCH}')
print(f'lr={LR}, lambda={LAMBD}, dropout={DP}, recurr_dropout={RDP}')

# %%
# Build the model

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # 4 capas LSTM: 2 encoder + 2 decoder
    model.add(
        LSTM(
            units=UNITS,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_regularizer=l2(LAMBD),
            recurrent_regularizer=l2(LAMBD),
            dropout=DP,
            recurrent_dropout=RDP,
            return_sequences=True,
            stateful=False,
            unroll=False,
        )
    )
    model.add(LayerNormalization())

    model.add(
        LSTM(
            units=MIDDLE_UNITS,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_regularizer=l2(LAMBD),
            recurrent_regularizer=l2(LAMBD),
            dropout=DP,
            recurrent_dropout=RDP,
            return_sequences=False,
            stateful=False,
            unroll=False,
        )
    )

    model.add(RepeatVector(input_shape[0]))

    model.add(
        LSTM(
            units=MIDDLE_UNITS,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_regularizer=l2(LAMBD),
            recurrent_regularizer=l2(LAMBD),
            dropout=DP,
            recurrent_dropout=RDP,
            return_sequences=True,
            stateful=False,
            unroll=False,
        )
    )
    model.add(LayerNormalization())

    model.add(
        LSTM(
            units=UNITS,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_regularizer=l2(LAMBD),
            recurrent_regularizer=l2(LAMBD),
            dropout=DP,
            recurrent_dropout=RDP,
            return_sequences=True,
            stateful=False,
            unroll=False,
        )
    )

    model.add(TimeDistributed(Dense(input_shape[1])))

    model.compile(loss="mse", optimizer=Adam(learning_rate=LR, clipnorm=1.0))
    return model, "MSE"

def make_callbacks(results_dir, best_name):
    lr_decay = ReduceLROnPlateau(monitor="val_loss", patience=3, verbose=1, factor=0.5, min_lr=1e-6)
    early_stop = EarlyStopping(monitor="val_loss", min_delta=5e-4, patience=6, verbose=1, mode="min", restore_best_weights=True)
    terminate_on_nan = TerminateOnNaN()
    checkpoint = ModelCheckpoint(
        filepath=str(results_dir / best_name),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    return [lr_decay, early_stop, terminate_on_nan, checkpoint]

results_dir = OUTPUT_DIR
results_dir.mkdir(parents=True, exist_ok=True)
model_path = results_dir / 'lstm_autoencoder.keras'
model_best_path = results_dir / 'lstm_autoencoder_best.keras'

checkpoint = ModelCheckpoint(
    filepath=str(model_best_path),
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
 )

model, loss_name = build_model((seq_length, n_features))

start = time()

History = model.fit(
    X_train, X_train,
    epochs=EPOCH,
    batch_size=BATCH,
    validation_data=(X_val_normal, X_val_normal),
    shuffle=True,
    verbose=1,
    callbacks=make_callbacks(results_dir, 'lstm_autoencoder.keras')
)

if model_best_path.exists():
    print(f'Loading best model from checkpoint: {model_best_path}')
    model = tf.keras.models.load_model(model_best_path)

# Verify that the model does not contain NaN or Inf weights before saving
bad_weights = 0
for w in model.weights:
    bad_weights += int((~np.isfinite(w.numpy())).sum())

if bad_weights > 0:
    raise ValueError(
        f'Invalid trained model: {bad_weights} weights NaN/Inf. '
        'Final model not saved. Lower LR or check numerical stability.'
    )

model.save(model_path)
print(f'Model saved to: {model_path}')

# Graph training and validation losses
history = History.history
train_loss = history.get('loss', [])
val_loss = history.get('val_loss', [])

if train_loss:
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train loss')
    if val_loss:
        plt.plot(val_loss, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.title('Training Loss of LSTM Autoencoder')
    plt.grid(alpha=0.25)
    plt.legend()

    loss_plot_path = results_dir / 'train_loss.png'
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=180, bbox_inches='tight')
    print(f'Graph of loss saved to: {loss_plot_path}')
    plt.show()

print('-' * 65)
print(f'Training was completed in {time() - start:.2f} secs')
print('-' * 65)

# %%
# Load model and calculate reconstruction errors
model = tf.keras.models.load_model(model_path)

X_val_pred = model.predict(X_val_normal)
X_val_mixed_pred = model.predict(X_val_mixed)
X_test_pred = model.predict(X_test)

# Total error per window
val_error = np.mean((X_val_normal - X_val_pred) ** 2, axis=(1, 2))
val_mixed_error = np.mean((X_val_mixed - X_val_mixed_pred) ** 2, axis=(1, 2))
test_error = np.mean((X_test - X_test_pred) ** 2, axis=(1, 2))

print('X_val/X_val_mixed/X_test:', X_val_normal.shape, X_val_mixed.shape, X_test.shape)
print('val_error:', val_error.shape, 'val_mixed_error:', val_mixed_error.shape, 'test_error:', test_error.shape)
print('val_error mean/std:', float(np.nanmean(val_error)), float(np.nanstd(val_error)))
print('val_mixed_error mean/std:', float(np.nanmean(val_mixed_error)), float(np.nanstd(val_mixed_error)))
print('test_error mean/std:', float(np.nanmean(test_error)), float(np.nanstd(test_error)))

# %%
# Sweep of threshold by percentiles to choose precision/recall
results_dir = OUTPUT_DIR
results_dir.mkdir(parents=True, exist_ok=True)
metrics_path = results_dir / 'best_f1_threshold.txt'

percentiles = np.concatenate([np.arange(85, 99.1, 0.5), np.arange(99.2, 99.91, 0.1)])

rows = []
for p in percentiles:
    th = np.percentile(val_error, p)
    y_pred = (val_mixed_error > th).astype(int)

    acc = accuracy_score(y_true_seq_val, y_pred)
    bacc = balanced_accuracy_score(y_true_seq_val, y_pred)
    prec = precision_score(y_true_seq_val, y_pred, zero_division=0)
    rec = recall_score(y_true_seq_val, y_pred, zero_division=0)
    f1 = f1_score(y_true_seq_val, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true_seq_val, y_pred)

    rows.append({
        "percentile": float(p),
        "threshold": float(th),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "kappa": float(kappa)
    })

thr_df = pd.DataFrame(rows).sort_values("percentile").reset_index(drop=True)

best_f1 = thr_df.loc[thr_df["f1"].idxmax()]
best_recall = thr_df.loc[thr_df["recall"].idxmax()]

print("Best by F1:")
print(best_f1.to_string())
print("\nBest by Recall:")
print(best_recall.to_string())

plt.figure(figsize=(10, 5))
plt.plot(thr_df["percentile"], thr_df["f1"], label="F1")
plt.plot(thr_df["percentile"], thr_df["precision"], label="Precision")
plt.plot(thr_df["percentile"], thr_df["recall"], label="Recall")
plt.xlabel("Percentile of train_error used as threshold")
plt.ylabel("Metric")
plt.title("Sensitivity of metrics to the threshold")
plt.grid(alpha=0.25)
plt.legend()

plot_path = results_dir / 'evolucion_percentiles.png'
plt.tight_layout()
plt.savefig(plot_path, dpi=180, bbox_inches='tight')
print(f'Graph saved to: {plot_path}')
plt.show()

# %%
# Evaluation in val_normal
metrics_path = results_dir / 'best_f1_threshold_val_normal.txt'

th = best_f1["threshold"]
y_pred = (val_error > th).astype(int)

# Array of true labels for val_normal (all 0, since it's a normal dataset)
y_true = np.zeros_like(y_pred)
cm = confusion_matrix(y_true, y_pred)

# Calculate FPR for val_normal
fp = sum(y_pred == 1)
tn = sum(y_true == 0)
fpr = fp / tn if tn > 0 else 0

metrics_lines = [
    f"Best F1 (val_mixed): {best_f1['f1']}",
    f"threshold: {float(th)}",
    f"Falsos positivos (val_normal): {fp}",
    f"fpr (val_normal): {fpr:.6f}",
    "",
    "confusion_matrix_window:",
    np.array2string(cm),
]

print("\n".join(metrics_lines))

metrics_path.write_text('\n'.join(metrics_lines), encoding='utf-8')
print(f'Results saved to: {metrics_path}')

# %%
# Graph for viewing predictions vs reality
idx = np.arange(y_true.size)

fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(idx, y_true, color='gray', alpha=0.45, linewidth=1.5, label='Real (gray)')
ax.plot(idx, y_pred, color='blue', alpha=0.45, linewidth=1.5, label='Prediction (blue)')

real_anomaly_mask = y_true == 1
pred_anomaly_mask = y_pred == 1

ax.scatter(idx[real_anomaly_mask], y_true[real_anomaly_mask],
           color='red', alpha=0.45, marker='o', s=24, label='True anomaly', zorder=3,)

ax.scatter(idx[pred_anomaly_mask], y_pred[pred_anomaly_mask],
    color='yellow', alpha=0.45, marker='x', s=28, label='Predicted anomaly', zorder=4,)

ax.set_ylim(-0.1, 1.1)
ax.set_yticks([0, 1])
ax.set_ylabel('Class')
ax.set_xlabel('Sequence index')
ax.set_title('Actual vs. predicted sequence')
ax.grid(alpha=0.2)
ax.legend(loc='upper right')

plot_path = results_dir / 'real_sequence_vs_predicted_val_normal.png'
plt.tight_layout()
plt.savefig(plot_path, dpi=180, bbox_inches='tight')
print(f'Graph saved to: {plot_path}')
plt.show()
# %%
# Evaluation in test
metrics_path = results_dir / 'best_f1_threshold_test.txt'

th = best_f1["threshold"]
y_pred = (test_error > th).astype(int)

acc = accuracy_score(y_true_seq_test, y_pred)
bacc = balanced_accuracy_score(y_true_seq_test, y_pred)
prec = precision_score(y_true_seq_test, y_pred, zero_division=0)
rec = recall_score(y_true_seq_test, y_pred, zero_division=0)
f1 = f1_score(y_true_seq_test, y_pred, zero_division=0)
kappa = cohen_kappa_score(y_true_seq_test, y_pred)
pr_auc = average_precision_score(y_true_seq_test, test_error)
cm = confusion_matrix(y_true_seq_test, y_pred)


metrics_lines = [
    f"Best F1 (val_mixed): {best_f1['f1']}",
    f"threshold (test): {float(th)}",
    f"accuracy (test): {acc:.6f}",
    f"balanced_accuracy (test): {float(bacc)}",
    f"precision (test): {prec:.6f}",
    f"recall (test): {rec:.6f}",
    f"f1 (test): {f1:.6f}",
    f"kappa (test): {float(kappa)}",
    f"pr_auc (test): {pr_auc:.6f}",
    "",
    "confusion_matrix_window:",
    np.array2string(cm),
]

print("\n".join(metrics_lines))

metrics_path.write_text('\n'.join(metrics_lines), encoding='utf-8')
print(f'Results saved to: {metrics_path}')
# %%
# Graph for viewing predictions vs reality
idx = np.arange(y_true_seq_test.size)

fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(idx, y_true_seq_test, color='gray', alpha=0.45, linewidth=1.5, label='Real (gray)')
ax.plot(idx, y_pred, color='blue', alpha=0.45, linewidth=1.5, label='Prediction (blue)')

real_anomaly_mask = y_true_seq_test == 1
pred_anomaly_mask = y_pred == 1

ax.scatter(idx[real_anomaly_mask], y_true_seq_test[real_anomaly_mask],
           color='red', alpha=0.45, marker='o', s=24, label='True anomaly', zorder=3,)

ax.scatter(idx[pred_anomaly_mask], y_pred[pred_anomaly_mask],
    color='yellow', alpha=0.45, marker='x', s=28, label='Predicted anomaly', zorder=4,)

ax.set_ylim(-0.1, 1.1)
ax.set_yticks([0, 1])
ax.set_ylabel('Class')
ax.set_xlabel('Sequence index')
ax.set_title('Actual vs. predicted sequence')
ax.grid(alpha=0.2)
ax.legend(loc='upper right')

plot_path = results_dir / 'real_sequence_vs_predicted_test.png'
plt.tight_layout()
plt.savefig(plot_path, dpi=180, bbox_inches='tight')
print(f'Graph saved to: {plot_path}')
plt.show()

results_dir = OUTPUT_DIR
results_dir.mkdir(parents=True, exist_ok=True)
cv_dir = results_dir / "cv"
cv_dir.mkdir(parents=True, exist_ok=True)

if len(X_train) <= args.n_splits:
    raise ValueError("No hay suficientes ventanas de entrenamiento para K-fold.")

# %% [Markdown]
# ### Cross-validation K-fold

# %% 
from sklearn.model_selection import KFold

kf = KFold(n_splits=args.n_splits, shuffle=False)
cv_rows = []
cv_lines = [
    "K-fold Cross-Validation",
    "-" * 50,
]
cv_percentiles = np.concatenate([np.arange(85, 99.1, 0.5), np.arange(99.2, 99.91, 0.1)])

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    tf.keras.backend.clear_session()
    X_tr, X_va = X_train[train_idx], X_train[val_idx]

    model_cv, loss_name = build_model((seq_length, n_features))
    model_cv_path = cv_dir / f"fold_{fold}.keras"

    print("-" * 80)
    print(f"Fold {fold}/{args.n_splits}")
    print("Train:", X_tr.shape, "Val:", X_va.shape)

    start_fold = time()
    history = model_cv.fit(
        X_tr, X_tr,
        epochs=EPOCH,
        batch_size=BATCH,
        validation_data=(X_va, X_va),
        shuffle=True,
        verbose=1,
        callbacks=make_callbacks(cv_dir, f"fold_{fold}_best.keras"),
    )

    fold_best = float(np.min(history.history["val_loss"]))
    if (cv_dir / f"fold_{fold}_best.keras").exists():
        model_cv = tf.keras.models.load_model(cv_dir / f"fold_{fold}_best.keras")
        model_cv.save(model_cv_path)

    X_va_pred = model_cv.predict(X_va, verbose=0)
    X_val_mixed_pred = model_cv.predict(X_val_mixed, verbose=0)
    X_test_pred = model_cv.predict(X_test, verbose=0)

    val_error_fold = np.mean((X_va - X_va_pred) ** 2, axis=(1, 2))
    val_mixed_error_fold = np.mean((X_val_mixed - X_val_mixed_pred) ** 2, axis=(1, 2))
    test_error_fold = np.mean((X_test - X_test_pred) ** 2, axis=(1, 2))

    thr_rows_fold = []
    for p in cv_percentiles:
        th = np.percentile(val_error_fold, p)
        y_pred_fold = (val_mixed_error_fold > th).astype(int)
        thr_rows_fold.append(
            {
                "percentile": float(p),
                "threshold": float(th),
                "f1": float(f1_score(y_true_seq_val, y_pred_fold, zero_division=0)),
            }
        )

    thr_df_fold = pd.DataFrame(thr_rows_fold)
    best_fold = thr_df_fold.loc[thr_df_fold["f1"].idxmax()]
    fold_threshold = float(best_fold["threshold"])
    y_pred_fold = (val_mixed_error_fold > fold_threshold).astype(int)
    y_pred_test_fold = (test_error_fold > fold_threshold).astype(int)

    acc = accuracy_score(y_true_seq_val, y_pred_fold)
    bacc = balanced_accuracy_score(y_true_seq_val, y_pred_fold)
    prec = precision_score(y_true_seq_val, y_pred_fold, zero_division=0)
    rec = recall_score(y_true_seq_val, y_pred_fold, zero_division=0)
    f1 = f1_score(y_true_seq_val, y_pred_fold, zero_division=0)
    kappa = cohen_kappa_score(y_true_seq_val, y_pred_fold)
    cm = confusion_matrix(y_true_seq_val, y_pred_fold)
    report = classification_report(y_true_seq_val, y_pred_fold, zero_division=0)

    test_acc = accuracy_score(y_true_seq_test, y_pred_test_fold)
    test_bacc = balanced_accuracy_score(y_true_seq_test, y_pred_test_fold)
    test_prec = precision_score(y_true_seq_test, y_pred_test_fold, zero_division=0)
    test_rec = recall_score(y_true_seq_test, y_pred_test_fold, zero_division=0)
    test_f1 = f1_score(y_true_seq_test, y_pred_test_fold, zero_division=0)
    test_kappa = cohen_kappa_score(y_true_seq_test, y_pred_test_fold)
    test_pr_auc = average_precision_score(y_true_seq_test, test_error_fold)
    test_cm = confusion_matrix(y_true_seq_test, y_pred_test_fold)
    test_report = classification_report(y_true_seq_test, y_pred_test_fold, zero_division=0)

    cv_rows.append(
        {
            "fold": fold,
            "train_windows": int(len(train_idx)),
            "val_windows": int(len(val_idx)),
            "best_val_loss": fold_best,
            "threshold": fold_threshold,
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "kappa": float(kappa),
            "test_accuracy": float(test_acc),
            "test_balanced_accuracy": float(test_bacc),
            "test_precision": float(test_prec),
            "test_recall": float(test_rec),
            "test_f1": float(test_f1),
            "test_kappa": float(test_kappa),
            "test_pr_auc": float(test_pr_auc),
            "seconds": float(time() - start_fold),
        }
    )

    cv_lines.extend(
        [
            f"Fold {fold}",
            f"Threshold (val_error pctl -> best F1): {fold_threshold:.6f}",
            f"Accuracy (validation): {acc:.4f}",
            f"Balanced accuracy (validation): {bacc:.4f}",
            f"Precision (validation): {prec:.4f}",
            f"Recall (validation): {rec:.4f}",
            f"F1 score (validation): {f1:.4f}",
            f"Coeficiente kappa (validation): {kappa:.4f}",
            "Confusion matrix:",
            str(cm),
            "",
            report,
            "Test metrics:",
            f"Accuracy (test): {test_acc:.4f}",
            f"Balanced accuracy (test): {test_bacc:.4f}",
            f"Precision (test): {test_prec:.4f}",
            f"Recall (test): {test_rec:.4f}",
            f"F1 score (test): {test_f1:.4f}",
            f"Coeficiente kappa (test): {test_kappa:.4f}",
            f"PR AUC (test): {test_pr_auc:.4f}",
            "Confusion matrix (test):",
            str(test_cm),
            "",
            test_report,
            "-" * 50,
            "",
        ]
    )

    print(f"Fold {fold} best val_loss: {fold_best:.8f}")
    print(f"Fold {fold} val metrics -> acc={acc:.4f}, bacc={bacc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, kappa={kappa:.4f}")
    print(f"Fold {fold} test metrics -> acc={test_acc:.4f}, bacc={test_bacc:.4f}, prec={test_prec:.4f}, rec={test_rec:.4f}, f1={test_f1:.4f}, kappa={test_kappa:.4f}, pr_auc={test_pr_auc:.4f}")
    print(f"Fold {fold} completed in {time() - start_fold:.2f} secs")

cv_df = pd.DataFrame(cv_rows)
cv_path = results_dir / "time_series_cv_results.txt"
cv_path.write_text(cv_df.to_string(index=False) + "\n", encoding="utf-8")

cv_lines.extend(
    [
        "Results:",
        f"Validation accuracies: {[f'{x:.4f}' for x in cv_df['accuracy']]}",
        f"Validation balanced accuracies: {[f'{x:.4f}' for x in cv_df['balanced_accuracy']]}",
        f"Validation precisions: {[f'{x:.4f}' for x in cv_df['precision']]}",
        f"Validation recalls: {[f'{x:.4f}' for x in cv_df['recall']]}",
        f"Validation F1 scores: {[f'{x:.4f}' for x in cv_df['f1']]}",
        f"Validation kappas: {[f'{x:.4f}' for x in cv_df['kappa']]}",
        f"Test accuracies: {[f'{x:.4f}' for x in cv_df['test_accuracy']]}",
        f"Test balanced accuracies: {[f'{x:.4f}' for x in cv_df['test_balanced_accuracy']]}",
        f"Test precisions: {[f'{x:.4f}' for x in cv_df['test_precision']]}",
        f"Test recalls: {[f'{x:.4f}' for x in cv_df['test_recall']]}",
        f"Test F1 scores: {[f'{x:.4f}' for x in cv_df['test_f1']]}",
        f"Test kappas: {[f'{x:.4f}' for x in cv_df['test_kappa']]}",
        f"Test PR AUCs: {[f'{x:.4f}' for x in cv_df['test_pr_auc']]}",
        f"Mean validation accuracy: {cv_df['accuracy'].mean():.4f}",
        f"Mean validation balanced accuracy: {cv_df['balanced_accuracy'].mean():.4f}",
        f"Mean validation precision: {cv_df['precision'].mean():.4f}",
        f"Mean validation recall: {cv_df['recall'].mean():.4f}",
        f"Mean validation F1 score: {cv_df['f1'].mean():.4f}",
        f"Mean validation kappa: {cv_df['kappa'].mean():.4f}",
        f"Mean test accuracy: {cv_df['test_accuracy'].mean():.4f}",
        f"Mean test balanced accuracy: {cv_df['test_balanced_accuracy'].mean():.4f}",
        f"Mean test precision: {cv_df['test_precision'].mean():.4f}",
        f"Mean test recall: {cv_df['test_recall'].mean():.4f}",
        f"Mean test F1 score: {cv_df['test_f1'].mean():.4f}",
        f"Mean test kappa: {cv_df['test_kappa'].mean():.4f}",
        f"Mean test PR AUC: {cv_df['test_pr_auc'].mean():.4f}",
    ]
)

cv_metrics_path = results_dir / "time_series_cv_metrics.txt"
cv_metrics_path.write_text("\n".join(cv_lines) + "\n", encoding="utf-8")

print("\nSummary CV:")
print(cv_df.to_string(index=False))
print(f"CV results saved in: {cv_path}")
print(f"CV metrics results saved in: {cv_metrics_path}")
print(f"mean best val_loss: {cv_df['best_val_loss'].mean():.8f} ± {cv_df['best_val_loss'].std(ddof=0):.8f}")
print("\nMean validation:")
print(cv_df[["accuracy", "balanced_accuracy", "precision", "recall", "f1", "kappa"]].mean().to_string())
print("\nMean test:")
print(cv_df[["test_accuracy", "test_balanced_accuracy", "test_precision", "test_recall", "test_f1", "test_kappa", "test_pr_auc"]].mean().to_string())
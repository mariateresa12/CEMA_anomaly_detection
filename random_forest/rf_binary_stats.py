# %% [markdown]
# # Random Forest with statistical features per-window

# %%
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.model_selection import KFold

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
    'packet_idx', 'src16', 'dst16', 'length', 'time_rel_us',
    'rssi', 'epoch_us', 'attack', 'dev', 'rpi'
]

frames = [pd.DataFrame(p, columns=columns) for p in (pcks1, pcks2, pcks3, pcks4) if len(p) > 0]
dataframe = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)

numeric_cols = ['packet_idx', 'src16', 'dst16', 'length', 'time_rel_us', 'rssi', 'epoch_us', 'rpi']
dataframe[numeric_cols] = dataframe[numeric_cols].apply(pd.to_numeric, errors='coerce')

dataframe = dataframe.sort_values(by='epoch_us', ignore_index=True)
print(dataframe['attack'].value_counts())

# %% [markdown]
# ## Preprocessing

# %%
df = dataframe.copy()

src_vals = [8957, 43231, 31646, 24250, 3078, 31863, 41643, 2578, 18154, 35010]
df = df[df['src16'].isin(src_vals)].reset_index(drop=True)
df = df[df['dst16'].isin(src_vals)].reset_index(drop=True)

df['date'] = pd.to_datetime(df['epoch_us'], unit='us', utc=True).dt.tz_convert('Europe/Paris')
start_time = pd.Timestamp('2022-07-03 07:00:00', tz='Europe/Paris')
end_time = pd.Timestamp('2022-07-09 00:00:00', tz='Europe/Paris')
df = df[df['date'].between(start_time, end_time)].copy()

mapped = {
    'NO_ATTACK': 0,
    'A': 1,
    'B': 1,
    'C': 1,
    'D': 1,
    'E': 1,
}
df['attack'] = df['attack'].map(mapped).astype(int)

df['delta_t'] = df.groupby(['src16', 'rpi'])['epoch_us'].diff().fillna(0).astype(int)
df['delta_l'] = df.groupby(['src16', 'rpi'])['length'].diff().fillna(0).astype(int)

print(df[['attack']].value_counts())
print(df.shape)

# %% [markdown]
# ## Building statistical features by window

# %%
base_features = ['length', 'rssi', 'delta_t', 'delta_l']
window_size = 8
step = 4
anomaly_fraction_threshold = 0.2

def summarize_window(window_df):
    stats = {}
    for col in base_features:
        s = pd.to_numeric(window_df[col], errors='coerce')
        stats[f'{col}_mean'] = float(s.mean())
        stats[f'{col}_median'] = float(s.median())
        stats[f'{col}_std'] = float(s.std(ddof=0))
        stats[f'{col}_min'] = float(s.min())
        stats[f'{col}_max'] = float(s.max())
        stats[f'{col}_q25'] = float(s.quantile(0.25))
        stats[f'{col}_q75'] = float(s.quantile(0.75))
        stats[f'{col}_iqr'] = float(stats[f'{col}_q75'] - stats[f'{col}_q25'])
    return stats

rows = []
for (_, _), g in df.groupby(['src16', 'rpi'], sort=False):
    g = g.sort_values('epoch_us').reset_index(drop=True)
    for start in range(0, len(g) - window_size + 1, step):
        w = g.iloc[start:start + window_size]
        feat = summarize_window(w)
        feat['label'] = np.int8(w['attack'].mean() >= anomaly_fraction_threshold)
        feat['t_start'] = int(w['epoch_us'].iloc[0])
        rows.append(feat)

stats_df = pd.DataFrame(rows).sort_values('t_start').reset_index(drop=True)

X_stats = stats_df.drop(columns=['label', 't_start'])
y_stats = stats_df['label']

print(f'Número de ventanas: {len(stats_df)}')
print(y_stats.value_counts(normalize=True).rename('ratio'))
X_stats.head()

# %% [markdown]
# ## Split temporal train/test

# %%
split_idx = int(len(X_stats) * 0.8)
X_train = X_stats.iloc[:split_idx].copy()
X_test = X_stats.iloc[split_idx:].copy()
y_train = y_stats.iloc[:split_idx].copy()
y_test = y_stats.iloc[split_idx:].copy()

print('Train shape:', X_train.shape, 'Test shape:', X_test.shape)
print('Train balance:', y_train.value_counts(normalize=True))
print('Test balance:', y_test.value_counts(normalize=True))

# %% [markdown]
# # Random Forest 

# %%
clf = RandomForestClassifier(
    random_state=42,
    n_estimators=800,
    max_features=None,
    max_depth=80,
    min_samples_split=20,
    min_samples_leaf=1,
    bootstrap=True,
    criterion='entropy',
    class_weight=None,
    n_jobs=-1,
    verbose=1
)

clf.fit(X_train, y_train)

model_path = Path('results/RFB_stats.pkl')
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(clf, model_path)

print(f'Model stored in: {model_path.resolve()}')

# %%
clf = joblib.load('results/RFB_stats.pkl')
preds_train = clf.predict(X_train)
preds_test = clf.predict(X_test)

acc_train = accuracy_score(y_train, preds_train)
acc_test = accuracy_score(y_test, preds_test)
bacc_test = balanced_accuracy_score(y_test, preds_test)
prec = precision_score(y_test, preds_test, zero_division=0)
rec = recall_score(y_test, preds_test, zero_division=0)
f1 = f1_score(y_test, preds_test, zero_division=0)
kappa = cohen_kappa_score(y_test, preds_test)
cm = confusion_matrix(y_test, preds_test)
report = classification_report(y_test, preds_test, zero_division=0)

output_text = '\n'.join([
    f'Accuracy (train): {acc_train:.4f}',
    f'Accuracy (test): {acc_test:.4f}',
    f'Balanced accuracy (test): {bacc_test:.4f}',
    f'Precision: {prec:.4f}',
    f'Recall: {rec:.4f}',
    f'F1 score: {f1:.4f}',
    f'Coeficiente kappa: {kappa:.4f}',
    'Confusion matrix:',
    str(cm),
    '',
    report,
])
print(output_text)

out_path = Path('results/RFB_stats.txt')
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(output_text + '\n', encoding='utf-8')

print(f'Results saved to: {out_path.resolve()}')

# %% [markdown]
### Cross-validation

# %%
kf = KFold(n_splits=5, shuffle=False)

accuracies = []
baccuracies = []
precs = []
recalls = []
f1scores = []
kappas = []
lines = []
lines.extend([
    'KFold Cross-Validation',
    '-' * 50,
])

for fold, (train_index, val_index) in enumerate(kf.split(X_stats, y_stats), start=1):
    X_tr = X_stats.iloc[train_index]
    y_tr = y_stats.iloc[train_index]
    X_val = X_stats.iloc[val_index]
    y_val = y_stats.iloc[val_index]

    
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=800,
        max_features=None,
        max_depth=80,
        min_samples_split=20,
        min_samples_leaf=1,
        bootstrap=True,
        criterion='entropy',
        class_weight=None,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_tr, y_tr)

    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)
    bacc = balanced_accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    kappa = cohen_kappa_score(y_val, preds)

    cm = confusion_matrix(y_val, preds)
    report = classification_report(y_val, preds, zero_division=0)

    accuracies.append(acc)
    precs.append(prec)
    recalls.append(rec)
    baccuracies.append(bacc)
    f1scores.append(f1)
    kappas.append(kappa)

    lines.extend([
        f'Fold {fold}',
        f'Accuracy (validation): {acc:.4f}',
        f'Balanced accuracy (validation): {bacc:.4f}',
        f'Precision (validation): {prec:.4f}',
        f'Recall (validation): {rec:.4f}',
        f'F1 score (validation): {f1:.4f}',
        f'Coeficiente kappa (validation): {kappa:.4f}',
        'Confusion matrix:',
        str(cm),
        '',
        report,
        '-' * 50,
        ''
    ])

lines.extend([
    'Results:',
    f'Accuracies: {[f"{x:.4f}" for x in accuracies]}',
    f'Balanced accuracies: {[f"{x:.4f}" for x in baccuracies]}',
    f'Precisions: {[f"{x:.4f}" for x in precs]}',
    f'Recalls: {[f"{x:.4f}" for x in recalls]}',
    f'F1 scores: {[f"{x:.4f}" for x in f1scores]}',
    f'Kappas: {[f"{x:.4f}" for x in kappas]}',
    f'Mean accuracy: {np.mean(accuracies):.4f}',
    f'Mean balanced accuracy: {np.mean(baccuracies):.4f}',
    f'Mean precision: {np.mean(precs):.4f}',
    f'Mean recall: {np.mean(recalls):.4f}',
    f'Mean F1 score: {np.mean(f1scores):.4f}',
    f'Mean kappa: {np.mean(kappas):.4f}',
])

output_text = '\n'.join(lines)
print(output_text)

out_path = Path('results/RFB_stats_CV.txt')
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(output_text + '\n', encoding='utf-8')

print(f'Results saved to: {out_path.resolve()}')



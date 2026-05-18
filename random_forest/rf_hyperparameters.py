# %%
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score

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

# convertir solo columnas numéricas
numeric_cols = ["packet_idx", "src16", "dst16", "length", "time_rel_us", "rssi", "epoch_us", "rpi"]
dataframe[numeric_cols] = dataframe[numeric_cols].apply(pd.to_numeric, errors="coerce")

dataframe = dataframe.sort_values(by="epoch_us", ignore_index=True)

# %% [markdown]
# ## Preprocessing

# %%
df = dataframe

df['date'] = pd.to_datetime(df['epoch_us'], unit='us', utc=True)
df['date'] = df['date'].dt.tz_convert('Europe/Paris')

print("First packet is at: \t", df['date'].min())
print("Last packet is at: \t", df['date'].max())

# %%
start_time = pd.Timestamp('2022-07-03 07:00:00', tz='Europe/Paris')
end_time = pd.Timestamp('2022-07-09 00:00:00', tz='Europe/Paris')

df = df[df['date'].between(start_time, end_time)]

n_at = df[df['attack'] != 'NO_ATTACK'].shape[0]
n_no_at = df[df['attack'] == 'NO_ATTACK'].shape[0]

print(df['attack'].value_counts())
print(f"Total de ataques: {n_at}")
print(f"Porcentaje de ataques: {n_at/(n_at+n_no_at)*100:.2f}%")

# %%
src_vals = [8957, 43231, 31646, 24250, 3078, 31863, 41643, 2578, 18154, 35010]

df = df[df['src16'].isin(src_vals)].reset_index(drop=True)
df = df[df['dst16'].isin(src_vals)].reset_index(drop=True)

print(src_vals)
print([hex(x) for x in src_vals])
print(dataframe.shape)
print(df.shape)

# %%
df["delta_t"] = df.groupby(["src16", "rpi"])["epoch_us"].diff().fillna(0).astype(int)
df["delta_l"] = df.groupby(["src16", "rpi"])["length"].diff().fillna(0).astype(int)

df.head()

# %%
df=df[['length', 'rssi', 'attack', 'delta_t', 'delta_l']]
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

# %% [markdown]
# ## Random Forest

# %%
X, X_test, y, y_test = train_test_split(df.drop(columns=['attack']), df['attack'], test_size=0.2, random_state=42, stratify=df['attack'])

# %% [markdown]
# ### Hyperparameters tuning

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# %%
# Number of trees in random forest
n_estimators = [400, 600, 800, 1000]
# Number of features to consider at every split
max_features = ["sqrt", None]
# Maximum number of levels in tree
max_depth = [None, 80]
# Minimum number of samples required to split a node
min_samples_split = [20, 50]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1]
# Method of selecting samples for training each tree
bootstrap = [True]
# Criterion
criterion=["gini", "entropy"]
# Class weight
class_weight=[None]



params_grid = { 'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap,
                'criterion': criterion,
                'class_weight': class_weight
                }

# %%
model_base = RandomForestClassifier(verbose=1)

rs = RandomizedSearchCV(estimator = model_base,
                        scoring='f1',
                        param_distributions = params_grid,
                        n_iter = , cv=5,
                        random_state=42, n_jobs = 4,
                        verbose=2)
rs.fit(X, y)

joblib.dump(rs, "results/RFB_hyperparameters.pkl")

# %%
rs = joblib.load("results/RFB_hyperparameters.pkl")

best_model = rs.best_estimator_
preds = best_model.predict(X_test)

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
kappa = cohen_kappa_score(y_test, preds)
cm = confusion_matrix(y_test, preds)
report = classification_report(y_test, preds)

cv_df = pd.DataFrame(rs.cv_results_)

cols_results = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "mean_fit_time",
    "mean_score_time",
    "params"
]

top_cv = cv_df[cols_results].sort_values("rank_test_score").head(10)

output_text = "\n".join([
    "Hyperparameters search results:",
    "-" * 50,
    f"Best CV F1 score: {rs.best_score_:.4f}",
    f"Best hyperparameters: {rs.best_params_}",
    "",
    "Top CV candidates:",
    top_cv.to_string(index=False),
    "",
    "Test results:",
    f"Accuracy: {acc:.4f}",
    f"F1 score: {f1:.4f}",
    f"Coeficiente kappa: {kappa:.4f}",
    "Confusion matrix:",
    str(cm),
    "",
    report,
])

print(output_text)

out_path = Path("results/RFB_hyperparameters.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(output_text + "\n", encoding="utf-8")

print(f"Results saved to: {out_path.resolve()}")




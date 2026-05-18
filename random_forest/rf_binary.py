# %% [markdown]
# Suspresión de columnas: source, dest, epoch, rpi, tipo de dispositivo

# %%
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

# %% [markdown]
# ## Cargar JSON y preparar dataframe

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
print(dataframe['attack'].value_counts())

# %% [markdown]
# ## Preprocesado del dataframe

# %% [markdown]
# Nos quedamos con las filas con identificadores conocidos (src_vals)

# %%
df = dataframe

src_vals = [8957, 43231, 31646, 24250, 3078, 31863, 41643, 2578, 18154, 35010]

df = df[df['src16'].isin(src_vals)].reset_index(drop=True)
df = df[df['dst16'].isin(src_vals)].reset_index(drop=True)

print(src_vals)
print([hex(x) for x in src_vals])
print(dataframe.shape)
print(df.shape)

# %% [markdown]
# Añadir fecha legible

# %%
df['date'] = pd.to_datetime(df['epoch_us'], unit='us', utc=True)
df['date'] = df['date'].dt.tz_convert('Europe/Paris')

print("First packet is at: \t", df['date'].min())
print("Last packet is at: \t", df['date'].max())

# %%
# Dataframe en franja de ataques
start_time = pd.Timestamp('2022-07-03 07:00:00', tz='Europe/Paris')
end_time = pd.Timestamp('2022-07-09 00:00:00', tz='Europe/Paris')

df = df[df['date'].between(start_time, end_time)]

n_at = df[df['attack'] != 'NO_ATTACK'].shape[0]
n_no_at = df[df['attack'] == 'NO_ATTACK'].shape[0]

print(df['attack'].value_counts())
print(f"Total de ataques: {n_at}")
print(f"Porcentaje de ataques: {n_at/(n_at+n_no_at)*100:.2f}%")

# %% [markdown]
# Extraer diferencia de tiempo entre paquetes de la misma rpi

# %%
df["delta_t"] = df.groupby(["src16", "rpi"])["epoch_us"].diff().fillna(0).astype(int)
df["delta_l"] = df.groupby(["src16", "rpi"])["length"].diff().fillna(0).astype(int)

df.head()

# %% [markdown]
# Eliminar columnas inncesarias (src16, dst16, packet_idx, time_rel_us, date, rpi)

# %%
df=df[['length', 'rssi', 'attack', 'delta_t', 'delta_l']]
df.head()

# %% [markdown]
# Mapear tipo de ataque a entero

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
# Separamos en train y test
# Ojo: estamos perdiendo la temporalidad al hacer el stratified split.
X, X_test, y, y_test = train_test_split(df.drop(columns=['attack']), df['attack'], test_size=0.2, random_state=42, stratify=df['attack'])

# %% [markdown]
# ### Básico

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
clf = RandomForestClassifier(random_state=42, verbose=1,
                             n_estimators=200,
                             max_features=None,
                             max_depth=80,
                             min_samples_split=50,
                             min_samples_leaf=1,
                             bootstrap = True,
                             criterion='entropy',
                             class_weight=None)
clf.fit(X, y)
joblib.dump(clf, "results3/RFB_length.pkl")

# %%
clf = joblib.load("results3/RFB_length.pkl")
preds = clf.predict(X_test)

acc_train = accuracy_score(y, clf.predict(X))
acc_test = accuracy_score(y_test, preds)
bacc_test = balanced_accuracy_score(y_test, preds)
prec = precision_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
kappa = cohen_kappa_score(y_test, preds)
cm = confusion_matrix(y_test, preds)
report = classification_report(y_test, preds)


output_text = "\n".join([
    f"Accuracy (train): {acc_train}",
    f"Accuracy (test): {acc_test}",
    f"Balanced accuracy (test): {bacc_test}",
    f"Precision: {prec}",
    f"Recall: {rec}",
    f"F1 score: {f1}",
    f"Coeficiente kappa: {kappa}",
    "Confusion matrix:",
    str(cm),
    "",
    report,
])
print(output_text)

out_path = Path("results3/RFB_length.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(output_text + "\n", encoding="utf-8")

print(f"Resultados guardados en: {out_path.resolve()}")

# %%
from sklearn.inspection import permutation_importance
feature_names = ['length', 'rssi', 'delta_t', 'delta_l']

# Feature importance based on mean decrease in impurity (MDI)
# MDI = importancia según cómo el bosque construyó sus árboles
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots(figsize=(10, 5))
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

out_path = Path("results3/RFB_basic_MDI_import.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()

# Feature importance based on permutation importance
# Permutation = importancia según cuánto empeora el rendimiento al quitar esa variable
result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
forest_importances = pd.Series(result.importances_mean, index=feature_names)

fig, ax = plt.subplots(figsize=(10, 5))
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()

out_path = Path("results3/RFB_basic_permut_import.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot guardado en: {out_path.resolve()}")

# %% [markdown]
# ### K-fold cross validation

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

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
    "K-fold Cross-Validation",
    "-" * 50,])

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_val = X.iloc[val_index]
    y_val = y.iloc[val_index]

    model = RandomForestClassifier(random_state=42, verbose=1,
                             n_estimators=200,
                             max_features=None,
                             max_depth=80,
                             min_samples_split=50,
                             min_samples_leaf=1,
                             bootstrap = True,
                             criterion='entropy',
                             class_weight=None)
    model.fit(X_train, y_train)

    # Guardar modelo en memoria
    model_name = f"results3/RFB_CV_fold{fold+1}.pkl"
    joblib.dump(model, model_name)

    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)
    bacc = balanced_accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds)
    rec = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    kappa = cohen_kappa_score(y_val, preds)

    cm = confusion_matrix(y_val, preds)
    report = classification_report(y_val, preds)

    accuracies.append(acc)
    precs.append(prec)
    recalls.append(rec)
    baccuracies.append(bacc)
    f1scores.append(f1)
    kappas.append(kappa)

    lines.extend([
        f"Fold {fold + 1}",
        f"Accuracy (validation): {acc:.4f}",
        f"Balanced accuracy (validation): {bacc:.4f}",
        f"Precision (validation): {prec:.4f}",
        f"Recall (validation): {rec:.4f}",
        f"F1 score (validation): {f1:.4f}",
        f"Coeficiente kappa (validation): {kappa}",
        "Confusion matrix:",
        str(cm),
        "",
        report,
        "-" * 50,
        ""
    ])


lines.extend([
    "Results:",
    f"Accuracies: {[f'{x:.4f}' for x in accuracies]}",
    f"Balanced accuracies: {[f'{x:.4f}' for x in baccuracies]}",
    f"Precisions: {[f'{x:.4f}' for x in precs]}",
    f"Recalls: {[f'{x:.4f}' for x in recalls]}",
    f"F1 scores: {[f'{x:.4f}' for x in f1scores]}",
    f"Kappas: {[f'{x:.4f}' for x in kappas]}",
    f"Mean accuracy: {np.mean(accuracies):.4f}",
    f"Mean balanced accuracy: {np.mean(baccuracies):.4f}",
    f"Mean precision: {np.mean(precs):.4f}",
    f"Mean recall: {np.mean(recalls):.4f}",
    f"Mean F1 score: {np.mean(f1scores):.4f}",
    f"Mean kappa: {np.mean(kappas):.4f}",
])

output_text = "\n".join(lines)

print(output_text)

out_path = Path("results3/RFB_basic_CV.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(output_text + "\n", encoding="utf-8")

print(f"Resultados guardados en: {out_path.resolve()}")

# %%
# kf = KFold(n_splits=5, shuffle=False)

# accuracies = []
# baccuracies = []
# precs = []
# recalls = []
# f1scores = []
# kappas = []
# lines = []
# lines.extend([
#     "K-fold Cross-Validation",
#     "-" * 50,])

# for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
#     X_train = X.iloc[train_index]
#     y_train = y.iloc[train_index]
#     X_val = X.iloc[val_index]
#     y_val = y.iloc[val_index]

#     # Guardar modelo en memoria
#     model_name = f"results3/RFB_CV_fold{fold+1}.pkl"
#     model = joblib.load(model_name)
    
#     preds = model.predict(X_val)

#     acc = accuracy_score(y_val, preds)
#     bacc = balanced_accuracy_score(y_val, preds)
#     prec = precision_score(y_val, preds)
#     rec = recall_score(y_val, preds)
#     f1 = f1_score(y_val, preds)
#     kappa = cohen_kappa_score(y_val, preds)

#     cm = confusion_matrix(y_val, preds)
#     report = classification_report(y_val, preds)

#     accuracies.append(acc)
#     precs.append(prec)
#     recalls.append(rec)
#     baccuracies.append(bacc)
#     f1scores.append(f1)
#     kappas.append(kappa)

#     lines.extend([
#         f"Fold {fold + 1}",
#         f"Accuracy (validation): {acc:.4f}",
#         f"Balanced accuracy (validation): {bacc:.4f}",
#         f"Precision (validation): {prec:.4f}",
#         f"Recall (validation): {rec:.4f}",
#         f"F1 score (validation): {f1:.4f}",
#         f"Coeficiente kappa (validation): {kappa}",
#         "Confusion matrix:",
#         str(cm),
#         "",
#         report,
#         "-" * 50,
#         ""
#     ])


# lines.extend([
#     "Results:",
#     f"Accuracies: {[f'{x:.4f}' for x in accuracies]}",
#     f"Balanced accuracies: {[f'{x:.4f}' for x in baccuracies]}",
#     f"Precisions: {[f'{x:.4f}' for x in precs]}",
#     f"Recalls: {[f'{x:.4f}' for x in recalls]}",
#     f"F1 scores: {[f'{x:.4f}' for x in f1scores]}",
#     f"Kappas: {[f'{x:.4f}' for x in kappas]}",
#     f"Mean accuracy: {np.mean(accuracies):.4f}",
#     f"Mean balanced accuracy: {np.mean(baccuracies):.4f}",
#     f"Mean precision: {np.mean(precs):.4f}",
#     f"Mean recall: {np.mean(recalls):.4f}",
#     f"Mean F1 score: {np.mean(f1scores):.4f}",
#     f"Mean kappa: {np.mean(kappas):.4f}",
# ])

# output_text = "\n".join(lines)

# print(output_text)

# out_path = Path("results3/RFB_basic_CV.txt")
# out_path.parent.mkdir(parents=True, exist_ok=True)
# out_path.write_text(output_text + "\n", encoding="utf-8")

# print(f"Resultados guardados en: {out_path.resolve()}")

# %% [markdown]
# ### Búsqueda de hiperparámetros

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# %%
# Number of trees in random forest
n_estimators = [200, 400, 800]
# Number of features to consider at every split
max_features = ["sqrt", None]
# Maximum number of levels in tree
max_depth = [None, 80, 120]
# Minimum number of samples required to split a node
min_samples_split = [20, 40, 50]
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
                        n_iter = 5, cv=5,
                        random_state=42, n_jobs = 4,
                        verbose=2)
rs.fit(X, y)

joblib.dump(rs, "results3/RFB_hyperparameters.pkl")

# %%
rs = joblib.load("results3/RFB_hyperparameters.pkl")

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

out_path = Path("results3/RFB_hyperparameters.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(output_text + "\n", encoding="utf-8")

print(f"Resultados guardados en: {out_path.resolve()}")

# %%




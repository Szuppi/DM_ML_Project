# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# %%
xtrain = pd.read_csv('X_train.csv')
ytrain = pd.read_csv('y_train.csv')
#print(f"X_train:\n{xtrain.describe()}\n\nY_train:\n{ytrain.describe()}")

# %%
#print(f"Y_train value counts:\n{ytrain['Prediction'].value_counts()},\nN/A-s in X_train second column:\n{xtrain['ab_000'].isna().sum()}")

# %%
max_na = 0
for col in xtrain.columns:
    na_count = xtrain[col].isna().sum()
    if na_count > max_na:
        max_na = na_count
        max_na_col = col
print(f"Column with max N/A-s: {max_na_col} ({max_na} N/A-s)")

# %%
# Core
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import fbeta_score, precision_recall_curve, average_precision_score

# XGBoost
from xgboost import XGBClassifier

# Bayesian Optimization
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Misc
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42


# %% [markdown]
# ## Load Data

# %%
train_path = "X_train.csv"   # <-- adjust paths if needed
y_path     = "y_train.csv"
test_path  = "X_test.csv"

X = pd.read_csv(train_path)
y = pd.read_csv(y_path)
X_test = pd.read_csv(test_path)

print("Shapes:")
print("X:", X.shape)
print("y:", y.shape)
print("X_test:", X_test.shape)

# %% [markdown]
# ## Convert "na" --> np.nan

# %%
X = X.replace("na", np.nan)
X_test = X_test.replace("na", np.nan)

# Convert to floats
X = X.astype(float)
X_test = X_test.astype(float)

# %% [markdown]
# ## Class Imbalance

# %%
print(y['Prediction'].value_counts())

neg = y['Prediction'].value_counts()['neg']
pos = y['Prediction'].value_counts()['pos']
scale_pos_weight = neg / pos

print(f"\nscale_pos_weight ≈ {scale_pos_weight:.2f}") # for XGBoost

# %% [markdown]
# ## Missing Values

# %%
missing_frac = X.isna().mean().sort_values(ascending=False)
missing_frac.head(20)

# Plot missing value fractions
plt.figure(figsize=(14,5))
missing_frac.head(40).plot(kind="bar")
plt.title("Top 40 Features by Missing Fraction")
plt.ylabel("Fraction Missing")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(missing_frac, bins=30)
plt.title("Distribution of Missing Fractions Across Features")
plt.xlabel("Fraction Missing")
plt.ylabel("Count")
plt.show()


# %% [markdown]
# ## Drop extremely sparse features

# %%
# Thresholds for the two pipelines
thr_kbest = 0.30   # only use relatively dense features for KBest
thr_xgb   = 0.95   # keep almost everything for XGBoost-based pipeline

cols_kbest = missing_frac[missing_frac <= thr_kbest].index
cols_xgb   = missing_frac[missing_frac <= thr_xgb].index

print(f"Features for KBest pipeline (<= {thr_kbest*100:.0f}% missing): {len(cols_kbest)}")
print(f"Features for XGB/SFM pipeline (<= {thr_xgb*100:.0f}% missing): {len(cols_xgb)}")


# %% [markdown]
# ## Single train-validation split, then branch per pipeline

# %%
from sklearn.model_selection import train_test_split

y['Prediction'] = y['Prediction'].map({'neg': 0, 'pos': 1})

# Single split so both pipelines see exactly the same instances
X_train_all, X_val_all, y_train, y_val = train_test_split(
    X, y['Prediction'],
    test_size=0.20,
    stratify=y['Prediction'],
    random_state=RANDOM_STATE
)

print("Overall Train:", X_train_all.shape)
print("Overall Val:  ", X_val_all.shape)

# Now create pipeline-specific views
Xk_train = X_train_all[cols_kbest].copy()
Xk_val   = X_val_all[cols_kbest].copy()

Xx_train = X_train_all[cols_xgb].copy()
Xx_val   = X_val_all[cols_xgb].copy()

# Also prepare test matrices for later
Xk_test = X_test[cols_kbest].copy()
Xx_test = X_test[cols_xgb].copy()

print("KBest Train shape:", Xk_train.shape)
print("SFM/XGB Train shape:", Xx_train.shape)


# %% [markdown]
# ## Idea
# - Maybe do a variance thresholding before doing the SelectKBest() and SelectFromModel() methods for feature selection?

# %% [markdown]
# ## Pipelines
# 
# - I will train two pipelines to begin with. Both will use XGBoost, but with different feature selection approaches:
#     - Pipeline A will use sklearn's SelectKBest() method
#     - Pipeline B will use sklearn's SelectFromModel() method, with XGBoost as the model

# %% [markdown]
# ### Pipeline A - SelectKBest()

# %%
from sklearn.impute import SimpleImputer

# %%
# -------- Pipeline A: SelectKBest + XGBoost --------
pipe_kbest = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # handle NaNs
    ("scaler", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif, k=40)),  # k tuned
    ("clf", XGBClassifier(
        tree_method="hist",
        #predictor="gpu_predictor",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )),
])

search_kbest = {
    "select__k": Integer(20, min(120, len(cols_kbest))),  # sanity upper bound
    "clf__max_depth": Integer(3, 10),
    "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
    "clf__subsample": Real(0.5, 1.0),
    "clf__colsample_bytree": Real(0.5, 1.0),
    "clf__min_child_weight": Integer(1, 10),
    "clf__gamma": Real(0.0, 5.0),
}

opt_kbest = BayesSearchCV(
    estimator=pipe_kbest,
    search_spaces=search_kbest,
    n_iter=40,              # adjust if needed
    cv=5,
    scoring="balanced_accuracy",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

print("Fitting KBest + XGBoost pipeline...")
opt_kbest.fit(Xk_train, y_train)
print("Best KBest params:", opt_kbest.best_params_)


# %%
pred_kb  = opt_kbest.best_estimator_.predict(Xk_val)
ba_kb  = balanced_accuracy_score(y_val, pred_kb)
print("Balanced Accuracy (KBest pipeline):       ", ba_kb)
print("\n=== KBest Classification Report ===")
print(classification_report(y_val, pred_kb))

# %% [markdown]
# ### Pipeline B - SelectFromModel()

# %%
# -------- Pipeline B: SelectFromModel(XGB) + XGBoost --------

# XGB model for feature selection
xgb_fs = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight
)

pipe_sfm = Pipeline([
    # no imputer, no scaler — let XGB see the raw NaN pattern
    ("select", SelectFromModel(estimator=xgb_fs, threshold="median")),
    ("clf", XGBClassifier(
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )),
])

search_sfm = {
    "clf__max_depth": Integer(3, 10),
    "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
    "clf__subsample": Real(0.5, 1.0),
    "clf__colsample_bytree": Real(0.5, 1.0),
    "clf__min_child_weight": Integer(1, 10),
    "clf__gamma": Real(0.0, 5.0),
}

opt_sfm = BayesSearchCV(
    estimator=pipe_sfm,
    search_spaces=search_sfm,
    n_iter=40,
    cv=5,
    scoring="balanced_accuracy",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

print("Fitting SelectFromModel + XGBoost pipeline...")
opt_sfm.fit(Xx_train, y_train)
print("Best SFM params:", opt_sfm.best_params_)


# %% [markdown]
# ## Check which model was better

# %%
# --- Predictions on hold-out validation ---
pred_kb  = opt_kbest.best_estimator_.predict(Xk_val)
pred_sfm = opt_sfm.best_estimator_.predict(Xx_val)

ba_kb  = balanced_accuracy_score(y_val, pred_kb)
ba_sfm = balanced_accuracy_score(y_val, pred_sfm)

print("Balanced Accuracy (KBest pipeline):       ", ba_kb)
print("Balanced Accuracy (SelectFromModel pipeline):", ba_sfm)

print("\n=== KBest Classification Report ===")
print(classification_report(y_val, pred_kb))

print("\n=== SelectFromModel Classification Report ===")
print(classification_report(y_val, pred_sfm))


# %% [markdown]
# ## Choose best pipeline and train on full train + valid data

# %%
# Full design matrices for final training
X_kbest_full = X[cols_kbest].copy()
X_xgb_full   = X[cols_xgb].copy()

# Choose model
if ba_sfm > ba_kb:
    print("\nUsing SelectFromModel pipeline for final training.")
    final_pipe = opt_sfm.best_estimator_
    X_train_final = X_xgb_full
    X_test_final  = Xx_test
else:
    print("\nUsing KBest pipeline for final training.")
    final_pipe = opt_kbest.best_estimator_
    X_train_final = X_kbest_full
    X_test_final  = Xk_test

# Fit on all available labelled data
final_pipe.fit(X_train_final, y['Prediction'])


# %%
print(final_pipe.named_steps['clf'])


# %%
print("Final model:", final_pipe)
print("Number of features used:", final_pipe.named_steps['select'].k)
preds = final_pipe.predict(X_train_final)
print("Balanced accuracy on full training:", balanced_accuracy_score(y['Prediction'], preds))


# %%
y_test_pred = final_pipe.predict(X_test_final)

# Convert predictions to original labels if needed
y_test_pred = np.where(y_test_pred == 1, 'pos', 'neg')

submission = pd.DataFrame({
    "Id": np.arange(len(y_test_pred)),
    "Prediction": y_test_pred
})

submission.to_csv("my_submission.csv", index=False)
print("Saved my_submission.csv")


# %% [markdown]
# ## Saving Models and Data for Reproducibility

# %%
# Get prefix for saved files
import datetime
date = datetime.datetime.now()
date_prefix = f"{date.year}{date.month}{date.day}"
print("Date prefix for saved files:", date_prefix)

# %% [markdown]
# ### Models

# %%
import joblib

# Save the best estimators found by BayesSearchCV
joblib.dump(opt_kbest.best_estimator_, f"model_kbest_pipeline{opt_kbest.best_estimator_.named_steps['select'].k}_{date_prefix}.pkl")
joblib.dump(opt_sfm.best_estimator_, f"model_sfm_pipeline_{date_prefix}.pkl")

print(f"Saved: model_kbest_pipeline{opt_kbest.best_estimator_.named_steps['select'].k}_{date_prefix}.pkl")
print(f"Saved: model_sfm_pipeline_{date_prefix}.pkl")


# %% [markdown]
# ### Hyperparameters

# %%
import json

params_to_save = {
    "kbest_best_params": opt_kbest.best_params_,
    "sfm_best_params": opt_sfm.best_params_,
    "scale_pos_weight": scale_pos_weight,
    "num_features_kbest": int(opt_kbest.best_estimator_.named_steps["select"].k),
    "um_features_sfm": int(len(cols_xgb)),   # SFM keeps features above median only at fit time
    "cols_kbest": list(cols_kbest),
    "ols_xgb": list(cols_xgb)
}

with open(f"model_metadata_{date_prefix}.json", "w") as f:
    json.dump(params_to_save, f, indent=4)

print(f"model_metadata_{date_prefix}.json")


# %% [markdown]
# ### Final Model

# %%
joblib.dump(final_pipe, f"final_model_{date_prefix}.pkl")
print(f"Saved: final_model_{date_prefix}.pkl")

# %% [markdown]
# ### BayesSearchCV() Results

# %%
df_kbest_cv = pd.DataFrame(opt_kbest.cv_results_)
f_sfm_cv   = pd.DataFrame(opt_sfm.cv_results_)

df_kbest_cv.to_csv(f"kbest_cv_results{opt_kbest.best_estimator_.named_steps['select'].k}_{date_prefix}.csv", index=False)
f_sfm_cv.to_csv(f"sfm_cv_results_{date_prefix}.csv", index=False)

print(f"Saved: kbest_cv_results_{date_prefix}.csv, sfm_cv_results_{date_prefix}.csv")


# %% [markdown]
# ## Load like this

# %%
import joblib

kbest_model = joblib.load("model_kbest_pipeline120_2025127.pkl")
sfm_model   = joblib.load("model_sfm_pipeline_2025127.pkl")
final_model = joblib.load("final_model_2025127.pkl")

# %%
# --- Predictions on hold-out validation ---
pred_final  = kbest_model.predict(Xk_val)
pred_sfm = sfm_model.predict(Xx_val)
pred_final_model = final_model.predict(X_train_final)

ba_kb  = balanced_accuracy_score(y_val, pred_kb)
ba_sfm = balanced_accuracy_score(y_val, pred_sfm)
ba_final = balanced_accuracy_score(y['Prediction'], pred_final_model)

print("Balanced Accuracy (KBest pipeline):       ", ba_kb)
print("Balanced Accuracy (SelectFromModel pipeline):", ba_sfm)
print("Balanced Accuracy (Final loaded model):      ", ba_final)

print("\n=== KBest Classification Report ===")
print(classification_report(y_val, pred_kb))

print("\n=== SelectFromModel Classification Report ===")
print(classification_report(y_val, pred_sfm))

print("\n=== Final Model Classification Report ===")
print(classification_report(y['Prediction'], pred_final_model))


# %% [markdown]
# # Visualizations

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, balanced_accuracy_score
from xgboost import XGBClassifier

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %% [markdown]
# ### Bar chart and histogram to show *missing values*

# %%
# Bar Chart of Missing Fractions
missing_frac = X.isna().mean().sort_values(ascending=False)

plt.figure(figsize=(10,6))
missing_frac.head(25).plot(kind='bar')
plt.ylabel("Fraction Missing")
plt.title("Top 25 Features by Missingness")
plt.tight_layout()
plt.show()


# %%
# Histogram of Missing Fractions
plt.figure(figsize=(8,5))
plt.hist(missing_frac, bins=30, color='steelblue')
plt.xlabel("Missing Fraction")
plt.ylabel("Number of Features")
plt.title("Distribution of Feature Missingness")
plt.show()

# %% [markdown]
# ### Class Imbalance

# %%
plt.figure(figsize=(5,5))
y['Prediction'].value_counts().plot(kind='bar', color=['cornflowerblue','salmon'])
plt.xticks([0,1], ['Negative','Positive'])
plt.title("Class Distribution")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# ### Feature Density Before vs After Filtering

# %%
plt.figure(figsize=(8,5))
sns.kdeplot(1 - missing_frac, label="All Features", linewidth=2)
sns.kdeplot(1 - missing_frac[cols_kbest], label="KBest Features", linewidth=2)
sns.kdeplot(1 - missing_frac[cols_xgb], label="XGB Features", linewidth=2, linestyle='--')
plt.xlabel("Observed Data Fraction (1 - Missingness)")
plt.ylabel("Density")
plt.title("Feature Density Before vs After Filtering")
plt.legend()
plt.show()

# %% [markdown]
# ### Table of Retained Counts

# %%
pd.DataFrame({
    "Stage": ["Original", "KBest Preprocessing", "XGB Preprocessing"],
    "Num Features": [X.shape[1], len(cols_kbest), len(cols_xgb)]
})


# %% [markdown]
# ## Hyperparamter Tuning

# %% [markdown]
# ### BayesSearchCV convergence (best score vs iteration)

# %%
df_kbest_cv = pd.read_csv("kbest_cv_results_2025127.csv")
df_sfm_cv   = pd.read_csv("sfm_cv_results_2025127.csv")

# %%
df_kbest_cv.head()


# %%
plt.figure(figsize=(8,5))
plt.plot(df_kbest_cv["mean_test_score"], marker='o')
plt.xlabel("Iteration")
plt.ylabel("Balanced Accuracy (CV)")
plt.title("BayesSearchCV Convergence – KBest Pipeline")
plt.grid(True)
plt.show()


# %%
plt.figure(figsize=(8,5))
plt.plot(df_sfm_cv["mean_test_score"], marker='o', color='orange')
plt.xlabel("Iteration")
plt.ylabel("Balanced Accuracy (CV)")
plt.title("BayesSearchCV Convergence – SFM Pipeline")
plt.grid(True)
plt.show()


# %%
pd.DataFrame(opt_kbest.best_params_.items(), columns=["Parameter", "Value"])
pd.DataFrame(opt_sfm.best_params_.items(), columns=["Parameter", "Value"])

# %% [markdown]
# ## Model Evaluation Visualizations

# %%
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()

plot_conf_matrix(y_val, pred_kb, "Confusion Matrix – KBest Pipeline (On Holdout)")
plot_conf_matrix(y_val, pred_sfm, "Confusion Matrix – SFM Pipeline (On Holdout)")


# %%
plt.figure(figsize=(6,4))
plt.bar(["KBest", "SFM"], [ba_kb, ba_sfm], color=['skyblue','salmon'])
plt.ylabel("Balanced Accuracy")
plt.title("Validation Balanced Accuracy Comparison")
plt.ylim(0.90, 0.97)
plt.show()


# %%
y_score_kbest = kbest_model.predict_proba(Xk_val)[:,1]
fpr_kbest, tpr_kbest, _ = roc_curve(y_val, y_score_kbest)
y_score_sfm = sfm_model.predict_proba(Xx_val)[:,1]
fpr_sfm, tpr_sfm, _ = roc_curve(y_val, y_score_sfm)

plt.figure(figsize=(6,5))
plt.plot(fpr_kbest, tpr_kbest, label=r"ROC$_{selk}$ curve")
plt.plot(fpr_sfm, tpr_sfm, label=r"ROC$_{selmod}$ curve")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Final Model")
plt.legend()
plt.show()


# %%
precision_kbest, recall_kbest, _ = precision_recall_curve(y_val, y_score_kbest)
precision_sfm, recall_sfm, _ = precision_recall_curve(y_val, y_score_sfm)

plt.figure(figsize=(6,5))
plt.plot(recall_kbest, precision_kbest, linewidth=2, label="kbest")
plt.plot(recall_sfm, precision_sfm, linewidth=2, label="sfm")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – Final Model")
plt.legend()
plt.show()


# %% [markdown]
# ## Feature Importance

# %%
# Extract the final classifier and selector from the pipeline
final_xgb = final_pipe.named_steps["clf"]
selector  = final_pipe.named_steps["select"]

# Mask of features that were kept by SelectKBest
support_mask = selector.get_support()  # boolean mask of shape (n_features_before_selection,)

# Column names corresponding to selected features
selected_feature_names = X_train_final.columns[support_mask]

# Sanity check: lengths must match now
print("Features seen by XGB:", len(selected_feature_names))
print("Length of importances:", len(final_xgb.feature_importances_))

# Build importance DataFrame
imp_df = pd.DataFrame({
    "feature": selected_feature_names,
    "importance": final_xgb.feature_importances_
}).sort_values("importance", ascending=False)

# Plot top 20
plt.figure(figsize=(8,6))
sns.barplot(data=imp_df.head(20), x="importance", y="feature")
plt.title("Top 20 Feature Importances – Final XGBoost Model")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# %%
imp_df.head(10)

# %% [markdown]
# ## Probability Distribution

# %%
plt.figure(figsize=(6,5))
sns.histplot(kbest_model.predict_proba(Xk_val)[:,1], bins=30, kde=True)
plt.title("Distribution of Predicted Positive Probabilities")
plt.xlabel("Predicted Probability (Positive Class)")
plt.ylabel("Count")
plt.show()

# mostly near 0/1 → confident
# mostly in the middle → uncertain

# %% [markdown]
# ## Calibration Curve

# %%
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_val, y_score, n_bins=10)

plt.figure(figsize=(6,5))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("Predicted Probability")
plt.ylabel("Actual Fraction Positives")
plt.title("Calibration Curve – Final Model")
plt.show()


# %%
from graphviz import Digraph

dot = Digraph(comment="Final ML Pipeline")
dot.attr(rankdir='LR', size='10,5')

dot.node("A", "Raw Input Features\n(170 variables)")
dot.node("B", "Missingness Filter\n(≤ 30% NaN)")
dot.node("C", "SimpleImputer\n(median)")
dot.node("D", "StandardScaler")
dot.node("E", "SelectKBest\n(ANOVA, k=120)")
dot.node("F", "XGBoost Classifier\n(depth=3, LR≈0.07)")
dot.node("G", "Predictions")

dot.edges(["AB", "BC", "CD", "DE", "EF", "FG"])

dot.render("pipeline_diagram", format="pdf")




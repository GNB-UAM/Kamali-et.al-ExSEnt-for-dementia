## Stability selection over band-limited complexity and spectral features extracted from EEG dataset for dementia

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Cell 1: Imports for Stability Selection (LOSO, nested, classification) 

import os, warnings, math, glob, re
from collections import defaultdict, Counter
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from joblib import Memory

try:
    from IPython.display import display, HTML
    _CAN_DISPLAY = True
except Exception:
    _CAN_DISPLAY = False
    
# Scikit-learn core
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,classification_report, 
                            confusion_matrix,  roc_curve, log_loss,average_precision_score, precision_recall_curve)
from sklearn.utils import check_random_state
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, StratifiedShuffleSplit, ParameterGrid, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.base import clone

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from matplotlib import cm, colors
from matplotlib.patches import Patch

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

import json, hashlib
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from scipy.io import loadmat

# ---- Threads cap to avoid hidden oversubscription  ----
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "6")
os.environ.setdefault("OMP_NUM_THREADS", "6")

warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
rng = check_random_state(SEED)

print("Imports OK! Ready for LR Stability Selection within LOSO! :)")


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Cell 2a: Load LPFC XX (N × n_trl × n_feat) + labels from MATLAB export 

mat_root = "features file path"
out_dir  = "path to save results"
os.makedirs(out_dir, exist_ok=True)

CLUSTER = "LPFC" # Adjust to the current data file name; we select from  {LPFC, RPFC, LVA, RVA}

mf = os.path.join(mat_root, f"tensor_pure_feats_{CLUSTER}.mat")
if not os.path.isfile(mf):
    hits = sorted(glob.glob(os.path.join(mat_root, f"tensor_pure_feats_*{CLUSTER}*.mat")))
    if len(hits) == 1: mf = hits[0]
    else: raise FileNotFoundError(f"Could not find LPFC MAT in {mat_root}")

dd = loadmat(mf, squeeze_me=True, struct_as_record=False)

X_all = np.asarray(dd["XX"], dtype=float)                      # (N, n_trl, n_feat)
subject_ids = np.asarray(dd["subject_id"]).reshape(-1)

y_CD   = np.asarray(dd["yCD"]).reshape(-1).astype(int)         # Control vs dementia
y_mmse = np.asarray(dd["ymmse"]).reshape(-1)

F = X_all.shape[2]
feat_arr = dd["feature_names"]
feature_names = [str(x).strip() for x in np.asarray(feat_arr).ravel().tolist()]

N = X_all.shape[0]
assert len(subject_ids) == N and len(y_CD) == N and len(feature_names) == F

print(f"[{CLUSTER}] file: {mf}")
print(f"[{CLUSTER}] X_all: {X_all.shape}")
print(f"[{CLUSTER}] y_CD counts: {pd.Series(y_CD).value_counts().to_dict()}")


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Cell 2b: Cohort summary (group/age/gender) 

dd = loadmat(mf, squeeze_me=True, struct_as_record=False)

group  = np.asarray(dd["group_str"]).reshape(-1)
age    = np.asarray(dd["age"], dtype=float).reshape(-1)
gender = np.asarray(dd["gender_raw"]).reshape(-1)

meta = pd.DataFrame({"subject_id": np.asarray(subject_ids).reshape(-1),
                     "group": group, "age": age, "gender": gender})

grp_counts = meta["group"].value_counts()
grp_pct = 100.0 * grp_counts / grp_counts.sum()

pct_table = pd.DataFrame({"count": grp_counts, "percent": grp_pct.round(2)})
pct_table.to_csv(os.path.join(out_dir, f"{CLUSTER}_group_percentages.csv"))

age_summary = (meta.groupby("group")["age"]
               .agg(n="count", mean="mean", std="std", median="median",
                    q25=lambda s: np.nanpercentile(s, 25),
                    q75=lambda s: np.nanpercentile(s, 75))
               .round(2))
age_summary.to_csv(os.path.join(out_dir, f"{CLUSTER}_age_summary.csv"))

gender_ct = (meta.groupby(["group", "gender"]).size()
             .unstack(fill_value=0).astype(int))
gender_ct.to_csv(os.path.join(out_dir, f"{CLUSTER}_gender_counts.csv"))

print(f"[{CLUSTER}] counts:\n{grp_counts.to_string()}")
print(f"\n[{CLUSTER}] %:\n{grp_pct.round(2).to_string()}")
print(f"\n[{CLUSTER}] age:\n{age_summary.to_string()}")
print(f"\n[{CLUSTER}] gender:\n{gender_ct.to_string()}")

group_order = grp_counts.index.tolist()

fig, ax = plt.subplots(figsize=(5, 4), dpi=130)
ax.bar(group_order, grp_pct[group_order].values)
ax.set_ylabel("Percentage of subjects (%)")
ax.set_title(f"{CLUSTER}: Subject distribution (N={grp_counts.sum()})")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, f"{CLUSTER}_group_percentages.png"), bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
ax.boxplot([meta.loc[meta["group"] == g, "age"].to_numpy() for g in group_order],
           labels=group_order, showfliers=False)
ax.set_ylabel("Age (years)")
ax.set_title(f"{CLUSTER}: Age distribution by group")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, f"{CLUSTER}_age_boxplot.png"), bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
bottom = np.zeros(len(group_order))
for col in gender_ct.columns:
    vals = gender_ct[col].reindex(group_order).to_numpy()
    ax.bar(group_order, vals, bottom=bottom, label=col)
    bottom += vals
ax.set_ylabel("Count")
ax.set_title(f"{CLUSTER}: Gender composition by group")
ax.legend(title="Gender", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, f"{CLUSTER}_gender_stacked_bar.png"), bbox_inches="tight")
plt.close(fig)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Cell 2c: Append features: Here we add TAR 

A_name, B_name = "powermean_Theta", "powermean_Alpha"

name2idx = {n.lower(): i for i, n in enumerate(feature_names)}
iA, iB = name2idx[A_name.lower()], name2idx[B_name.lower()]

A = X_all[:, :, iA]
B = X_all[:, :, iB]

ratio = A / B
X_all = np.concatenate([X_all, ratio[:, :, None]], axis=2)

feature_names.append(f"{A_name}/{B_name}")

print("Added:", feature_names[-1])
print("X_all:", X_all.shape, "| n_feat:", len(feature_names))


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Cell 2d: Drop features by name 

DROP_FEATURES = [
    'powermean_Theta','powermean_Alpha',
    'signalmean_Delta','signalmean_Theta','signalmean_Alpha','signalmean_Beta','signalmean_Low Gamma','signalmean_High Gamma',
]

drop = set(n.lower() for n in DROP_FEATURES)
keep_idx = [i for i, n in enumerate(feature_names) if n.lower() not in drop]

X_all = X_all[:, :, keep_idx]
feature_names = [feature_names[i] for i in keep_idx]

print("X_all:", X_all.shape, "| n_feat:", len(feature_names))


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Cell 3: Aggregation; up to here we have the features over consecutive windows, here we aggregate them by computing their statistics

y_labels = y_CD.astype(int)
subject_id = subject_ids.astype(int)

AGGS = ("median", "iqr", "cv") # Informatic and outlier-resistant stats

X_med = np.median(X_all, axis=1)
X_iqr = np.percentile(X_all, 75, axis=1) - np.percentile(X_all, 25, axis=1)
X_cv  = np.std(X_all, axis=1) / (np.abs(np.mean(X_all, axis=1)) + 1e-8)

X_subj = np.concatenate([X_med, X_iqr, X_cv], axis=1)

F = X_all.shape[2]
agg_names = (
    [f"{n}|med" for n in feature_names] +
    [f"{n}|iqr" for n in feature_names] +
    [f"{n}|cv"  for n in feature_names]
)

print("X_subj:", X_subj.shape, "| classes:", pd.Series(y_labels).value_counts().to_dict(),
      "| unique subjects:", len(np.unique(subject_id)))
print("Example agg names:", agg_names[:8])



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Cell 4: Helper function: The main Stability Selection helper (Elastic-Net with LR: from saga) 

def _stratified_subsample_indices(y, frac, rng):
    idx = np.arange(len(y))
    take = []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        n_take = max(2, int(np.floor(frac * len(cls_idx))))
        sel = rng.choice(cls_idx, size=n_take, replace=False)
        take.append(sel)
    return np.sort(np.concatenate(take))


def _sweep_C_path_right_edge(Xs, y, C_grid, q, l1_ratio=None, class_weight='balanced', max_iter=5000, tol=1e-4, seed=0):
    penalty = 'l1' if l1_ratio is None else 'elasticnet'

    clf = LogisticRegression(  penalty=penalty, solver='saga', C=float(C_grid[0]),
        l1_ratio=(None if l1_ratio is None else float(l1_ratio)), class_weight=class_weight, max_iter=max_iter, tol=tol,
        n_jobs=1, random_state=seed, warm_start=True)

    nz = np.empty(len(C_grid), dtype=int)
    models = []

    clf.fit(Xs, y)
    nz[0] = int(np.count_nonzero(clf.coef_))
    models.append(deepcopy(clf))

    for k in range(1, len(C_grid)):
        clf.set_params(C=float(C_grid[k]))
        clf.fit(Xs, y)
        nz[k] = int(np.count_nonzero(clf.coef_))
        models.append(deepcopy(clf))

    idx = np.where(nz == q)[0]
    if idx.size:
        j = idx[-1]
    else:
        d = np.abs(nz - q)
        j = np.flatnonzero(d == d.min())[-1]

    boundary_hit = (j == 0) or (j == len(C_grid)-1)
    chosen = models[j]
    mask = (np.abs(chosen.coef_.ravel()) > 1e-6)
    return float(C_grid[j]), mask, nz, boundary_hit


# ============================== This is the main stability selection function =============================
def stability_selection_logreg( X, y, C_grid, n_subsamples=500, subsample_frac=0.5, l1_ratio=None, pi_thr=0.7, 
    class_weight='balanced',rng=None, verbose=False, return_refit=False, PFER_base= 3):
    rng = check_random_state(rng)
    n, p = X.shape

    # scale once
    Xs_full = StandardScaler().fit_transform(X)
    q_target = max(1, int(np.sqrt( PFER_base * (2 * pi_thr - 1) * p)))

    counts = np.zeros(p, dtype=int)
    q_per_run = []
    C_per_run = []
    boundary_hits = 0

    for t in range(n_subsamples):
        idx = _stratified_subsample_indices(y, frac=subsample_frac, rng=rng)
        Xh = Xs_full[idx]
        yh = y[idx]

        C_sel, mask, nz, bh = _sweep_C_path_right_edge( Xh, yh, C_grid=C_grid,
            q=q_target, l1_ratio=l1_ratio, class_weight=class_weight, seed=int(rng.randint(1_000_000_000)))

        counts += mask.astype(int)
        q_per_run.append(int(mask.sum()))
        C_per_run.append(float(C_sel))
        boundary_hits += bh

        if verbose and (t + 1) % max(1, n_subsamples // 10) == 0:
            print(f"  subsample {t+1}/{n_subsamples}: selected={q_per_run[-1]}")

    freq = counts / float(n_subsamples)
    q_hat = float(np.mean(q_per_run))
    boundary_rate = boundary_hits / float(n_subsamples)

    pfer_bound = (q_hat ** 2) / ((2 * pi_thr - 1 ) * p)
    selected_idx = np.where(freq >= pi_thr)[0]

    out = {'freq': freq, 'selected_idx': selected_idx,'pi_thr': float(pi_thr),
        'q_hat': q_hat,'pfer_bound': pfer_bound, 'C_per_run': np.array(C_per_run),
        'q_per_run': np.array(q_per_run, dtype=int),'boundary_rate': boundary_rate}

    if verbose:
        print(f"----->> l1_ratio:{l1_ratio} <<-------------------------------------------------------")
        print(f"(pi_thr: {pi_thr}, q_hat:{q_hat} and q_target: {q_target} on  C_grid: [{np.min(C_grid)},...,{np.max(C_grid)}] )")
        print(f"unique C = {np.unique(C_per_run)}")
        print("max freq:", freq.max(), "mean nonzero freq:", freq[freq > 0].mean())

    if return_refit and selected_idx.size > 0:
        Xs = StandardScaler().fit_transform(X[:, selected_idx])
        refit = LogisticRegression( penalty='none', solver='lbfgs',  class_weight=class_weight, max_iter=2000)
        refit.fit(Xs, y)
        out['refit_coef'] = refit.coef_.ravel()
        out['refit_intercept'] = float(refit.intercept_)

    return out 

print("Stability selection helper ready (plateau-aware C picking + refit option).")



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Cell 5: More helper functions
def youden_threshold_oof(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    keep = np.isfinite(thr)
    if not np.any(keep):
        return 0.5
    J = tpr[keep] - fpr[keep]
    return float(thr[keep][int(np.argmax(J))])
    
SCALERS = { "robust": RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)),
    "standard": StandardScaler() }


def _stable_seed(params: dict, outer_fold: int, base: int = SEED):
    key = json.dumps(params, sort_keys=True, default=str)
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 1_000_000_000
    return (base + outer_fold * 1_000_000 + h) % (2**31 - 1)


def _oof_probs_with_C(X_tr_sel, y_tr, C, seed, inner_folds, scaler_key="standard", penalty='l2', l1_ratio=None):
    
    n = len(y_tr); oof = np.full(n, np.nan, dtype=float)
    k = max(2, min(inner_folds, int(np.bincount(y_tr).min())))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)


    if penalty in ('l1','elasticnet'):
        solver, l1r = 'saga', (None if penalty=='l1' else float(l1_ratio))
    elif penalty in ('l2','none'):
        solver, l1r = 'lbfgs', None
    else:
        raise ValueError("penalty must be one of: 'l1','elasticnet','l2','none'.")

    base = Pipeline(steps=[ ("imp", SimpleImputer(strategy="median")),  ("sc", SCALERS[scaler_key]),
        ("clf", LogisticRegression( penalty=penalty, solver=solver, C=C, l1_ratio=l1r, class_weight="balanced",
            max_iter=5000, tol=1e-4,  warm_start=False, random_state=seed)) ])
    for tr_i, va_i in skf.split(X_tr_sel, y_tr):
        pipe = clone(base)
        pipe.fit(X_tr_sel[tr_i], y_tr[tr_i])
        oof[va_i] = pipe.predict_proba(X_tr_sel[va_i])[:, 1]
    return oof


# Compute inner-CV balanced accuracies for a given C and preprocessing setup
def _inner_baccs_for_C(X_tr_sel, y_tr, C, seed, inner_folds, scaler_key="standard", penalty='l2', l1_ratio=None, class_weight="balanced"):
    baccs = []
    # skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    k = max(2, min(inner_folds, int(np.bincount(y_tr).min())))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    if penalty in ('l1', 'elasticnet'):
        solver, l1r = 'saga', (None if penalty == 'l1' else float(l1_ratio))
    elif penalty in ('l2', 'none'):
        solver, l1r = 'lbfgs', None
    else:
        raise ValueError("penalty must be one of: 'l1','elasticnet','l2','none'.")

    base_pipe = Pipeline(steps=[ ("imp", SimpleImputer(strategy="median")), ("sc", SCALERS[scaler_key]),
        ("clf", LogisticRegression(penalty=penalty, solver=solver, C=C, l1_ratio=l1r, class_weight=class_weight, 
                                   max_iter=5000, tol=1e-4, warm_start=False, random_state=seed)) ])

    for tr_i, va_i in skf.split(X_tr_sel, y_tr):
        pipe = clone(base_pipe)
        pipe.fit(X_tr_sel[tr_i], y_tr[tr_i])
        prob = pipe.predict_proba(X_tr_sel[va_i])[:, 1]
        y_va = y_tr[va_i]
        preds = (prob >= 0.5).astype(int)
        baccs.append(balanced_accuracy_score(y_va, preds))

    return np.asarray(baccs, float)



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Cell 6: Function to iterate over subjects to perform stability selection with LOSO  

def run_loso_once(X_subj, y_labels, subject_id, STAB_SEARCH_GRID, FINAL_PENALTIES=('l2',), FINAL_C_GRID=np.logspace(-5, 4, 180),
                  GLOBAL_MIN_FREQ=0.7, GLOBAL_TOP_K=20, INNER_FOLDS=3, SCALERS=SCALERS,  agg_names=None, # optional names for features (len = p)
                  out_dir=out_dir,   cluster_label=CLUSTER,    target_label='D',  SEED=42, verbose=True ):

    n, p = X_subj.shape #number of subjects and number of features
    selection_counts = np.zeros(p, dtype=int)
    coef_sum = np.zeros(p, dtype=float)
    coef_abs_sum = np.zeros(p, dtype=float)
    coef_n = np.zeros(p, dtype=int)

    per_fold_freq_full = []
    per_fold_selected = []
    y_true_all, y_prob_all, y_pred_all = [], [], []
    fold_rows = []


    outer_fold = 0
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X_subj, y_labels, subject_id):
        outer_fold += 1
        print(f"----------------------  Fold: {outer_fold}/{n} --------------------- ")
        X_tr_raw, X_te_raw = X_subj[train_idx], X_subj[test_idx]
        y_tr, y_te = y_labels[train_idx], y_labels[test_idx]

        # --- initialise per-fold bests (use BAcc, not AUC) ---
        best_inner_bacc = -np.inf
        BEST_STAB = None
        best_sel_idx_filt = None
        best_q_hat = np.nan
        best_pfer_bound = np.nan
        BEST_SCALER_KEY, BEST_C_FINAL = None, None
        BEST_L1_RATIO, BEST_PENALTY = None, None
        BEST_FREQ_FULL = None
        CHOSEN_BOUNDARY = np.nan
        BEST_C_STATS = None   # will store list of dicts per C for the chosen cand/scaler/penalty

        # TRAIN-only zero-variance filter
        tr_var = np.var(X_tr_raw, axis=0)
        idx_keep = np.where(tr_var > 1e-16)[0]
        X_tr, X_te = X_tr_raw[:, idx_keep], X_te_raw[:, idx_keep]

        best_inner_auc = -np.inf
        BEST_CACC_MIN, BEST_CACC_MAX = np.nan, np.nan
        best_stats_this = None


        # 1) Stability selection over grid + quick screen
        screened = []
        for params in ParameterGrid(STAB_SEARCH_GRID):
            params_local = params.copy()
            params_local["rng"] = _stable_seed(params_local, outer_fold)

            stab = stability_selection_logreg( X_tr, y_tr, C_grid=params_local["C_grid"],
                n_subsamples=params_local["n_subsamples"], subsample_frac=params_local["subsample_frac"],
                l1_ratio=params_local["l1_ratio"], pi_thr=params_local["pi_thr"],
                class_weight=params_local["class_weight"], rng=params_local["rng"],
                verbose=params_local["verbose"], return_refit=False,PFER_base=params_local['PFER'])
            """ stab: 'freq', 'selected_idx' ,'pi_thr','q_hat','pfer_bound', 'C_per_run', 'q_per_run','boundary_rate'"""

            # map freq to FULL feature space
            f_local = np.asarray(stab.get("freq", []), float)
            freq_full = np.zeros(p, dtype=float)
            if f_local.size == X_tr.shape[1]:
                freq_full[idx_keep] = f_local
            elif f_local.size == p:
                freq_full = f_local.copy()
            elif verbose and outer_fold == 1 and f_local.size > 0:
                print(f"[warn] freq len {f_local.size} != {X_tr.shape[1]} or {p}")

            sel_idx_filt = np.array(stab.get("selected_idx", []), dtype=int)
            if sel_idx_filt.size == 0 and f_local.size:
                m = max(1, int(round(max(1 , stab.get("q_hat", 1)))))
                sel_idx_filt = np.argsort(f_local)[::-1][:m]
                print(f" [fold {outer_fold}] no features above pi_thr; forcing top-{m} by freq.")
            if sel_idx_filt.size == 0:
                print(f"  [fold {outer_fold}] candidate skipped (no features selected).")
                continue

            screened.append({ "params": params_local,  "sel_idx_filt": sel_idx_filt,
                "q_hat": float(stab.get("q_hat", np.nan)),  "pfer": float(stab.get("pfer_bound", np.nan)),
                "C_per_run": stab.get("C_per_run", None),  "freq_full": freq_full,
                "boundary_rate": float(stab.get("boundary_rate", np.nan)) })

        if not screened: raise RuntimeError("stability_selection produced no viable candidates.")

        
        # ============================
        # MINIMAL PATCH: if FINAL_PENALTIES are only 'none', skip the entire inner FINAL sweep
        # and choose stability-selection output deterministically (no dependence on FINAL_C_GRID).
        # ============================
        only_none = all([(pen in ("none", None)) for pen in FINAL_PENALTIES])

        if only_none:
            # Deterministic choice among screened candidates without using FINAL_C_GRID.
            # Primary: max q_hat; tie-break: min boundary_rate; then max mean(freq_full).
            def _cand_key(c):
                q  = c.get("q_hat", np.nan)
                br = c.get("boundary_rate", np.nan)
                ff = c.get("freq_full", None)
                ffm = float(np.nanmean(ff)) if isinstance(ff, np.ndarray) and ff.size else -np.inf
                # max q_hat, then min boundary_rate, then max mean freq
                return (np.nan_to_num(q, nan=-np.inf),
                        -np.nan_to_num(br, nan=np.inf),
                        ffm)

            cand_best = max(screened, key=_cand_key)

            # Freeze features from stability selection (NOT from the final fit / inner sweep)
            BEST_STAB          = cand_best["params"]
            best_sel_idx_filt  = cand_best["sel_idx_filt"]
            best_q_hat         = float(cand_best.get("q_hat", np.nan))
            best_pfer_bound    = float(cand_best.get("pfer", np.nan))
            BEST_FREQ_FULL     = cand_best.get("freq_full", None)
            CHOSEN_BOUNDARY    = float(cand_best.get("boundary_rate", np.nan))

            # Placeholders so downstream code doesn't crash; they won't affect selected features.
            BEST_SCALER_KEY    = next(iter(SCALERS.keys()))  # deterministic
            BEST_C_FINAL       = 1.0
            BEST_L1_RATIO      = None
            BEST_PENALTY       = "none"
            BEST_CACC_MIN      = np.nan
            BEST_CACC_MAX      = np.nan
            BEST_C_STATS       = None

        else:
        
            # 2) Inner FINAL sweep with 1-SE rule
            for cand in screened:
                sel_idx_filt = cand["sel_idx_filt"]
                X_tr_sel = X_tr[:, sel_idx_filt]
            
                # best inner-CV BAcc for this stability-selection candidate
                best_bacc_this = -np.inf
                best_C_this, best_scaler_key, best_l1r_this, best_penalty_this = None, None, None, None
                best_Cacc_min_this, best_Cacc_max_this = np.nan, np.nan
    
            
                for scaler_key in SCALERS.keys():
                    for pen in FINAL_PENALTIES:
                        if pen in ("none",None, "l2"):
                            l1r_try = None
                        elif pen == "elasticnet":
                            l1r_try = cand["params"]["l1_ratio"]
                            if l1r_try is None:
                                continue
                        elif pen == "l1":
                            l1r_try = None
                        else:
                            raise ValueError(f"Unknown penalty: {pen}")
            
                        stats = []  # (C, mean_bacc, se_bacc)
                        for C in FINAL_C_GRID:
                            baccs = _inner_baccs_for_C(
                                X_tr_sel, y_tr, C, SEED, INNER_FOLDS,
                                scaler_key=scaler_key, penalty=pen, l1_ratio=l1r_try
                            )
                            mean_bacc = baccs.mean()
                            se_bacc = baccs.std(ddof=1) / np.sqrt(len(baccs)) if baccs.size > 1 else 0.0
                            stats.append((C, mean_bacc, se_bacc))
                        
                        stats_rec = [{"C": float(C), "mean_bacc": float(m), "se_bacc": float(se)} for (C, m, se) in stats]
    
            
                        means = np.array([m for (_, m, _) in stats], float)
                        best_i = int(np.argmax(means))
                        best_mean, best_se = stats[best_i][1], stats[best_i][2]
            
                        # 1-SE rule on BAcc, then choose smallest C among acceptable-> we want the least overfitting
                        thr = best_mean - best_se
                        acceptable = [(C, m) for (C, m, se) in stats if np.isfinite(m) and (m >= thr)]
                        if len(acceptable) == 0:
                            continue
                        
                        C_acc_min = min(acceptable, key=lambda cm: cm[0])[0]
                        C_acc_max = max(acceptable, key=lambda cm: cm[0])[0]
                        
                        C_choice, score_choice = min(acceptable, key=lambda cm: cm[0])
                        
                        if (score_choice > best_bacc_this) or (np.isclose(score_choice, best_bacc_this) and
                                                               (best_C_this is None or C_choice < best_C_this)):
                            best_bacc_this = score_choice
                            best_C_this = C_choice
                            best_scaler_key = scaler_key
                            best_penalty_this = pen
                            best_l1r_this = l1r_try
                            best_Cacc_min_this = C_acc_min
                            best_Cacc_max_this = C_acc_max
                            best_stats_this = stats_rec
    
                                
                if best_bacc_this > best_inner_bacc:
                    best_inner_bacc = best_bacc_this
                    BEST_STAB = cand["params"]
                    best_sel_idx_filt = sel_idx_filt
                    best_q_hat = cand["q_hat"]
                    best_pfer_bound = cand["pfer"]
                    BEST_SCALER_KEY, BEST_C_FINAL = best_scaler_key, best_C_this
                    BEST_PENALTY, BEST_L1_RATIO = best_penalty_this, best_l1r_this
                    BEST_FREQ_FULL = cand["freq_full"]
                    CHOSEN_BOUNDARY = cand["boundary_rate"]
                    BEST_CACC_MIN, BEST_CACC_MAX = best_Cacc_min_this, best_Cacc_max_this
                    BEST_C_STATS = best_stats_this
    
                    
                    # defensive check (optional)
                    if best_sel_idx_filt is None:
                        raise RuntimeError("Inner FINAL sweep failed to select any viable model.")
    
        
        # Map back to full index space
        sel_idx_full = idx_keep[best_sel_idx_filt]
        selection_counts[sel_idx_full] += 1
        if agg_names is not None:
            per_fold_selected.append([agg_names[i] for i in sel_idx_full])
        else:
            per_fold_selected.append(sel_idx_full.tolist())
        per_fold_freq_full.append(BEST_FREQ_FULL if BEST_FREQ_FULL is not None else np.zeros(p, float))

        # OOF train probabilities for metrics + threshold
        X_tr_sel = X_tr[:, best_sel_idx_filt]
        X_te_sel = X_te[:, best_sel_idx_filt]

        oof_probs = _oof_probs_with_C( X_tr_sel, y_tr, BEST_C_FINAL, SEED, INNER_FOLDS, scaler_key=BEST_SCALER_KEY, 
                                       penalty=BEST_PENALTY, l1_ratio=BEST_L1_RATIO)
        thr = 0.5
        # thr = youden_threshold_oof(y_tr, oof_probs)

        loss_tr_oof = log_loss(y_tr, oof_probs, labels=[0, 1])
        yhat_tr_oof = (oof_probs >= thr).astype(int)
        acc_tr_oof  = accuracy_score(y_tr, yhat_tr_oof)
        bacc_tr_oof = balanced_accuracy_score(y_tr, yhat_tr_oof)
        auc_tr_oof  = roc_auc_score(y_tr, oof_probs) if len(np.unique(y_tr)) == 2 else np.nan

        solver = 'lbfgs' if BEST_PENALTY in ('l2','none') else 'saga'
        l1r_use = (float(BEST_L1_RATIO) if BEST_PENALTY == 'elasticnet' else None)
        C_final = (1.0 if BEST_PENALTY == 'none' else (BEST_C_FINAL if BEST_C_FINAL is not None else 1.0))

        final_pipe = Pipeline(steps=[("imp", SimpleImputer(strategy="median")),("sc", SCALERS[BEST_SCALER_KEY]),
                                      ("clf", LogisticRegression(penalty=BEST_PENALTY, solver=solver, C=C_final,
                                                                 l1_ratio=l1r_use, class_weight="balanced", 
                                                                 max_iter=5000, tol=1e-4, warm_start=False, random_state=SEED))
                                    ])
        final_pipe.fit(X_tr_sel, y_tr)

        prob_te = final_pipe.predict_proba(X_te_sel)[:, 1]
        yhat_te = (prob_te >= thr).astype(int)
        loss_te = log_loss(y_te, prob_te, labels=[0, 1])
        acc_te  = accuracy_score(y_te, yhat_te)

        clf = final_pipe.named_steps["clf"]
        if hasattr(clf, "coef_"):
            coef_sel = clf.coef_.ravel()
            coef_sum[sel_idx_full]     += coef_sel
            coef_abs_sum[sel_idx_full] += np.abs(coef_sel)
            coef_n[sel_idx_full]       += 1

        # pooled-outer vectors (first element if LOSO yields single test sample)
        y_true_all.append(int(y_te[0]))
        y_prob_all.append(float(prob_te[0]))
        y_pred_all.append(int(yhat_te[0]))

        bacc_te = balanced_accuracy_score(y_te, yhat_te)

        C_report = float(BEST_C_FINAL if BEST_C_FINAL is not None else 1.0)
        l1r_report = (None if BEST_PENALTY != 'elasticnet' else (None if BEST_L1_RATIO is None else float(BEST_L1_RATIO)))

        if verbose:
            print(
                f"left_out={int(test_idx[0])} | feats={sel_idx_full.size} | "
                f"pen={BEST_PENALTY}{'' if l1r_report is None else f' (l1_ratio={l1r_report:.2f})'}| "
                f"C={C_report:.4g}| scaler={BEST_SCALER_KEY}| "
                f"AUC(tr-OOF)={auc_tr_oof:.3f} | "
                f"BAcc(tr-OOF)={bacc_tr_oof:.3f}  BAcc(te)={bacc_te:.0f} | "
                f"Loss(tr-OOF)={loss_tr_oof:.3f}  Loss(te)={loss_te:.3f}"
            )

        fold_rows.append({
            "fold": outer_fold,
            "left_out_idx": int(test_idx[0]),
            "selected_idx_full": sel_idx_full.tolist(),
            "n_selected": int(sel_idx_full.size),
            "C_used": C_report,
            "C_acc_min": float(BEST_CACC_MIN),
            "C_acc_max": float(BEST_CACC_MAX),
            "q_hat": float(best_q_hat),
            "pfer": float(best_pfer_bound),
            "thr_train": float(thr),
            "auc_train": float(auc_tr_oof),
            "acc_train": float(acc_tr_oof),
            "bacc_train": float(bacc_tr_oof),
            "logloss_train": float(loss_tr_oof),
            "acc_test": float(acc_te),
            "bacc_test": float(bacc_te),
            "logloss_test": float(loss_te),
            "boundary_rate": float(CHOSEN_BOUNDARY),
            "stab_params": BEST_STAB,
            "scaler": BEST_SCALER_KEY,
            "C_curve_final": BEST_C_STATS,   # list of dicts, JSON-serializable
        })


    # --- Outer pooled metrics ---
    y_true_all = np.array(y_true_all, dtype=int)
    y_prob_all = np.array(y_prob_all, dtype=float)
    
    thr_outer = 0.5
    y_pred_all = (y_prob_all >= thr_outer).astype(int)
    
    AUC  = roc_auc_score(y_true_all, y_prob_all)
    ACC  = accuracy_score(y_true_all, y_pred_all)
    BACC = balanced_accuracy_score(y_true_all, y_pred_all)
    F1   = f1_score(y_true_all, y_pred_all)


    # --- Outer metrics at Youden threshold (post-hoc) ---
    thr_youden_outer = youden_threshold_oof(y_true_all, y_prob_all)
    y_pred_youden    = (y_prob_all >= thr_youden_outer).astype(int)
    ACC_y  = accuracy_score(y_true_all, y_pred_youden)
    BACC_y = balanced_accuracy_score(y_true_all, y_pred_youden)
    F1_y   = f1_score(y_true_all, y_pred_youden)
    
    fold_df = pd.DataFrame(fold_rows)

    print("thr_youden_outer (8dp):", f"{thr_youden_outer:.8f}")
    print("thr_youden_outer 17g:", f"{thr_youden_outer:.17g}")
    
    pred05 = (y_prob_all >= 0.5).astype(int)
    predY  = (y_prob_all >= thr_youden_outer).astype(int)
    print("n_pred_diff:", np.sum(pred05 != predY))
    
    lo, hi = (thr_youden_outer, 0.5) if thr_youden_outer < 0.5 else (0.5, thr_youden_outer)
    print("n_scores_between:", np.sum((y_prob_all >= lo) & (y_prob_all < hi)))
# ==================================================

    if verbose:
        tag = target_label if target_label is not None else "—"
        print(f"[Stability-LOSO] N={len(y_true_all)}  Acc={ACC:.3f}  BalAcc={BACC:.3f}  F1={F1:.3f}  AUC={AUC:.3f}")


    # ===== Stability summaries (keep only this block; delete the earlier duplicate) =====
    n_folds = len(per_fold_freq_full)
    stability_freq = selection_counts / max(1, n_folds)
    
    freq_mat  = np.vstack(per_fold_freq_full) if n_folds else np.zeros((0, p), float)
    freq_mean = freq_mat.mean(axis=0) if n_folds else np.zeros(p, float)
    freq_min  = freq_mat.min(axis=0)  if n_folds else np.zeros(p, float)
    
    coef_mean = np.zeros(p, float)
    coef_abs_mean = np.zeros(p, float)
    np.divide(coef_sum,     np.maximum(coef_n, 1), out=coef_mean,     where=(coef_n > 0))
    np.divide(coef_abs_sum, np.maximum(coef_n, 1), out=coef_abs_mean, where=(coef_n > 0))
    
    names = agg_names if agg_names is not None else [f"f{i}" for i in range(p)]
    
    stab_df = pd.DataFrame({
        "feat_idx":          np.arange(p, dtype=int),
        "feature":           names,
        "fold_select_count": selection_counts,
        "fold_select_freq":  stability_freq,
        "freq_mean":         freq_mean,
        "freq_min":          freq_min,
        "coef_mean":         coef_mean,
        "coef_abs_mean":     coef_abs_mean,
        "coef_n":            coef_n,
    }).sort_values(
        ["freq_mean", "coef_abs_mean", "fold_select_freq"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    
    # ===== Global stable set (NOW based on mean-freq) =====
    stab_stable = stab_df[stab_df["freq_mean"] >= float(GLOBAL_MIN_FREQ)].copy()
    # stab_stable = stab_df[stab_df["fold_select_freq"] >= float(GLOBAL_MIN_FREQ)].copy()
    if GLOBAL_TOP_K is not None:
        stab_stable = stab_stable.head(GLOBAL_TOP_K)
    
    global_stable_idx = stab_stable["feat_idx"].to_numpy(dtype=int)
    global_stable_features = stab_stable["feature"].tolist()
    
    if verbose:
        print(f"\n[GLOBAL STABLE SET by mean freq] {len(global_stable_idx)} features (freq_mean >= {GLOBAL_MIN_FREQ})")
        for n_ in global_stable_features:
            print("  -", n_)

    # Save
    if out_dir is not None and cluster_label is not None and target_label is not None:
        stab_path = os.path.join(out_dir, f"stability_table_{cluster_label}_{target_label}.csv")
        global_path = os.path.join(out_dir, f"global_stable_features_{cluster_label}_{target_label}.csv")
        stab_df.to_csv(stab_path, index=False)
        pd.DataFrame({"feature": global_stable_features}).to_csv(global_path, index=False)
        if verbose:
            print("\nSaved:")
            print(" ", stab_path)
            print(" ", global_path)

    outer_preds = pd.DataFrame({  "left_out_idx": fold_df["left_out_idx"].values,  "y_true": y_true_all,
                                  "y_prob": y_prob_all,  "y_pred": y_pred_all})

    return { "fold_df": fold_df,  "stab_df": stab_df, 
             "outer_metrics": {"AUC": AUC, "ACC": ACC, "BACC": BACC, "F1": F1, "thr_youden_outer": float(thr_youden_outer),
                               "ACC_youden_outer":  ACC_y, "BACC_youden_outer": BACC_y,"F1_youden_outer":   F1_y},
        "outer_preds": outer_preds,  "global_stable_idx": global_stable_idx, "global_stable_features": global_stable_features,
        "per_fold_freq_full": per_fold_freq_full, "per_fold_selected_names": per_fold_selected }



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Cell 7: Evaluation for hyperparam tuning: 3-fold cross-validation

def evaluate_block(X, y, pfer, pi_thr, l1_ratio, C_grid, cv_folds=3, base_seed=42):

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=base_seed)
    fold_baccs = []
    fold_accs = []
    fold_scalers = []
    C_used = []
    fold_metrics = []
    SCALERS = {"standard": StandardScaler(), "robust": RobustScaler()}

    for fold_n, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        stab_res = stability_selection_logreg(
            X_tr, y_tr, C_grid=C_grid, n_subsamples=500, subsample_frac=0.5,
            l1_ratio=l1_ratio, pi_thr=pi_thr, class_weight="balanced",
            rng=base_seed + fold_n, PFER_base=pfer)

        for c in stab_res.get("C_per_run", []):
            if c not in C_used:
                C_used.append(c)

        stable_idx = stab_res["selected_idx"]

        if stable_idx.size == 0:
            fold_baccs.append(np.nan)
            fold_accs.append(np.nan)
            fold_scalers.append("none")
            fold_metrics.append({"fold": fold_n, "scaler": "none", "bacc": np.nan, "acc": np.nan, "features": 0})
            continue

        X_tr_s = X_tr[:, stable_idx]
        X_val_s = X_val[:, stable_idx]
        n_features = int(stable_idx.size)

        best_bacc = -np.inf
        best_acc = np.nan
        best_scaler = None

        for scaler_name, scaler_obj in SCALERS.items():
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", scaler_obj),
                ("clf", LogisticRegression(penalty="none", solver="lbfgs",class_weight="balanced", max_iter=5000))])

            pipe.fit(X_tr_s, y_tr)
            y_pred = pipe.predict(X_val_s)

            bacc = balanced_accuracy_score(y_val, y_pred)
            acc = accuracy_score(y_val, y_pred)

            if bacc > best_bacc:
                best_bacc = bacc
                best_acc = acc
                best_scaler = scaler_name

        fold_baccs.append(best_bacc)
        fold_accs.append(best_acc)
        fold_scalers.append(best_scaler)

        # store ONLY the best-per-fold (prevents later overwrite bugs)
        fold_metrics.append({
            "fold": fold_n,
            "scaler": best_scaler,
            "bacc": float(best_bacc),
            "acc": float(best_acc),
            "features": n_features })

    avg_bacc = float(np.nanmean(fold_baccs))
    avg_acc  = float(np.nanmean(fold_accs))

    return avg_bacc, avg_acc, fold_scalers, C_used, fold_metrics


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
# Cell 8: Tuning call (Run to roughly find how each triplet of (pfer, pi,l1) performas 

PFER_MAX = np.ceil(X_subj.shape[1]/10)   # we set the max range of PFER exploration equal to the 10% of the total number of features rounded up
PFER_GRID     = list(range(1, PFER_MAX))
PI_THR_GRID   = [round(x, 2) for x in np.arange(0.6, 1.0, 0.05)]
L1_RATIO_GRID = [round(x, 2) for x in np.arange(0.3, 1.05, 0.05)]
C_stab_grid   = np.logspace(-5, 3, 30)

n_pfer, n_pi, n_l1 = len(PFER_GRID), len(PI_THR_GRID), len(L1_RATIO_GRID)
CV_FOLDS = 3

bacc_grid = np.full((n_pfer, n_pi, n_l1), np.nan)
acc_grid  = np.full((n_pfer, n_pi, n_l1), np.nan)

fold_bacc_grid = np.full((n_pfer, n_pi, n_l1, CV_FOLDS), np.nan)
fold_acc_grid  = np.full((n_pfer, n_pi, n_l1, CV_FOLDS), np.nan)
fold_feat_grid = np.zeros((n_pfer, n_pi, n_l1, CV_FOLDS), dtype=int)

fold_metrics_dict = {}   # key: (ip,ii,il) -> list of fold dicts
C_values_dict     = {}   # key: (ip,ii,il) -> list
scaler_votes_dict = {}   # key: (ip,ii,il) -> list

jobs = [(ip, ii, il, pfer, pi, l1)
        for ip, pfer in enumerate(PFER_GRID)
        for ii, pi   in enumerate(PI_THR_GRID)
        for il, l1   in enumerate(L1_RATIO_GRID)]

def _run_one_combo(ip, ii, il, pfer, pi, l1):
    avg_bacc, avg_acc, scalers, Cvals, fold_metrics = evaluate_block(
        X_subj, y_labels, pfer=pfer, pi_thr=pi, l1_ratio=l1,
        C_grid=C_stab_grid, cv_folds=CV_FOLDS)
    
    return ip, ii, il, avg_bacc, avg_acc, scalers, Cvals, fold_metrics

results = Parallel(n_jobs=-1, backend="loky", verbose=10, batch_size=1)(delayed(_run_one_combo)(*j) for j in jobs)

for ip, ii, il, avg_bacc, avg_acc, scalers, Cvals, fm_list in results:
    bacc_grid[ip, ii, il] = float(avg_bacc)
    acc_grid[ip,  ii, il] = float(avg_acc)

    key = (ip, ii, il)
    fold_metrics_dict[key] = fm_list
    C_values_dict[key]     = list(Cvals) if Cvals is not None else []
    scaler_votes_dict[key] = list(scalers) if scalers is not None else []

    if fm_list is not None:
        for fm in fm_list:
            f = int(fm.get("fold", -1))
            if 0 <= f < CV_FOLDS:
                fold_bacc_grid[ip, ii, il, f] = float(fm.get("bacc", np.nan))
                fold_acc_grid[ip,  ii, il, f] = float(fm.get("acc",  np.nan))
                fold_feat_grid[ip, ii, il, f] = int(fm.get("features", 0))

if out_dir is not None:
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"stability_grid_{CLUSTER}_WE.npz"),
             bacc_grid=bacc_grid, acc_grid=acc_grid,
             fold_bacc_grid=fold_bacc_grid, fold_acc_grid=fold_acc_grid, fold_feat_grid=fold_feat_grid,
             fold_metrics_dict=fold_metrics_dict,
             PFER_GRID=np.array(PFER_GRID), PI_THR_GRID=np.array(PI_THR_GRID), L1_RATIO_GRID=np.array(L1_RATIO_GRID),
             C_values_dict=C_values_dict, scaler_votes_dict=scaler_votes_dict)



# To load the already calculated tuning data: So if we run the code later, use the pre-saved npz file from previous cell!
out_dir  = "./stability_results/LPFC_WE"
data_path = os.path.join(out_dir,  f"stability_grid_{CLUSTER}_WE.npz")
out_dir  = "./stability_results/LPFC_WE_firstNonePenalty"
data = np.load(data_path, allow_pickle = True)
bacc_grid=data['bacc_grid']
acc_grid=data['acc_grid']
fold_bacc_grid=data['fold_bacc_grid']
fold_acc_grid=data['fold_acc_grid']
fold_feat_grid=data['fold_feat_grid']
PFER_GRID=data['PFER_GRID']
PI_THR_GRID=data['PI_THR_GRID']
L1_RATIO_GRID=data['L1_RATIO_GRID']

# dict-like objects (saved as pickled Python objects)
fold_metrics_dict  = data["fold_metrics_dict"].item()
C_values_dict      = data["C_values_dict"].item()
scaler_votes_dict  = data["scaler_votes_dict"].item()



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
# Cell 9: Visualize the BAcc of the hyperparam grid space to get an idea!
# ---- df_summary from grids ----
rows = []
for ip, pfer in enumerate(PFER_GRID):
    for ii, pi in enumerate(PI_THR_GRID):
        for il, l1 in enumerate(L1_RATIO_GRID):
            b = bacc_grid[ip, ii, il]
            if np.isnan(b): 
                continue
            rows.append((pfer, pi, l1, float(b), float(acc_grid[ip, ii, il])))

df_summary = pd.DataFrame(rows, columns=["pfer","pi_thr","l1_ratio","mean_bacc","mean_acc"])

# ---- 1D mean + IQR bands ----
def compute_iqr(df, var):
    g = df.groupby(var)[["mean_bacc","mean_acc"]]
    out = g.agg(["mean",
                 lambda x: x.quantile(0.25),
                 lambda x: x.quantile(0.75)])
    out.columns = ["_".join([a, b.replace("<lambda_0>", "q25").replace("<lambda_1>", "q75")])
                   for a, b in out.columns]
    return out.reset_index()

df_pfer = compute_iqr(df_summary, "pfer")
df_pi   = compute_iqr(df_summary, "pi_thr")
df_l1   = compute_iqr(df_summary, "l1_ratio")

fig, axes = plt.subplots(1, 3, figsize=(18, 4), dpi=130)

for ax, df_iqr, label in zip(axes, [df_pfer, df_pi, df_l1], ["PFER", "pi_thr", "l1_ratio"]):
    x = df_iqr[label.lower() if label != "PFER" else "pfer"]

    ax.plot(x, df_iqr["mean_bacc_mean"], marker="o", label="BAcc")
    ax.fill_between(x, df_iqr["mean_bacc_q25"], df_iqr["mean_bacc_q75"], alpha=0.2)

    ax.plot(x, df_iqr["mean_acc_mean"], marker="x", label="Acc")
    ax.fill_between(x, df_iqr["mean_acc_q25"], df_iqr["mean_acc_q75"], alpha=0.2)

    ax.set_title(label)
    ax.set_xlabel(label)
    ax.set_ylabel("Score")
    ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(out_dir, "1D_performance_curves_IQR.png"), dpi=200)
plt.close(fig)

# ---- 2D heatmaps (mean over the third variable) ----
pairs = [("pfer","pi_thr"), ("pfer","l1_ratio"), ("pi_thr","l1_ratio")]

def show_heat(ax, piv, title):
    im = ax.imshow(piv.values, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xticks(np.arange(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(piv.shape[0])); ax.set_yticklabels(piv.index)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig, axes = plt.subplots(3, 2, figsize=(14, 14), dpi=130)

for r, (row, col) in enumerate(pairs):
    piv_b = df_summary.pivot_table(index=row, columns=col, values="mean_bacc", aggfunc="mean")
    piv_a = df_summary.pivot_table(index=row, columns=col, values="mean_acc",  aggfunc="mean")
    show_heat(axes[r, 0], piv_b, f"{row} vs {col} (mean BAcc)")
    show_heat(axes[r, 1], piv_a, f"{row} vs {col} (mean Acc)")

fig.tight_layout()
fig.savefig(os.path.join(out_dir, "2D_heatmaps_mean_performance.png"), dpi=200)
plt.close(fig)

# ---- 2D best-BAcc surfaces ----
fig, axes = plt.subplots(3, 1, figsize=(10, 14), dpi=130)

for ax, (row, col) in zip(axes, pairs):
    best = df_summary.groupby([row, col])["mean_bacc"].max().reset_index()
    piv  = best.pivot(index=row, columns=col, values="mean_bacc")
    show_heat(ax, piv, f"{row} vs {col} (best BAcc)")

fig.tight_layout()
# fig.savefig(os.path.join(out_dir, "2D_best_BAcc_surfaces.png"), dpi=200)
# plt.close(fig)


# # **Find the best triplet for generalizability**
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
#  Cell 10: Hierarchical biomarker selection  based on the grid search results (above) 

assert "fold_bacc_grid" in globals() and "fold_feat_grid" in globals()
assert "PFER_GRID" in globals() and "PI_THR_GRID" in globals() and "L1_RATIO_GRID" in globals()

C_values_dict = globals().get("C_values_dict", None)
if C_values_dict is None:
    if "D" in globals() and isinstance(D, np.lib.npyio.NpzFile) and ("C_values_dict" in D.files):
        C_values_dict = D["C_values_dict"].item()
    elif "data" in globals() and isinstance(data, np.lib.npyio.NpzFile) and ("C_values_dict" in data.files):
        C_values_dict = data["C_values_dict"].item()
    else:
        C_values_dict = None

n_pfer, n_pi, n_l1, n_folds = fold_bacc_grid.shape

valid = (fold_feat_grid > 0) & np.isfinite(fold_bacc_grid)
bacc_valid = np.where(valid, fold_bacc_grid, np.nan)

n_valid    = np.sum(np.isfinite(bacc_valid), axis=-1)
mean_bacc  = np.nanmean(bacc_valid, axis=-1)
std_bacc   = np.nanstd(bacc_valid, axis=-1, ddof=1)
se_bacc    = std_bacc / np.sqrt(np.maximum(n_valid, 1))

feat_med   = np.median(fold_feat_grid, axis=-1)
feat_iqr   = np.percentile(fold_feat_grid, 75, axis=-1) - np.percentile(fold_feat_grid, 25, axis=-1)
empty_rate = np.mean(fold_feat_grid == 0, axis=-1)

def _cmin_cmax_from_key(pfer, pi, l1, C_values_dict):
    if C_values_dict is None:
        return (np.nan, np.nan)
    key = (float(pfer), float(pi), float(l1))
    vals = C_values_dict.get(key, None)
    if vals is None:
        return (np.nan, np.nan)
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (np.nan, np.nan)
    return (float(np.min(vals)), float(np.max(vals)))

rows = []
for ip, pfer in enumerate(PFER_GRID):
    for ii, pi in enumerate(PI_THR_GRID):
        for il, l1 in enumerate(L1_RATIO_GRID):
            cmin, cmax = _cmin_cmax_from_key(pfer, pi, l1, C_values_dict)
            rows.append({
                "pfer": float(pfer),
                "pi_thr": float(pi),
                "l1_ratio": float(l1),
                "mean_bacc": float(mean_bacc[ip, ii, il]),
                "se_bacc": float(se_bacc[ip, ii, il]),
                "n_valid_folds": int(n_valid[ip, ii, il]),
                "feat_median": float(feat_med[ip, ii, il]),
                "feat_iqr": float(feat_iqr[ip, ii, il]),
                "empty_rate": float(empty_rate[ip, ii, il]),
                "cmin": cmin,
                "cmax": cmax,})
df = pd.DataFrame(rows)

DELTA_BACC   = 0.01 # We only search the top 1% 
MIN_FEATS    = 1
MAX_FEATS    = 14
REQUIRE_ALLFOLDS = True

EPS_EMPTY = 1e-12
if REQUIRE_ALLFOLDS:
    eligible = df[(df["n_valid_folds"] == n_folds) &
                  (df["empty_rate"] <= EPS_EMPTY) &
                  np.isfinite(df["mean_bacc"])].copy()
else:
    eligible = df[(df["n_valid_folds"] >= max(2, n_folds-1)) &
                  np.isfinite(df["mean_bacc"])].copy()

if eligible.empty:
    raise RuntimeError("No eligible configs (check fold grids / validity definition).")

best_perf_row = eligible.loc[eligible["mean_bacc"].idxmax()].copy()
best_mean = float(best_perf_row["mean_bacc"])

near = eligible[eligible["mean_bacc"] >= (best_mean - DELTA_BACC)].copy()

near_int = near[(near["feat_median"] >= MIN_FEATS) & (near["feat_median"] <= MAX_FEATS)].copy()
if near_int.empty:
    print("No configs met the feature-count window within ΔBAcc; relaxing MIN/MAX feats.")
    near_int = near.copy()

IQR_THR = 2.0
near_stable = near_int[near_int["feat_iqr"] <= IQR_THR].copy()
if near_stable.empty:
    print("No configs met feat_iqr <= 2; relaxing IQR threshold.")
    near_stable = near_int.copy()

near_stable["l1_dist_to_0p6"] = np.abs(near_stable["l1_ratio"] - 0.6)

# # ---- Priority: pi_thr before pfer ----
ranked = near_stable.sort_values(
    ["mean_bacc", "feat_iqr", "feat_median", "pi_thr", "pfer", "l1_dist_to_0p6"],
    ascending=[False,     True,      True,    False,   True,      True]   # pi high, pfer low
).reset_index(drop=True)


# ---- Priority: PFER before pi_thr ----
# ranked = near_stable.sort_values(
#     ["mean_bacc", "feat_iqr", "feat_median", "pfer", "pi_thr", "l1_dist_to_0p6"],
#     ascending=[False, True, True, True, False, True]   # low PFER first, high pi_thr next
# ).reset_index(drop=True)


TOP_K = 15
print(f"Best eligible mean_BAcc = {best_mean:.3f}. Keeping configs within ΔBAcc = {DELTA_BACC:.3f} -> threshold = {(best_mean-DELTA_BACC):.3f}")
print(f"Showing top-{TOP_K} biomarker-first candidates (including cmin/cmax):")
display_cols = ["pfer","pi_thr","l1_ratio","mean_bacc","se_bacc","feat_median","feat_iqr","empty_rate","n_valid_folds","cmin","cmax"]
display(ranked[display_cols].head(TOP_K))



print("\nSelected biomarker-first config:")
print({"pfer": float(best_biomarker["pfer"]), "pi_thr": float(best_biomarker["pi_thr"]), "l1_ratio": float(best_biomarker["l1_ratio"])},
      "| mean_bacc=", f"{float(best_biomarker['mean_bacc']):.3f}",
      "| feat_median=", int(round(float(best_biomarker["feat_median"]))),
      "| feat_iqr=", f"{float(best_biomarker['feat_iqr']):.2f}",
      "| cmin,cmax=", (best_biomarker["cmin"], best_biomarker["cmax"]))

print("\nBest-performance config (eligible, regardless of interpretability window):")
print({"pfer": float(best_perf_row["pfer"]), "pi_thr": float(best_perf_row["pi_thr"]), "l1_ratio": float(best_perf_row["l1_ratio"])},
      "| mean_bacc=", f"{float(best_perf_row['mean_bacc']):.3f}",
      "| feat_median=", int(round(float(best_perf_row["feat_median"]))),
      "| feat_iqr=", f"{float(best_perf_row['feat_iqr']):.2f}",
      "| cmin,cmax=", (best_perf_row["cmin"], best_perf_row["cmax"]))

# tiers = []
# for pfer_val in sorted(near_stable["pfer"].unique()):  # lower PFER first
#     sub = near_stable[near_stable["pfer"] == pfer_val].copy()
#     sub = sub.sort_values(
#         ["mean_bacc", "feat_iqr", "feat_median", "pi_thr", "l1_dist_to_0p6"],
#         ascending=[False, True, True, False, True]      # then higher pi_thr
#     ).head(1)
#     tiers.append(sub)
# tiers_df = pd.concat(tiers, axis=0).sort_values("pfer", ascending=True).reset_index(drop=True)
# display(tiers_df[["pfer","pi_thr","l1_ratio","mean_bacc","feat_median","feat_iqr","cmin","cmax"]])

tiers = []
for pi_val in sorted(near_stable["pi_thr"].unique(), reverse=True):  # higher pi_thr first
    sub = near_stable[near_stable["pi_thr"] == pi_val].copy()
    sub = sub.sort_values(
        ["mean_bacc", "feat_iqr", "feat_median", "pfer", "l1_dist_to_0p6"],
        ascending=[False,     True,      True,    True,      True]   # then lower pfer
    ).head(1)
    tiers.append(sub)

tiers_df = pd.concat(tiers, axis=0).sort_values("pi_thr", ascending=False).reset_index(drop=True)
display(tiers_df[["pfer","pi_thr","l1_ratio","mean_bacc","feat_median","feat_iqr","cmin","cmax"]])

best_biomarker = tiers_df.iloc[0]



# # ***Running unpenalized LOSO to extract the stable features per fold*** 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
# Cell 11: LOSO call 

SCALERS = {"standard": StandardScaler(), "robust": RobustScaler()}

# ---- best triplet from Cell 11 selection ----
pfer_best = int(best_biomarker["pfer"])
pi_best   = float(best_biomarker["pi_thr"])
l1_best   = float(best_biomarker["l1_ratio"])

cmin = float(best_biomarker["cmin"])
cmax =  float(best_biomarker["cmax"])
ppd = 20  # points/decade
lo1, hi1 = np.log10(cmin), np.log10(cmax)
n1 = int(np.clip(np.ceil((hi1 - lo1) * ppd) + 1, 20, 500))
C_STAB = np.logspace(lo1, hi1, n1)

# final C sweep (fixed wide range: here unused since penalty is none)
ppd = 20
# lo2, hi2 = -5, 4
n2 = int(np.clip(np.ceil((hi2 - lo2) * ppd) + 1, 20, 500))
C_FINAL = np.logspace(lo2, hi2, n2)

print(f"best: pfer={pfer_best}, pi_thr={pi_best}, l1_ratio={l1_best} | |C_STAB|={len(C_STAB)}, |C_FINAL|={len(C_FINAL)}")

FINAL_PENALTIES = ('none',)#("l2",) if we set C, for ridge, the we set the penalty to l2

STAB_SEARCH_GRID = {
    "n_subsamples":   [1000],
    "subsample_frac": [0.5],
    "C_grid":         [C_STAB],
    "l1_ratio":       [l1_best],
    "pi_thr":         [pi_best],
    "class_weight":   ["balanced"],
    "verbose":        [False],
    "PFER":           [pfer_best],}

res = run_loso_once(
    X_subj, y_labels, subject_id=subject_ids,
    STAB_SEARCH_GRID=STAB_SEARCH_GRID,
    FINAL_PENALTIES=FINAL_PENALTIES,
    FINAL_C_GRID=C_FINAL,
    GLOBAL_MIN_FREQ=0.7,
    GLOBAL_TOP_K=None,
    INNER_FOLDS=3,   
    SCALERS=SCALERS,
    agg_names=agg_names,        # or None if not available
    out_dir=out_dir,            # or None
    cluster_label=CLUSTER,      # or None
    SEED=42,
    verbose=True )


fold_df    = res["fold_df"]
stab_df    = res["stab_df"]
metrics    = res["outer_metrics"]
glob_idx   = res["global_stable_idx"]
glob_feats = res["global_stable_features"]

print("\nOuter pooled metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.3f}")

print(f"\nNumber of globally stable features (freq_mean >= {GLOBAL_MIN_FREQ}): {len(glob_idx)}")
for name in glob_feats[:10]:
    print("  -", name)


# # Next: the goal is to evaluate the performance of the LR with a given C value, using the selected features per fold (from the loso fold above)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
# Cell 12: Compute the BAcc vs log10(C) curve

def pooled_loso_curves_all(X_subj, y_labels, fold_df, sel_list, agg_names, C_grid, SCALERS, inner_folds=3, seed=42):
    """
    sel_list: per-fold selected features aligned with fold_df rows.
              Either list[list[str]] (feature names) or list[list[int]] (indices).
    Returns:
      xlog, y_true, P_te, THR_nested, bacc05, bacc_posthocY, bacc_nestedY, thr_posthoc
    """
    nC = len(C_grid); N = len(fold_df)

    y_true = np.array([int(y_labels[int(i)]) for i in fold_df["left_out_idx"].values], dtype=int)
    P_te = np.full((N, nC), np.nan, float)
    THR_nested = np.full((N, nC), np.nan, float)

    # map names -> indices only if sel_list stores names
    use_names = (len(sel_list) > 0 and len(sel_list[0]) > 0 and isinstance(sel_list[0][0], str))
    name_to_idx = {nm: i for i, nm in enumerate(agg_names)} if use_names else None

    for j in range(N):
        te_idx = int(fold_df.loc[j, "left_out_idx"])
        scaler_key = fold_df.loc[j, "scaler"]

        sel = np.array([name_to_idx[nm] for nm in sel_list[j]], int) if use_names else np.asarray(sel_list[j], int)

        tr_mask = np.ones(len(y_labels), bool); tr_mask[te_idx] = False
        tr_idx = np.where(tr_mask)[0]

        X_tr = X_subj[tr_idx][:, sel]
        y_tr = y_labels[tr_idx].astype(int)
        X_te = X_subj[[te_idx]][:, sel]

        for iC, C in enumerate(C_grid):
            oof = _oof_probs_with_C(X_tr, y_tr, C=float(C), seed=seed, inner_folds=inner_folds,
                                    scaler_key=scaler_key, penalty="l2", l1_ratio=None)
            THR_nested[j, iC] = youden_threshold_oof(y_tr, oof)

            pipe = Pipeline(steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("sc",  SCALERS[scaler_key]),
                ("clf", LogisticRegression(penalty="l2", solver="lbfgs", C=float(C),
                                           class_weight="balanced", max_iter=5000, tol=1e-4,
                                           warm_start=False, random_state=seed))])
            pipe.fit(X_tr, y_tr)
            P_te[j, iC] = float(pipe.predict_proba(X_te)[:, 1][0])

    bacc05 = np.array([balanced_accuracy_score(y_true, (P_te[:, i] >= 0.5).astype(int)) for i in range(nC)], float)
    thr_posthoc = np.array([youden_threshold_oof(y_true, P_te[:, i]) for i in range(nC)], float)
    bacc_posthocY = np.array([balanced_accuracy_score(y_true, (P_te[:, i] >= thr_posthoc[i]).astype(int))
                              for i in range(nC)], float)
    bacc_nestedY = np.array([balanced_accuracy_score(y_true, (P_te[:, i] >= THR_nested[:, i]).astype(int))
                             for i in range(nC)], float)
    xlog = np.log10(C_grid)
    return xlog, y_true, P_te, THR_nested, bacc05, bacc_posthocY, bacc_nestedY, thr_posthoc


# Inputs
ppd = 20   #setting the resolution
lo2, hi2 = -5, 4  # seting the range
n2 = int(np.clip(np.ceil((hi2 - lo2) * ppd) + 1, 20, 500))

C_grid = np.asarray(np.logspace(lo2, hi2, n2), float)

fold_df  = res["fold_df"]
sel_list = res["per_fold_selected_names"]   # <-- per-fold selected feature NAMES (or indices)

# Defensive alignment check
assert len(sel_list) == len(fold_df), "sel_list and fold_df must have the same number of folds"

# If sel_list accidentally contains indices (happens if agg_names was None in run_loso_once),
# convert to names (recommended) so your name_to_idx mapping works.
if len(sel_list) > 0 and len(sel_list[0]) > 0 and isinstance(sel_list[0][0], (int, np.integer)):
    sel_list = [[agg_names[i] for i in idxs] for idxs in sel_list]

# Compute curves (THIS defines bacc05, bacc_posthocY, bacc_nestedY)
xlog, y_true, P_te, THR_nested, bacc05, bacc_posthocY, bacc_nestedY, thr_posthoc = pooled_loso_curves_all(
    X_subj, y_labels, fold_df, sel_list, agg_names, C_grid, SCALERS,
    inner_folds=3, seed=42
)

iC_rep = int(np.nanargmax(bacc05))
C_rep  = float(C_grid[iC_rep])
p_rep  = P_te[:, iC_rep].astype(float)


# Plot the BAcc vs log10(C)
fig, ax = plt.subplots()
ax.plot(xlog, bacc05,        linewidth=3.0, alpha=1.00, color="orange", label="BAcc (thr=0.5)")
ax.plot(xlog, bacc_posthocY, linewidth=2.2, alpha=0.75, color="blue",   label="BAcc (post-hoc Youden)")
ax.plot(xlog, bacc_nestedY,  linewidth=2.0, alpha=0.75, color="green",  label="BAcc (nested Youden)")

ax.set_xlabel("log10(C)",fontsize=22, fontweight='bold')
ax.set_ylabel("BAcc", fontsize=22,fontweight='bold')
# ax.set_title("Pooled LOSO performance vs C_final")
xmin = int(np.floor(np.nanmin(xlog)))
xmax = int(np.ceil(np.nanmax(xlog)))
ax.set_xticks(np.arange(xmin, xmax + 1))
ax.set_ylim(0.6, 0.9)
ax.tick_params(axis="both", which="both", direction="in", labelsize=18)
ax.legend(frameon=False, fontsize=16, loc="best")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, f"BAcc_vs_C_{CLUSTER}.png"), dpi=300, bbox_inches="tight")
plt.show()



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
# Cell 13: Summary of the best results for the selected triplet 

C_FINAL = np.logspace(lo2, hi2, n2)
C_grid = np.asarray(C_FINAL, float)
b = np.asarray(bacc05, float)

DELTA = 0.01
b_star = float(np.nanmax(b))
good = np.isfinite(b) & (b >= (b_star - DELTA))

if not np.any(good):
    raise RuntimeError("No C values in the ΔBAcc plateau; decrease DELTA.")

C_good = C_grid[good]
C_lo, C_hi = float(np.min(C_good)), float(np.max(C_good))
C_rep = float(np.min(C_good))   # conservative

print("Best pooled LOSO BAcc (thr=0.5):", b_star)
print(f"C plateau (Δ={DELTA}): [{C_lo:.3g}, {C_hi:.3g}]")
print("Representative C for coefficient reporting:", C_rep)

auc = np.array([ roc_auc_score(y_true, P_te[:, i]) for i in range(P_te.shape[1]) ], float)

auc_star = float(np.nanmax(auc))
auc_plateau_max = float(np.nanmax(auc[good]))

print("\nAUC (pooled LOSO):")
print("  Best AUC (any C):", auc_star)
print("  Best AUC within BAcc plateau:", auc_plateau_max)


# Post-hoc Youden (optimistic)
bacc_posthocY_star = float(np.nanmax(bacc_posthocY))
bacc_posthocY_plateau = float(np.nanmax(bacc_posthocY[good]))

# Nested Youden (unbiased, noisy)
bacc_nestedY_star = float(np.nanmax(bacc_nestedY))
bacc_nestedY_plateau = float(np.nanmax(bacc_nestedY[good]))

print("\nYouden BAcc (pooled LOSO):")
print("  Post-hoc Youden  | best any C:", bacc_posthocY_star)
print("  Post-hoc Youden  | best in plateau:", bacc_posthocY_plateau)
print("  Nested Youden    | best any C:", bacc_nestedY_star)
print("  Nested Youden    | best in plateau:", bacc_nestedY_plateau)


tag  = f"{CLUSTER}_WE"
base = os.path.join(out_dir, tag)

# preserve list columns (optional but useful)
fold_df.to_pickle(base + "_fold_df.pkl")
stab_df.to_pickle(base + "_stab_df.pkl")

# plateau-based C_rep from pooled LOSO BAcc@0.5
DELTA  = 0.01
b      = np.asarray(bacc05, float)
b_star = float(np.nanmax(b))

good = np.isfinite(b) & (b >= (b_star - DELTA))
if not np.any(good):
    raise RuntimeError("No C values in ΔBAcc plateau; decrease DELTA.")

C_good = np.asarray(C_grid, float)[good]
C_lo, C_hi = float(C_good.min()), float(C_good.max())

C_rep  = float(C_lo)                                 
iC_rep = int(np.argmin(np.abs(C_grid - C_rep)))
p_rep  = np.asarray(P_te[:, iC_rep], float)

print(f"BAcc@0.5 best={b_star:.3f} | plateau=[{C_lo:.3g},{C_hi:.3g}] | C_rep={C_rep:.3g} | BAcc@C_rep={bacc05[iC_rep]:.3f}")

np.savez_compressed(base + "_fusionpack_thr05.npz", left_out_idx=fold_df["left_out_idx"].to_numpy(int), y_true=np.asarray(y_true, int),
    p_rep=p_rep,C_rep=C_rep, iC_rep=iC_rep, C_lo=C_lo,C_hi=C_hi, b_star=b_star, b_at_Crep=float(bacc05[iC_rep]),)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
# FROM HERE ON EVERYTHING IS VISUALIZATION AND WE ALREADY HAVE ALL THE RESULTS
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
#  Cell 14 : Subject-bootstrap CIs + ROC mean(IQR) + Youden operating point (pooled LOSO) 
# Choose C for ROC/CI cell
i_auc_best_any = int(np.nanargmax(auc))
C_auc_best_any = float(C_grid[i_auc_best_any])

# Best AUC restricted to the BAcc plateau
i_auc_best_plateau = int(np.nanargmax(np.where(good, auc, -np.inf)))
C_auc_best_plateau = float(C_grid[i_auc_best_plateau])

print("C_auc_best_any:", C_auc_best_any, "AUC:", float(auc[i_auc_best_any]))
print("C_auc_best_plateau:", C_auc_best_plateau, "AUC:", float(auc[i_auc_best_plateau]))
# C_USE_FOR_ROC = C_auc_best_plateau   # recommended
C_USE_FOR_ROC = C_auc_best_any     # i max AUC

# ---------- Use pooled LOSO predictions at a specific C ----------
C_grid = np.asarray(C_grid, float)          # the same grid used for P_te / auc
xlog = np.log10(C_grid)

# pick the column matching C_USE_FOR_ROC
i_use = int(np.argmin(np.abs(C_grid - float(C_USE_FOR_ROC))))
C_used_exact = float(C_grid[i_use])

# pooled LOSO labels/probs at that C
y_true = np.asarray(y_true, dtype=int)
y_prob = np.asarray(P_te[:, i_use], dtype=float)

print(f"ROC/CI computed at C={C_used_exact:.6g} (index {i_use}/{len(C_grid)})")
print("AUC check:", roc_auc_score(y_true, y_prob))

# Build df in the same format your bootstrap expects
df = pd.DataFrame({"left_out_idx": fold_df["left_out_idx"].to_numpy(int),"y_true": y_true,"y_prob": y_prob})

# ----------  Choose operating point ----------
# post-hoc pooled Youden at this C (this is what your ROC point is)
THR = float(youden_threshold_oof(y_true, y_prob))

def sens_spec(y_true_, y_pred_):
    tn, fp, fn, tp = confusion_matrix(y_true_, y_pred_, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return sens, spec

def metrics_at_thr(y_true_, y_prob_, thr=0.5):
    y_pred_ = (y_prob_ >= thr).astype(int)
    auc  = roc_auc_score(y_true_, y_prob_)
    bacc = balanced_accuracy_score(y_true_, y_pred_)
    sens, spec = sens_spec(y_true_, y_pred_)
    return auc, bacc, sens, spec

auc_hat, bacc_hat, sens_hat, spec_hat = metrics_at_thr(y_true, y_prob, thr=0.5)
fpr_youden = 1.0 - spec_hat
tpr_youden = sens_hat

# ----------  Subject-bootstrap (stratified by class) ----------
N_BOOT = 1000
SEED = 42
rng = np.random.default_rng(SEED)

subj_y = df.groupby("left_out_idx")["y_true"].first()
pos_subj = subj_y[subj_y == 1].index.to_numpy()
neg_subj = subj_y[subj_y == 0].index.to_numpy()
n_pos, n_neg = pos_subj.size, neg_subj.size

FPR_GRID = np.linspace(0.0, 1.0, 201)
boot_auc, boot_bacc, boot_sens, boot_spec = [], [], [], []
boot_tpr_mat = []

for _ in range(N_BOOT):
    samp_pos = rng.choice(pos_subj, size=n_pos, replace=True)
    samp_neg = rng.choice(neg_subj, size=n_neg, replace=True)
    samp = np.concatenate([samp_pos, samp_neg])

    boot_df = pd.concat([df[df["left_out_idx"] == s] for s in samp], ignore_index=True)
    yb = boot_df["y_true"].to_numpy(int)
    pb = boot_df["y_prob"].to_numpy(float)

    a, b_, se, sp = metrics_at_thr(yb, pb, THR)
    boot_auc.append(a); boot_bacc.append(b_); boot_sens.append(se); boot_spec.append(sp)

    fpr_b, tpr_b, _ = roc_curve(yb, pb, drop_intermediate=False)
    tpr_i = np.interp(FPR_GRID, fpr_b, tpr_b)
    tpr_i = np.maximum.accumulate(tpr_i)
    tpr_i = np.clip(tpr_i, 0.0, 1.0)
    boot_tpr_mat.append(tpr_i)

boot_auc  = np.asarray(boot_auc, float)
boot_bacc = np.asarray(boot_bacc, float)
boot_sens = np.asarray(boot_sens, float)
boot_spec = np.asarray(boot_spec, float)
boot_tpr_mat = np.asarray(boot_tpr_mat, float)

def ci95(x):
    return (np.percentile(x, 2.5), np.percentile(x, 97.5))

auc_ci  = ci95(boot_auc)
bacc_ci = ci95(boot_bacc)
sens_ci = ci95(boot_sens)
spec_ci = ci95(boot_spec)

tpr_med = np.percentile(boot_tpr_mat, 50, axis=0)
tpr_q25 = np.percentile(boot_tpr_mat, 25, axis=0)
tpr_q75 = np.percentile(boot_tpr_mat, 75, axis=0)

# ----------  Summary table ----------
summary = pd.DataFrame({
    "metric": ["AUC", "BalAcc@THR", "Sensitivity@THR", "Specificity@THR", "THR (Youden pooled-LOSO at fixed C)", "C_used"],
    "point":  [auc_hat, bacc_hat, sens_hat, spec_hat, THR, C_used_exact],
    "ci_low": [auc_ci[0], bacc_ci[0], sens_ci[0], spec_ci[0], np.nan, np.nan],
    "ci_high":[auc_ci[1], bacc_ci[1], sens_ci[1], spec_ci[1], np.nan, np.nan],
})
display(summary)

print(f"THR = {THR:.8f} | Youden point: (FPR={fpr_youden:.3f}, TPR={tpr_youden:.3f})")

# ---------- Plot ROC median + IQR + Youden point ----------
FS_TITLE = 12
FS_LABEL = 14
FS_TICK  = 13
FS_LEG   = 12
BOLD = "bold"

fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=120)
ax.grid(False)

ax.plot(FPR_GRID, tpr_med, linewidth=2, label="Bootstrap median ROC")
ax.fill_between(FPR_GRID, tpr_q25, tpr_q75, alpha=0.25, label="IQR (25–75%)")
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
ax.scatter([fpr_youden], [tpr_youden], s=60, label=f"Youden THR={THR:.5f}")

ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=FS_LABEL, fontweight=BOLD)
ax.set_ylabel("True Positive Rate (Sensitivity)",      fontsize=FS_LABEL, fontweight=BOLD)
ax.set_title(
    f"LPFC with ExSEnt | AUC={auc_hat:.3f} [{auc_ci[0]:.3f},{auc_ci[1]:.3f}]",
    fontsize=FS_TITLE, fontweight=BOLD)

ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ticks: size + bold
ax.tick_params(axis="both", labelsize=FS_TICK)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontweight(BOLD)

leg = ax.legend(frameon=False, fontsize=FS_LEG)
for t in leg.get_texts():
    t.set_fontweight(BOLD)

fig.tight_layout()
fig_name = f"ROC_fixedC_{CLUSTER}.png"
fig.savefig(os.path.join(out_dir, fig_name), dpi=300, bbox_inches="tight")
plt.show()



# # **Visualization and reporting of the results**
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
# Cell 15 : Top stable features + final-fit coef at fixed C 

best_params = best_biomarker
def make_pretty_label(feat):
    f = str(feat)
    f = f.replace("powermean_Theta/powermean_Alpha", "TAR")
    f = f.replace("powermean", r"$\hat{P}$")
    f = f.replace("_|", "|")
    return f

# Choose the fixed C setting
# Use the same C as reported coefficients/ROC: C_rep (from plateau) or C_USE_FOR_ROC (best AUC within plateau)
C_use = float(C_rep)

# choose scaler for final refit (mode across folds)
scaler_mode = fold_df["scaler"].value_counts().idxmax()

# Nested stability table (from LOSO run)
stab_df = res["stab_df"].copy()
p = stab_df.shape[0]
pi_thr = float(best_params["pi_thr"])

if (2 * pi_thr - 1) > 0:
    n_top = max(1, int(np.sqrt((2 * pi_thr - 1) * p)))
else:
    n_top = 7

print(f"Showing top {n_top} features (p={p}, pi_thr={pi_thr:.2f})")

# Final refit at fixed C to get coef_final for the SAME features
# Choose which feature set to refit:
# (A) all features in stab_df (could be large)
# (B) globally stable set only (recommended!)
USE_GLOBAL_STABLE_ONLY = True
if USE_GLOBAL_STABLE_ONLY:
    feat_idx = np.asarray(res["global_stable_idx"], dtype=int)
else:
    feat_idx = np.arange(len(agg_names), dtype=int)

feat_names = [agg_names[i] for i in feat_idx]
X = X_subj[:, feat_idx]
y = y_labels.astype(int)

final_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  SCALERS[scaler_mode]),
    ("clf", LogisticRegression(
        penalty="l2", solver="lbfgs", C=C_use,
        class_weight="balanced", max_iter=5000, tol=1e-4,
        random_state=42))
])
final_pipe.fit(X, y)
coef = final_pipe.named_steps["clf"].coef_.ravel()

coef_final_df = pd.DataFrame({
    "feature": feat_names,
    "coef_final": coef,
    "abs_coef_final": np.abs(coef)})


# Merge nested stability + fixed-C coefficients
stab_use = stab_df.copy()
if USE_GLOBAL_STABLE_ONLY:
    stab_use = stab_use[stab_use["feat_idx"].isin(feat_idx)].copy()

stab_sorted = stab_use.sort_values(
    ["freq_mean", "coef_abs_mean", "fold_select_freq"],
    ascending=[False, False, False])

top_df = stab_sorted.head(n_top).copy()
top_df = top_df.merge(coef_final_df, on="feature", how="left")

# Pretty labels
top_df["Feature"] = [make_pretty_label(f) for f in top_df["feature"]]

# Keep both stability notions + nested coef summaries + final-fit coef at fixed C
top_df = top_df[[
    "Feature",
    "freq_mean",
    "freq_min",
    "fold_select_freq",
    "fold_select_count",
    "coef_mean",
    "coef_abs_mean",
    "coef_final",
    "abs_coef_final",
]]

# Rename for readability
top_df = top_df.rename(columns={
    "freq_mean":         r"StabFreq$_{mean}$",
    "freq_min":          r"StabFreq$_{min}$",
    "fold_select_freq":  r"LOSO SelectFreq",
    "fold_select_count": r"LOSO SelectCount",
    "coef_mean":         r"Coef$_{nested}$ (mean)",
    "coef_abs_mean":     r"|Coef$_{nested}$| (abs mean)",
    "coef_final":        r"Coef$_{final}$ (C fixed)",
    "abs_coef_final":    r"|Coef$_{final}$|",
})

# Round numeric columns
num_cols = [c for c in top_df.columns if c != "Feature"]
top_df[num_cols] = top_df[num_cols].astype(float).round(3)

# Display + export LaTeX
caption_txt = (
    f"<b>Top {n_top} Stable Features — {CLUSTER}</b><br>"
    f"(PFER={best_params['pfer']:.0f}, "
    f"π<sub>thr</sub>={best_params['pi_thr']:.2f}, "
    f"L1 ∈ {best_params['l1_ratio']})<br>"
    f"Fixed-C coefficients computed from final refit using scaler={scaler_mode}, C={C_use:.3g}.<br>"
    f"StabFreq: mean/min subsample-selection frequency within LOSO training folds; "
    f"LOSO SelectFreq: fraction of LOSO folds where the feature was selected.")

styled = (
    top_df.style
    .set_table_attributes('style="font-size:14px; border-collapse:collapse;"')
    .set_caption(caption_txt)
    .set_properties(**{
        "font-weight": "bold",
        "text-align": "center",
        "border": "1px solid black",
        "padding": "4px"}))

display(HTML(styled.to_html()))
tex_path = os.path.join(out_dir, f"top_features_{CLUSTER}_Cfixed.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(top_df.to_latex(index=False, escape=False))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
# ======= Cell 16: Publication plots & tables  ========
# Output directory (assumed defined earlier as out_dir)
OUT_DIR = globals().get("out_dir", ".")
os.makedirs(OUT_DIR, exist_ok=True)

# Get outputs from run_loso_once
loso_res   = res
fold_df    = loso_res["fold_df"]
stab_df    = loso_res["stab_df"]
outer_preds = loso_res["outer_preds"]   # contains left_out_idx, y_true (and old y_prob/y_pred)

# labels
USE_BOOTSTRAP   = True
N_BOOT          = 1500
CLASS_NAMES     = globals().get("CLASS_NAMES", {0: "Healthy", 1: "Dementia"})
CLUSTER_LABEL   = str(globals().get("CLUSTER", "cluster"))
TARGET_LABEL    = str(globals().get("TARGET", "target"))
TITLE_PREFIX    = f"{CLUSTER_LABEL} | {TARGET_LABEL}"
 
# FIXED-C setting: best AUC within BAcc plateau
C_use = float(C_auc_best_plateau)

# P_te is the pooled LOSO probabilities vs C from the sweep: shape: (N_subjects, nC)
C_grid = np.asarray(C_grid, float)
i_use = int(np.argmin(np.abs(C_grid - C_use)))
C_used_exact = float(C_grid[i_use])

# ----- Prepare arrays (FIXED C) -----
# Keep y_true from outer_preds (same LOSO subject order as the sweep)
y_true = outer_preds["y_true"].to_numpy(dtype=int)

# Use fixed-C probabilities from the sweep
y_prob = np.asarray(P_te[:, i_use], float)

# Primary decision at 0.5
y_pred = (y_prob >= 0.5).astype(int)

outer_preds = outer_preds.copy()
outer_preds["y_true"] = y_true
outer_preds["y_prob"] = y_prob
outer_preds["y_pred"] = y_pred

mask = np.isfinite(y_true) & np.isfinite(y_prob) & np.isfinite(y_pred)
has_binary = (mask.sum() >= 2) and (np.unique(y_true[mask]).size == 2)

print(f"[Cell16] Fixed-C reporting: requested C={C_use:.6g} | using C={C_used_exact:.6g} | n={mask.sum()}")
print(f"[Cell16] AUC sanity (fixed C): {roc_auc_score(y_true[mask], y_prob[mask]):.6f}")


# ------------------------- Helpers -------------------------
def _bootstrap_curve_stats(y, s, *, n_boot=1000, seed=42, grid=None, mode="roc"):
    rng = np.random.RandomState(seed)
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    if grid is None:
        grid = np.linspace(0., 1., 101)
    collects = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)  # resample subjects with replacement
        yy = y[idx]; ss = s[idx]
        if np.unique(yy).size < 2:
            continue
        if mode == "roc":
            fpr, tpr, _ = roc_curve(yy, ss)
            fpr_u, idx_u = np.unique(fpr, return_index=True)
            tpr_u = tpr[idx_u]
            collects.append(np.interp(grid, fpr_u, tpr_u))
        elif mode == "pr":
            prec, rec, _ = precision_recall_curve(yy, ss)
            rec_u, idx_u = np.unique(rec, return_index=True)
            prec_u = prec[idx_u]
            collects.append(np.interp(grid, rec_u, prec_u))
        else:
            raise ValueError("mode must be 'roc' or 'pr'")
    if len(collects) == 0:
        return grid, None, None, None
    M = np.vstack(collects)
    mean = np.nanmean(M, axis=0)
    q25  = np.nanpercentile(M, 25, axis=0)
    q75  = np.nanpercentile(M, 75, axis=0)
    return grid, mean, q25, q75

def _summ_stats(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(mean=np.nan, median=np.nan, iqr=np.nan, min=np.nan, max=np.nan)
    q25, q75 = np.percentile(x, [25, 75])
    return dict(mean=float(np.mean(x)), median=float(np.median(x)),
                iqr=float(q75 - q25), min=float(np.min(x)), max=float(np.max(x)))

# Text summary block (outer unbiased + fold means) =====
summary_outer = {}
if has_binary:
    y_t = y_true[mask]
    y_p = y_prob[mask]
    y_hat = (y_p >= 0.5).astype(int)   # instead of y_pred[mask]
    
    # Pooled AUC and AP
    summary_outer["AUC_outer"] = roc_auc_score(y_t, y_p)
    summary_outer["AP_outer"]  = average_precision_score(y_t, y_p)

    # --- Metrics at fixed threshold 0.5 (primary, matches y_pred) ---
    summary_outer["ACC_thr05"] = accuracy_score(y_t, y_hat)
    tn, fp, fn, tp = confusion_matrix(y_t, y_hat).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    summary_outer["BACC_thr05"] = 0.5 * (sens + spec)

    # For convenience: alias as ACC_outer / BACC_outer (used later)
    summary_outer["ACC_outer"]  = summary_outer["ACC_thr05"]
    summary_outer["BACC_outer"] = summary_outer["BACC_thr05"]

    # --- Metrics at Youden threshold on pooled outer scores (post-hoc, optimistic) ---
    thr_youden_outer = youden_threshold_oof(y_t, y_p)
    y_pred_youden    = (y_p >= thr_youden_outer).astype(int)
    summary_outer["thr_youden_outer"] = float(thr_youden_outer)
    summary_outer["ACC_youden_outer"] = accuracy_score(y_t, y_pred_youden)

    tn_y, fp_y, fn_y, tp_y = confusion_matrix(y_t, y_pred_youden).ravel()
    sens_y = tp_y / max(tp_y + fn_y, 1)
    spec_y = tn_y / max(tn_y + fp_y, 1)
    summary_outer["BACC_youden_outer"] = 0.5 * (sens_y + spec_y)

    summary_outer["n_subjects"] = int(mask.sum())
else:
    summary_outer = {
        "AUC_outer": np.nan,
        "AP_outer":  np.nan,
        "ACC_thr05": np.nan,
        "BACC_thr05": np.nan,
        "ACC_outer": np.nan,
        "BACC_outer": np.nan,
        "thr_youden_outer": np.nan,
        "ACC_youden_outer": np.nan,
        "BACC_youden_outer": np.nan,
        "n_subjects": int(mask.sum()),
    }

# Per-fold means 
means_fold = {}
for col in ["auc_train","auc_test","bacc_train","bacc_test",
            "logloss_train","logloss_test"]:
    if col in fold_df.columns:
        means_fold[col] = float(np.nanmean(fold_df[col]))
    else:
        means_fold[col] = np.nan

print("=== Outer LOSO (pooled subjects, thr=0.5) ===")
for k in ["AUC_outer","AP_outer","ACC_outer","BACC_outer","n_subjects"]:
    val = summary_outer.get(k, np.nan)
    if isinstance(val, (int, float)):
        if isinstance(val, int) and k == "n_subjects":
            print(f"{k}: {val}")
        else:
            print(f"{k}: {val:.3f}")
    else:
        print(f"{k}: {val}")

print("\n=== Outer LOSO (pooled, Youden threshold) ===")
for k in ["thr_youden_outer","ACC_youden_outer","BACC_youden_outer"]:
    val = summary_outer.get(k, np.nan)
    if isinstance(val, (int, float)):
        print(f"{k}: {val:.3f}")
    else:
        print(f"{k}: {val}")

print("\n=== Per-fold means ===")
for k,v in means_fold.items():
    if isinstance(v, float) and not np.isnan(v):
        print(f"{k}: {v:.3f}")
    else:
        print(f"{k}: {v}")

# ===== 1) ROC curve (outer pooled) with optional subject-bootstrap mean + IQR =====
fig, ax = plt.subplots(figsize=(6.0, 4.8), dpi=120)
if has_binary:
    fpr, tpr, _ = roc_curve(y_true[mask], y_prob[mask])
    auc_val = roc_auc_score(y_true[mask], y_prob[mask])

    if USE_BOOTSTRAP:
        grid, mean_tpr, tpr_q25, tpr_q75 = _bootstrap_curve_stats(
            y_true[mask], y_prob[mask],
            n_boot=N_BOOT, seed=int(globals().get("SEED", 42)),
            grid=np.linspace(0,1,201), mode="roc"
        )
        if mean_tpr is not None:
            ax.fill_between(grid, tpr_q25, tpr_q75, alpha=0.2,
                            label="IQR band (subject bootstrap)")
            ax.plot(grid, mean_tpr, lw=2, label="Mean ROC (subject bootstrap)")

    ax.plot(fpr, tpr, lw=1, alpha=0.8, label=f"Pooled ROC (AUC={auc_val:.3f})")
    ax.plot([0,1],[0,1], "--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{TITLE_PREFIX} | ROC (outer LOSO pooled)")
    ax.legend(loc="lower right")
else:
    ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, f"roc_outer_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
            dpi=300, bbox_inches="tight")
if _CAN_DISPLAY: plt.show()
else: plt.close(fig)

# ===== 2) Precision–Recall curve (outer pooled) with optional subject-bootstrap mean + IQR =====
fig, ax = plt.subplots(figsize=(6.0, 4.8), dpi=120)
if has_binary:
    precision, recall, _ = precision_recall_curve(y_true[mask], y_prob[mask])
    ap = average_precision_score(y_true[mask], y_prob[mask])

    if USE_BOOTSTRAP:
        grid, mean_prec, prec_q25, prec_q75 = _bootstrap_curve_stats(
            y_true[mask], y_prob[mask],
            n_boot=N_BOOT, seed=int(globals().get("SEED", 42)),
            grid=np.linspace(0,1,201), mode="pr"
        )
        if mean_prec is not None:
            ax.fill_between(grid, prec_q25, prec_q75, alpha=0.2,
                            label="IQR band (subject bootstrap)")
            ax.plot(grid, mean_prec, lw=2, label="Mean PR (subject bootstrap)")

    ax.plot(recall, precision, lw=1, alpha=0.8, label=f"Pooled PR (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{TITLE_PREFIX} | Precision–Recall (outer LOSO pooled)")
    ax.legend(loc="lower left")
else:
    ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, f"pr_outer_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
            dpi=300, bbox_inches="tight")
if _CAN_DISPLAY: plt.show()
else: plt.close(fig)

# ===== 3) Confusion matrix (outer pooled decision) =====
if has_binary:
    labels = [0, 1]
    cm = confusion_matrix(y_true[mask], y_pred[mask], labels=labels)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    FS_TITLE = 8
    FS_LABEL = 14
    FS_TICK  = 12
    FS_CBAR  = 14
    FS_CELL  = 24  # numbers inside cells
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=120)
    ax.grid(False)
    im = ax.imshow(cm_norm, cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(
        [CLASS_NAMES.get(0, "Class 0"), CLASS_NAMES.get(1, "Class 1")],
        fontsize=FS_TICK
    )
    ax.set_yticklabels(
        [CLASS_NAMES.get(0, "Class 0"), CLASS_NAMES.get(1, "Class 1")],
        fontsize=FS_TICK)
    ax.set_xlabel("Predicted", fontsize=FS_LABEL)
    ax.set_ylabel("True", fontsize=FS_LABEL)
    ax.set_title(f"{TITLE_PREFIX} | Confusion Matrix (outer LOSO pooled)", fontsize=FS_TITLE)

    ax.tick_params(axis="both", labelsize=FS_TICK)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                fontsize=FS_CELL, fontweight="bold",
                color="white" if cm_norm[i, j] > 0.5 else "black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized", fontsize=FS_CBAR)
    cbar.ax.tick_params(labelsize=FS_TICK)
    cbar.ax.yaxis.get_offset_text().set_size(FS_TICK)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"cm_outer_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),dpi=300, bbox_inches="tight")
    if _CAN_DISPLAY:
        plt.show()
    else:
        plt.close(fig)


if has_binary:
    labels = [0, 1]
    thr_y = float(summary_outer["thr_youden_outer"])
    y_pred_y = (y_prob[mask] >= thr_y).astype(int)

    cm_y = confusion_matrix(y_true[mask], y_pred_y, labels=labels)
    cm_norm_y = cm_y.astype(float) / np.clip(cm_y.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=120)
    ax.grid(False)
    im = ax.imshow(cm_norm_y, cmap="Purples", vmin=0, vmax=1)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([CLASS_NAMES.get(0,"Class 0"), CLASS_NAMES.get(1,"Class 1")], fontsize=FS_TICK)
    ax.set_yticklabels([CLASS_NAMES.get(0,"Class 0"), CLASS_NAMES.get(1,"Class 1")], fontsize=FS_TICK)

    ax.set_xlabel("Predicted", fontsize=FS_LABEL)
    ax.set_ylabel("True", fontsize=FS_LABEL)
    ax.set_title(f"{TITLE_PREFIX} | Confusion Matrix (Youden thr={thr_y:.3f})", fontsize=FS_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_y[i,j]}", ha="center", va="center",
                    fontsize=FS_CELL, fontweight="bold",
                    color="white" if cm_norm_y[i,j] > 0.5 else "black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized", fontsize=FS_CBAR)
    cbar.ax.tick_params(labelsize=FS_TICK)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"cm_outer_youden_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
                dpi=300, bbox_inches="tight")
    if _CAN_DISPLAY: plt.show()
    else: plt.close(fig)



# ===== 4) Per-fold AUC (train vs test) =====
if {"auc_train","fold"}.issubset(fold_df.columns):
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=120)
    ax.plot(fold_df["fold"], fold_df["auc_train"], marker="o", label="Train AUC (OOF)")
    if "auc_test" in fold_df.columns and not fold_df["auc_test"].isna().all():
        ax.plot(fold_df["fold"], fold_df["auc_test"], marker="o", label="Test AUC (held-out)")
    ax.set_xlabel("Outer fold"); ax.set_ylabel("AUC"); ax.set_ylim(0,1.05)
    ax.set_title(f"{TITLE_PREFIX} | Per-fold AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"per_fold_auc_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
                dpi=300, bbox_inches="tight")
    if _CAN_DISPLAY: plt.show()
    else: plt.close(fig)

# ===== 5) Per-fold #selected features =====
if {"fold","n_selected"}.issubset(fold_df.columns):
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=120)
    ax.bar(fold_df["fold"], fold_df["n_selected"])
    ax.set_xlabel("Outer fold"); ax.set_ylabel("# selected features")
    ax.set_title(f"{TITLE_PREFIX} | Selected features per fold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"per_fold_n_features_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
                dpi=300, bbox_inches="tight")
    if _CAN_DISPLAY: plt.show()
    else: plt.close(fig)

# ===== 6) Feature stability histogram + top-k barh =====
fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=120)
ax.hist(stab_df["fold_select_freq"].to_numpy(), bins=20, edgecolor="black")
ax.set_xlabel("Fold selection frequency"); ax.set_ylabel("Number of features")
ax.set_title(f"{TITLE_PREFIX} | Stability frequency histogram")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, f"stab_hist_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
            dpi=300, bbox_inches="tight")
if _CAN_DISPLAY: plt.show()
else: plt.close(fig)

TOP_K = min(20, stab_df.shape[0])
top_df_plot = stab_df.head(TOP_K)[["feature","fold_select_freq"]].iloc[::-1]
fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=120)
ax.barh(top_df_plot["feature"], top_df_plot["fold_select_freq"])
ax.set_xlabel("Fold selection frequency")
ax.set_title(f"{TITLE_PREFIX} | Top-{TOP_K} stable features")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, f"top{TOP_K}_features_barh_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
            dpi=300, bbox_inches="tight")
if _CAN_DISPLAY: plt.show()
else: plt.close(fig)

# ===== 7) Accuracy and per-class recall per fold =====
per_fold = []
if "left_out_idx" in fold_df.columns and "left_out_idx" in outer_preds.columns:
    for _, row in fold_df.iterrows():
        idx = int(row["left_out_idx"])
        m = (outer_preds["left_out_idx"] == idx)
        if m.any():
            yt = int(outer_preds.loc[m, "y_true"].iloc[0])
            yp = int(outer_preds.loc[m, "y_pred"].iloc[0])
            acc = float(yt == yp)
            sens = 1.0 if (yt == 1 and yp == 1) else (0.0 if yt == 1 else np.nan)
            spec = 1.0 if (yt == 0 and yp == 0) else (0.0 if yt == 0 else np.nan)
            per_fold.append({"fold": int(row["fold"]), "acc": acc, "sens": sens, "spec": spec})
per_fold_df = pd.DataFrame(per_fold).sort_values("fold")

if not per_fold_df.empty:
    # Accuracy per fold
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=120)
    ax.plot(per_fold_df["fold"], per_fold_df["acc"], marker="o")
    ax.set_xlabel("Outer fold"); ax.set_ylabel("Accuracy"); ax.set_ylim(-0.05,1.05)
    ax.set_title(f"{TITLE_PREFIX} | Accuracy per fold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"per_fold_acc_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
                dpi=300, bbox_inches="tight")
    if _CAN_DISPLAY: plt.show()
    else: plt.close(fig)

    # Sensitivity & specificity per fold
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=120)
    ax.plot(per_fold_df["fold"], per_fold_df["sens"], marker="o", label="Sensitivity (Recall 1)")
    ax.plot(per_fold_df["fold"], per_fold_df["spec"], marker="o", label="Specificity (Recall 0)")
    ax.set_xlabel("Outer fold"); ax.set_ylabel("Recall"); ax.set_ylim(-0.05,1.05)
    ax.set_title(f"{TITLE_PREFIX} | Per-class recall per fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"per_fold_recall_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
                dpi=300, bbox_inches="tight")
    if _CAN_DISPLAY: plt.show()
    else: plt.close(fig)

# ===== 8) Per-fold log-loss (train OOF vs test) =====
if {"fold","logloss_train","logloss_test"}.issubset(fold_df.columns):
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=120)
    ax.plot(fold_df["fold"], fold_df["logloss_train"], marker="o", label="Train LogLoss (OOF)")
    ax.plot(fold_df["fold"], fold_df["logloss_test"],  marker="o", label="Test LogLoss (held-out)")
    ax.set_xlabel("Outer fold"); ax.set_ylabel("Log loss")
    ax.set_title(f"{TITLE_PREFIX} | Per-fold LogLoss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"per_fold_logloss_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
                dpi=300, bbox_inches="tight")
    if _CAN_DISPLAY: plt.show()
    else: plt.close(fig)

# ===== 9) PFER reporting =====
if "pfer" in fold_df.columns:
    pfer_stats = _summ_stats(fold_df["pfer"])
    pfer_df = pd.DataFrame([{
        "PFER_mean":   pfer_stats["mean"],
        "PFER_median": pfer_stats["median"],
        "PFER_IQR":    pfer_stats["iqr"],
        "PFER_min":    pfer_stats["min"],
        "PFER_max":    pfer_stats["max"],
    }])
    print("=== PFER summary across outer folds (Meinshausen–Bühlmann bound) ===")
    if _CAN_DISPLAY:
        display(pfer_df)
    pfer_df.to_csv(os.path.join(OUT_DIR, f"pfer_summary_{CLUSTER_LABEL}_{TARGET_LABEL}.csv"),
                   index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=120)
    vals = np.asarray(fold_df["pfer"], float)
    vals = vals[np.isfinite(vals)]
    if vals.size:
        ax.hist(vals, bins=min(20, max(5, int(np.sqrt(vals.size)))), edgecolor="black")
        ax.set_xlabel("PFER bound"); ax.set_ylabel("# folds")
        ax.set_title(f"{TITLE_PREFIX} | PFER bound (per fold)")
    else:
        ax.text(0.5, 0.5, "No finite PFER values", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"pfer_hist_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
                dpi=300, bbox_inches="tight")
    if _CAN_DISPLAY: plt.show()
    else: plt.close(fig)

# ===== 10) Compact overall summary table =====
overall = {
    "AUC_outer":  summary_outer.get("AUC_outer", np.nan),
    "AP_outer":   summary_outer.get("AP_outer", np.nan),
    "ACC_outer":  summary_outer.get("ACC_outer", np.nan),
    "BACC_outer": summary_outer.get("BACC_outer", np.nan),
    "thr_youden_outer": summary_outer.get("thr_youden_outer", np.nan),
    "ACC_youden_outer": summary_outer.get("ACC_youden_outer", np.nan),
    "BACC_youden_outer": summary_outer.get("BACC_youden_outer", np.nan),
    "mean_auc_train":  means_fold.get("auc_train", np.nan),
    "mean_auc_test":   means_fold.get("auc_test", np.nan),
    "mean_bacc_train": means_fold.get("bacc_train", np.nan),
    "mean_bacc_test":  means_fold.get("bacc_test", np.nan),
    "mean_logloss_train": means_fold.get("logloss_train", np.nan),
    "mean_logloss_test":  means_fold.get("logloss_test", np.nan),
    "n_subjects": summary_outer.get("n_subjects", np.nan), }
overall_df = pd.DataFrame([overall])
print("=== Overall summary ===")
if _CAN_DISPLAY:
    display(overall_df)
overall_df.to_csv(os.path.join(OUT_DIR, f"overall_summary_{CLUSTER_LABEL}_{TARGET_LABEL}.csv"),
                  index=False)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
# Cell 17: Features plot (FIXED-C coefficients) — order: freq primary, |coef| tie-break 

# Output dir (same convention as above)
OUT_DIR = globals().get("out_dir", ".")
os.makedirs(OUT_DIR, exist_ok=True)

# Try to detect if we can display
try:
    from IPython.display import display  # noqa
    _CAN_DISPLAY = True
except Exception:
    _CAN_DISPLAY = False

CLUSTER_LABEL = str(globals().get("CLUSTER", "cluster"))
TARGET_LABEL  = str(globals().get("TARGET", "target"))

def make_pretty_label(feat):
    f = str(feat)
    # --- Replace TAR and S̄ ---
    f = f.replace("powermean_Theta/powermean_Alpha", "TAR")
    f = f.replace("signalmean", r"$\overline{S}$")
    f = f.replace("_|", "|")
    return f
 
# Required inputs
assert "C_used_exact" in globals() and np.isfinite(C_used_exact), "Define C_used_exact first (selected fixed C)."
assert "SCALERS" in globals() and isinstance(SCALERS, dict), "SCALERS must exist."
assert "fold_df" in globals() and "scaler" in fold_df.columns, "fold_df['scaler'] must exist."
assert "agg_names" in globals() and agg_names is not None, "agg_names required."
assert "stab_df" in globals(), "stab_df required."
assert "X_subj" in globals() and "y_labels" in globals(), "X_subj and y_labels required."

# pick scaler for final interpretability (same convention you used elsewhere)
scaler_mode = fold_df["scaler"].value_counts().idxmax()

name_to_idx = {nm: i for i, nm in enumerate(agg_names)}
y = y_labels.astype(int)

def _fixedC_coefs_for_features(feature_list):
    sel_idx = np.array([name_to_idx[f] for f in feature_list], dtype=int)
    X = X_subj[:, sel_idx]

    final_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  SCALERS[scaler_mode]),
        ("clf", LogisticRegression(
            penalty="l2", solver="lbfgs", C=float(C_used_exact),
            class_weight="balanced", max_iter=5000, tol=1e-4,
            warm_start=False, random_state=42))
    ])
    final_pipe.fit(X, y)
    c = final_pipe.named_steps["clf"].coef_.ravel()
    return c

def _make_top_df(freq_col, TOP_K):
    # pick TOP_K by frequency first (same as before, but explicit)
    df = (stab_df
          .sort_values(freq_col, ascending=False)
          .head(TOP_K)[["feature", freq_col]]
          .copy())

    # compute fixed-C coefficients for these TOP_K features
    coef_fixedC = _fixedC_coefs_for_features(df["feature"].tolist())
    df["coef_mean"] = coef_fixedC
    df["coef_abs_mean"] = np.abs(coef_fixedC)

    # ensure plotting code can stay unchanged (uses 'fold_select_freq')
    df["fold_select_freq"] = df[freq_col].to_numpy(float)

    # ORDER: frequency primary, |coef| secondary
    df = df.sort_values([freq_col, "coef_abs_mean"], ascending=[False, False]).copy()

    # keep only columns the plot expects
    df = df[["feature", "fold_select_freq", "coef_mean", "coef_abs_mean"]]

    # keep your original convention (largest appears at top in barh)
    return df.iloc[::-1].reset_index(drop=True)

# ---- build both top_dfs ----
TOP_K = min(20, stab_df.shape[0])

top_df_final = _make_top_df("fold_select_freq", TOP_K)  # "final fit" frequency proxy
top_df_stab  = _make_top_df("freq_mean", TOP_K)         # stability-selection frequency (mean)

# ====================================================================
# PLOT 1: FINAL-FIT frequency proxy (fold_select_freq) — UNCHANGED FIG
# ====================================================================
top_df = top_df_final

# Pretty labels for y-axis
x_labels = [make_pretty_label(f) for f in top_df["feature"]]

coef_vals = top_df["coef_mean"].to_numpy()   # signed coefficients
abs_vals  = np.abs(coef_vals)

vmax_raw = float(np.nanmax(abs_vals)) if np.isfinite(np.nanmax(abs_vals)) else 0.0
if vmax_raw == 0.0:
    vmax_raw = 1.0

cmap = cm.get_cmap("Purples")
norm = colors.Normalize(vmin=0.0, vmax=vmax_raw)   # raw scale
cols = cmap(norm(abs_vals))

# ---- Plot ----
fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
bars = ax.barh(
    x_labels,
    top_df["fold_select_freq"],
    color=cols,
    edgecolor="black",
    linewidth=0.8)

ax.set_xlabel("Fold selection frequency", fontsize=16, fontweight="bold")
ax.set_title(f"{CLUSTER_LABEL} | {TARGET_LABEL} | Top-{TOP_K} stable features",
             fontsize=18, fontweight="bold")
ax.grid(axis="x", alpha=0.2)
ax.grid(axis="y", which="both", visible=False)  # remove horizontal gridlines
ax.set_xlim(0, 1.05)
# --- Hatch overlay for negative coefficients ---
for bar, c in zip(bars, coef_vals):
    if c < 0:
        ax.barh(
            y=bar.get_y() + bar.get_height() / 2,
            width=bar.get_width(),
            height=bar.get_height(),
            left=bar.get_x(),
            align="center",
            facecolor="none",
            edgecolor="orange",
            linewidth=0.0,
            hatch="//",)

# Colorbar for |coef| normalized 0–1
sm = cm.ScalarMappable(cmap=cmap)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("|coef|", fontsize=16, fontweight="bold")
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.ax.tick_params(labelsize=14)
for tick in cb.ax.get_yticklabels():
    tick.set_fontweight("bold")

# Legend for sign meaning
legend_patches = [
    Patch(facecolor=cmap(0.7), edgecolor="black", label="Positive coef"),
    Patch(facecolor=cmap(0.8), edgecolor="orange", hatch="//", label="Negative coef"),
]
# ax.legend(handles=legend_patches, loc="lower right", frameon=False)
ax.legend(handles=legend_patches, loc="lower right", frameon=False, fontsize=14)

# Axis formatting
ax.tick_params(axis='x', labelsize=16)
for ticklabel in ax.get_xticklabels():
    ticklabel.set_fontweight("bold")

ax.tick_params(axis='y', labelsize=12)
for ticklabel in ax.get_yticklabels():
    ticklabel.set_fontweight("bold")

fig.tight_layout()

# ---- Save and (optionally) display ----
fig_path = os.path.join(OUT_DIR, f"top{TOP_K}_features_fancy_FINALFREQ_{CLUSTER_LABEL}_{TARGET_LABEL}.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Saved feature plot to: {fig_path}")

if _CAN_DISPLAY:
    plt.show()
else:
    plt.close(fig)

# ====================================================================
# PLOT 2: STABILITY frequency (freq_mean) — SAME FIG SETTINGS
# ====================================================================
top_df = top_df_stab

# Pretty labels for y-axis
x_labels = [make_pretty_label(f) for f in top_df["feature"]]

coef_vals = top_df["coef_mean"].to_numpy()   # signed coefficients
abs_vals  = np.abs(coef_vals)

vmax_raw = float(np.nanmax(abs_vals)) if np.isfinite(np.nanmax(abs_vals)) else 0.0
if vmax_raw == 0.0:
    vmax_raw = 1.0

cmap = cm.get_cmap("Purples")
norm = colors.Normalize(vmin=0.0, vmax=vmax_raw)   # raw scale
cols = cmap(norm(abs_vals))

# ---- Plot ----
fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
bars = ax.barh(
    x_labels,
    top_df["fold_select_freq"],
    color=cols,
    edgecolor="black",
    linewidth=0.8
)

ax.set_xlabel("Fold selection frequency", fontsize=16, fontweight="bold")
ax.set_title(f"{CLUSTER_LABEL} | {TARGET_LABEL} | Top-{TOP_K} stable features",
             fontsize=18, fontweight="bold")
ax.grid(axis="x", alpha=0.2)
ax.grid(axis="y", which="both", visible=False)  # remove horizontal gridlines
ax.set_xlim(0, 1.05)
# --- Hatch overlay for negative coefficients ---
for bar, c in zip(bars, coef_vals):
    if c < 0:
        ax.barh(
            y=bar.get_y() + bar.get_height() / 2,
            width=bar.get_width(),
            height=bar.get_height(),
            left=bar.get_x(),
            align="center",
            facecolor="none",
            edgecolor="orange",
            linewidth=0.0,
            hatch="//",)

# Colorbar for |coef| normalized 0–1
sm = cm.ScalarMappable(cmap=cmap)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("|coef|", fontsize=16, fontweight="bold")
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.ax.tick_params(labelsize=14)
for tick in cb.ax.get_yticklabels():
    tick.set_fontweight("bold")

# Legend for sign meaning
legend_patches = [
    Patch(facecolor=cmap(0.7), edgecolor="black", label="Positive coef"),
    Patch(facecolor=cmap(0.8), edgecolor="orange", hatch="//", label="Negative coef"),
]
# ax.legend(handles=legend_patches, loc="lower right", frameon=False)
ax.legend(handles=legend_patches, loc="lower right", frameon=False, fontsize=14)

# Axis formatting
ax.tick_params(axis='x', labelsize=16)
for ticklabel in ax.get_xticklabels():
    ticklabel.set_fontweight("bold")

ax.tick_params(axis='y', labelsize=12)
for ticklabel in ax.get_yticklabels():
    ticklabel.set_fontweight("bold")
fig.tight_layout()

# ---- Save and  display ----
fig_path = os.path.join(OUT_DIR, f"top{TOP_K}_features_fancy_STABFREQ_{CLUSTER_LABEL}_{TARGET_LABEL}.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Saved feature plot to: {fig_path}")

if _CAN_DISPLAY:
    plt.show()
else:
    plt.close(fig)



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
#  Cell 18: misclassified subjects table via subject_id join  
assert "P_te" in globals()
assert "mf" in globals(), "Need mf (path to the cohort .mat file) in scope."
assert "C_grid" in globals() and "b" in globals() and "C_rep" in globals()
assert "out_dir" in globals() and "CLUSTER" in globals()
assert "subject_id" in globals(), "Need subject_id from Cell 3 (the LOSO group ids)."

# y_true aligned to P_te
if "y_labels" in globals():
    y_true_vec = np.asarray(y_labels, int)
elif "y_true" in globals():
    y_true_vec = np.asarray(y_true, int)
else:
    raise RuntimeError("Need y_labels (preferred) or y_true in scope.")

n = int(P_te.shape[0])
if y_true_vec.shape[0] != n:
    raise ValueError(f"y_true length {y_true_vec.shape[0]} != P_te rows {n}")

# ---- helper: coerce ids to integer keys (handles 1.0 vs 1) ----
def _to_int_key(x):
    a = pd.to_numeric(pd.Series(np.asarray(x).reshape(-1)), errors="coerce")
    # round if essentially integer
    frac = np.abs(a - np.round(a))
    if np.nanmax(frac.to_numpy()) < 1e-6:
        return np.round(a).astype("Int64")
    # fallback: still cast (will set non-finite to <NA>)
    return a.astype("Int64")

# ---- load full cohort meta from MAT (always length N subjects in file) ----
dd = loadmat(mf, squeeze_me=True, struct_as_record=False)

sid_full    = np.asarray(dd.get("subject_id", np.nan)).reshape(-1)
group_full  = np.asarray(dd.get("group_str",  np.nan)).reshape(-1).astype(object)
age_full    = np.asarray(dd.get("age",        np.nan), dtype=float).reshape(-1)
gender_full = np.asarray(dd.get("gender_raw", np.nan)).reshape(-1).astype(object)
mmse_full   = np.asarray(dd.get("ymmse",      np.nan), dtype=float).reshape(-1)

meta_full = pd.DataFrame({
    "subject_id_raw": sid_full,
    "group": group_full,
    "age": age_full,
    "gender": gender_full,
    "mmse": mmse_full,
})
meta_full["subject_id_int"] = _to_int_key(meta_full["subject_id_raw"].to_numpy())

# ---- pick C* and C_rep indices ----
C_grid = np.asarray(C_grid, float)
b = np.asarray(b, float)

i_star = int(np.nanargmax(b))
C_star = float(C_grid[i_star])

i_rep = int(np.nanargmin(np.abs(C_grid - float(C_rep))))
C_rep_used = float(C_grid[i_rep])

p1_rep = np.asarray(P_te[:, i_rep], float)   # P(class=1|x) at C_rep_used
THR = 0.5
yhat_rep = (p1_rep >= THR).astype(int)
mis = (yhat_rep != y_true_vec)

# ---- build prediction df keyed by subject_id ----
pred = pd.DataFrame({
    "subject_id_raw": np.asarray(subject_id).reshape(-1),
    "y_true": y_true_vec,
    "yhat": yhat_rep,
    "p1": p1_rep,
})
pred["subject_id_int"] = _to_int_key(pred["subject_id_raw"].to_numpy())

# ---- merge predictions with meta by subject_id_int ----
df = pred.merge(
    meta_full.drop(columns=["subject_id_raw"]),
    on="subject_id_int",
    how="left",
    validate="many_to_one",)

# If some subjects did not match, show them (usually means ID mismatch)
n_missing = int(df["group"].isna().sum())
if n_missing > 0:
    missing_ids = df.loc[df["group"].isna(), "subject_id_raw"].tolist()
    print(f"WARNING: {n_missing}/{n} subjects did not match metadata by subject_id. Examples:", missing_ids[:10])

# ---- label strings (optional) ----
if "CLASS_NAMES" in globals() and isinstance(CLASS_NAMES, dict):
    lab = lambda z: CLASS_NAMES.get(int(z), str(int(z)))
else:
    lab = lambda z: str(int(z))

df["true_class"] = [lab(z) for z in df["y_true"].to_numpy()]
df["pred_class"] = [lab(z) for z in df["yhat"].to_numpy()]
df["error_type"] = np.where(
    (df["y_true"] == 1) & (df["yhat"] == 0), "FN",
    np.where((df["y_true"] == 0) & (df["yhat"] == 1), "FP", "OK"))

# ---- final misclassified table (requested columns) ----
df_mis = df.loc[mis, [
    "subject_id_raw", "group", "age", "gender", "mmse", "true_class", "pred_class", "p1", "error_type"
]].copy()

df_mis = df_mis.rename(columns={
    "subject_id_raw": "subject_id",
    "p1": "pred_prob_class1",})

# Sort: FN/FP then confidence
df_mis["|p-0.5|"] = np.abs(df_mis["pred_prob_class1"] - 0.5)
df_mis = df_mis.sort_values(["error_type", "|p-0.5|"], ascending=[True, False]).drop(columns=["|p-0.5|"])
df_mis = df_mis.reset_index(drop=True)

# Save CSV
tag = f"{CLUSTER}_misclassified_thr{THR:g}_Crep{C_rep_used:.3g}_Cstar{C_star:.3g}"
csv_path = os.path.join(out_dir, f"{tag}.csv")
df_mis.to_csv(csv_path, index=False)

print(f"C_rep (used on grid): {C_rep_used:.6g} | C*: {C_star:.6g}")
print(f"Misclassified at C_rep (thr={THR}): {df_mis.shape[0]} / {n}")
display(df_mis)
print("Saved:", csv_path)


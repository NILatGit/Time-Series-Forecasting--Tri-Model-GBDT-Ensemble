"""
OrderVolume Prediction — Clean & Precise
Focused on CV-LB alignment: fewer features, multi-fold CV, simple average.
"""
import os, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

warnings.filterwarnings('ignore')

SEED  = 42
SEEDS = [42, 123, 456, 789, 2025]
SHIFT = 42  # test horizon — all lags must be >= this

# Auto-detect GPU
try:
    import torch; HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False
CB_TASK = 'GPU' if HAS_GPU else 'CPU'
XGB_DEV = 'cuda' if HAS_GPU else 'cpu'

# ── 1. Load Data ────────────────────────────────────────────
DATA_DIR = '/kaggle/input/competitions/ch-27-celebal-technologies-jadavour-university'
if not os.path.exists(DATA_DIR):
    DATA_DIR = './'

train = pd.read_csv(os.path.join(DATA_DIR, 'orders_train.csv'))
test  = pd.read_csv(os.path.join(DATA_DIR, 'orders_test.csv'))
meta  = pd.read_csv(os.path.join(DATA_DIR, 'hub_metadata.csv'))
train['Date'] = pd.to_datetime(train['Date'])
test['Date']  = pd.to_datetime(test['Date'])
print(f"Train: {train.shape}  Test: {test.shape}")

# ── 2. Clean & Merge Metadata ──────────────────────────────
meta['CompetitorDistance']       = meta['CompetitorDistance'].fillna(meta['CompetitorDistance'].median())
meta['CompetitorOpenSinceMonth'] = meta['CompetitorOpenSinceMonth'].fillna(0).astype(int)
meta['CompetitorOpenSinceYear']  = meta['CompetitorOpenSinceYear'].fillna(0).astype(int)
meta['LoyaltyProgramSinceYear']  = meta['LoyaltyProgramSinceYear'].fillna(0).astype(int)
meta['LoyaltyProgramSinceWeek']  = meta['LoyaltyProgramSinceWeek'].fillna(0).astype(int)
meta['LoyaltyProgramInterval']   = meta['LoyaltyProgramInterval'].fillna('None')

train = train.merge(meta, on='HubID', how='left')
test  = test.merge(meta, on='HubID', how='left')

# ── 3. Feature Engineering (focused, high-signal only) ─────
INTERVAL_MONTHS = {
    'Jan,Apr,Jul,Oct':  {1, 4, 7, 10},
    'Feb,May,Aug,Nov':  {2, 5, 8, 11},
    'Mar,Jun,Sept,Dec': {3, 6, 9, 12},   # note: Sept not Sep
}

for df in [train, test]:
    df.sort_values(['HubID', 'Date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Calendar
    df['Year']      = df['Date'].dt.year
    df['Month']     = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['sin_dow']   = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['cos_dow']   = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['sin_doy']   = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['cos_doy']   = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)

    # Competitor
    df['LogCompDist'] = np.log1p(df['CompetitorDistance'])
    df['CompOpenMonths'] = (
        12 * (df['Year'] - df['CompetitorOpenSinceYear']) +
        (df['Month'] - df['CompetitorOpenSinceMonth'])
    ).clip(lower=0)
    df.loc[df['CompetitorOpenSinceYear'] == 0, 'CompOpenMonths'] = 0
    df['CompIsOpen'] = (df['CompOpenMonths'] > 0).astype(int)

    # Loyalty (vectorized — fixes the Sept/Sep bug in original code)
    df['IsLoyaltyActive'] = 0
    for interval, months in INTERVAL_MONTHS.items():
        mask = (
            (df['LoyaltyProgram'] == 1) &
            (df['LoyaltyProgramInterval'] == interval) &
            (df['Month'].isin(months)) &
            (df['Year'] >= df['LoyaltyProgramSinceYear'])
        )
        df.loc[mask, 'IsLoyaltyActive'] = 1

    # Promo context
    df['PromoPrev'] = df.groupby('HubID')['PromoActive'].shift(1).fillna(0).astype(int)
    df['PromoNext'] = df.groupby('HubID')['PromoActive'].shift(-1).fillna(0).astype(int)
    df['Promo_DoW'] = df['PromoActive'] * 10 + df['DayOfWeek']

print("✓ Base features built")

# ── 4. Lag Features (on concat, shift ≥ 42) ────────────────
train['is_test'] = 0;  test['is_test'] = 1
test['OrderVolume'] = np.nan

full = pd.concat([train, test], ignore_index=True) \
       .sort_values(['HubID', 'Date']).reset_index(drop=True)
full['LogVol'] = np.log1p(full['OrderVolume'])

for s in [42, 49, 56, 63]:
    full[f'Lag_{s}'] = full.groupby('HubID')['LogVol'].shift(s)

full['RollMean14_L42'] = full.groupby('HubID')['Lag_42'].transform(
    lambda x: x.rolling(14, min_periods=1).mean())
full['RollMean28_L42'] = full.groupby('HubID')['Lag_42'].transform(
    lambda x: x.rolling(28, min_periods=1).mean())
full['LagDiff_42_49'] = full['Lag_42'] - full['Lag_49']

train_df = full[full['is_test'] == 0].copy()
test_df  = full[full['is_test'] == 1].copy()
del full

lag_cols = ['Lag_42', 'Lag_49', 'Lag_56', 'Lag_63',
            'RollMean14_L42', 'RollMean28_L42', 'LagDiff_42_49']

print("✓ Lag features built")

# ── 5. Target Encoding Helper ──────────────────────────────
def add_hub_encodings(source_df, target_df, global_mean, K=30):
    """
    Compute Bayesian-smoothed hub-level means from source_df,
    map them onto target_df. No merge — index-safe.
    """
    src = source_df[(source_df['IsOpen'] == 1) & (source_df['OrderVolume'] > 0)]

    # Hub mean
    h = src.groupby('HubID')['LogVol'].agg(['mean', 'count'])
    a = h['count'] / (h['count'] + K)
    hub_map = (a * h['mean'] + (1 - a) * global_mean).to_dict()

    # Hub × DayOfWeek mean
    hd = src.groupby(['HubID', 'DayOfWeek'])['LogVol'].agg(['mean', 'count'])
    a = hd['count'] / (hd['count'] + K)
    dow_map = (a * hd['mean'] + (1 - a) * global_mean).to_dict()

    out = target_df.copy()
    out['HubMeanLV']    = out['HubID'].map(hub_map).fillna(global_mean)
    keys = list(zip(out['HubID'], out['DayOfWeek']))
    out['HubDoWMeanLV'] = [dow_map.get(k, global_mean) for k in keys]
    return out


GLOBAL_MEAN = train_df[
    (train_df['IsOpen'] == 1) & (train_df['OrderVolume'] > 0)
]['LogVol'].mean()

# ── 6. Feature & Category Lists ────────────────────────────
features = [
    # Hub identity & metadata
    'HubID', 'HubFormat', 'AssortmentTier',
    'CompetitorDistance', 'LogCompDist', 'CompOpenMonths', 'CompIsOpen',
    'LoyaltyProgram', 'IsLoyaltyActive',
    # Calendar
    'DayOfWeek', 'Month', 'Year', 'DayOfYear',
    'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy',
    # Operational
    'PromoActive', 'PromoPrev', 'PromoNext', 'Promo_DoW',
    'RegionalHoliday', 'SchoolClosureFlag',
    # Lag (all ≥ 42, safe for test horizon)
    *lag_cols,
    # Target encoding
    'HubMeanLV', 'HubDoWMeanLV',
]
cat_cols = ['HubFormat', 'AssortmentTier', 'DayOfWeek', 'Month', 'Promo_DoW']

print(f"✓ {len(features)} features selected")


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(np.clip(y_pred, 0, None)) - np.log1p(y_true)) ** 2))


# ── 7. 3-Fold Time-Series CV ───────────────────────────────
# WHY 3 folds + simple average fixes CV-LB gap:
#   • Multiple folds average out period-specific noise
#   • No scipy weight optimization = no overfitting to val signal
#   • Fewer features = less overfitting to spurious patterns

print("\n" + "=" * 60)
print("3-Fold Time-Series CV  (42-day validation windows)")
print("=" * 60)

max_date = train_df['Date'].max()
folds = []
for i in range(3):
    ve = max_date - pd.Timedelta(days=SHIFT * i)
    vs = ve - pd.Timedelta(days=SHIFT - 1)
    te = vs - pd.Timedelta(days=1)
    folds.append((te, vs, ve))
folds.reverse()

mask_ok = (train_df['IsOpen'] == 1) & (train_df['OrderVolume'] > 0)

cv = {'LGB': [], 'CB': [], 'XGB': [], 'AVG': []}
best_iters = {'LGB': [], 'CB': [], 'XGB': []}

for fi, (tr_end, vs, ve) in enumerate(folds):
    print(f"\n─── Fold {fi+1}: train ≤ {tr_end.date()},  val {vs.date()} → {ve.date()} ───")

    tr_m = (train_df['Date'] <= tr_end) & mask_ok & train_df['Lag_42'].notna()
    va_m = (train_df['Date'] >= vs) & (train_df['Date'] <= ve) & mask_ok & train_df['Lag_42'].notna()

    tr = train_df[tr_m].copy()
    va = train_df[va_m].copy()

    # Target encodings from fold-train only (no leakage)
    tr = add_hub_encodings(tr, tr, GLOBAL_MEAN)
    va = add_hub_encodings(tr, va, GLOBAL_MEAN)

    for c in lag_cols:
        tr[c] = tr[c].fillna(GLOBAL_MEAN)
        va[c] = va[c].fillna(GLOBAL_MEAN)

    X_tr, y_tr = tr[features], tr['LogVol']
    X_va, y_va = va[features], va['LogVol']
    y_true = va['OrderVolume'].values

    # ─ LightGBM ─
    Xt, Xv = X_tr.copy(), X_va.copy()
    for c in cat_cols:
        Xt[c] = Xt[c].astype('category')
        Xv[c] = Xv[c].astype('category')

    dt = lgb.Dataset(Xt, y_tr, categorical_feature=cat_cols)
    dv = lgb.Dataset(Xv, y_va, categorical_feature=cat_cols, reference=dt)
    m = lgb.train(
        {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
         'learning_rate': 0.02, 'num_leaves': 127, 'max_depth': 10,
         'feature_fraction': 0.7, 'bagging_fraction': 0.8, 'bagging_freq': 1,
         'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
         'verbose': -1, 'n_jobs': -1, 'random_state': SEED},
        dt, 3000, valid_sets=[dv],
        callbacks=[lgb.early_stopping(200, verbose=False)])
    p_lgb = np.expm1(m.predict(Xv))
    best_iters['LGB'].append(m.best_iteration)
    s = rmsle(y_true, p_lgb); cv['LGB'].append(s)
    print(f"  LGB : {s:.5f}  (iter {m.best_iteration})")

    # ─ CatBoost ─
    Xt, Xv = X_tr.copy(), X_va.copy()
    for c in cat_cols:
        Xt[c] = Xt[c].astype(str)
        Xv[c] = Xv[c].astype(str)

    m = cb.CatBoostRegressor(
        iterations=3000, learning_rate=0.02, depth=8,
        l2_leaf_reg=3.0, random_strength=0.5,
        task_type=CB_TASK, verbose=0, random_seed=SEED)
    m.fit(Xt, y_tr, cat_features=cat_cols,
          eval_set=(Xv, y_va), early_stopping_rounds=200, verbose=0)
    p_cb = np.expm1(m.predict(Xv))
    best_iters['CB'].append(m.best_iteration_)
    s = rmsle(y_true, p_cb); cv['CB'].append(s)
    print(f"  CB  : {s:.5f}  (iter {m.best_iteration_})")

    # ─ XGBoost ─
    Xt, Xv = X_tr.copy(), X_va.copy()
    for c in cat_cols:
        Xt[c] = Xt[c].astype('category')
        Xv[c] = Xv[c].astype('category')

    m = xgb.XGBRegressor(
        n_estimators=3000, learning_rate=0.02, max_depth=9,
        colsample_bytree=0.7, subsample=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        tree_method='hist', device=XGB_DEV, enable_categorical=True,
        verbosity=0, random_state=SEED, early_stopping_rounds=200)
    m.fit(Xt, y_tr, eval_set=[(Xv, y_va)], verbose=0)
    p_xgb = np.expm1(m.predict(Xv))
    best_iters['XGB'].append(m.best_iteration)
    s = rmsle(y_true, p_xgb); cv['XGB'].append(s)
    print(f"  XGB : {s:.5f}  (iter {m.best_iteration})")

    # ─ Simple Average (no weight optimization!) ─
    p_avg = (p_lgb + p_cb + p_xgb) / 3.0
    s = rmsle(y_true, p_avg); cv['AVG'].append(s)
    print(f"  AVG : {s:.5f}")

print("\n" + "=" * 60)
print("CV SUMMARY  (mean ± std)")
for name, scores in cv.items():
    print(f"  {name:4s}  RMSLE = {np.mean(scores):.5f} ± {np.std(scores):.5f}")
print("=" * 60)

# Final boost rounds = median best iteration × 1.05 (conservative)
final_rounds = {}
for name, iters in best_iters.items():
    final_rounds[name] = min(int(np.median(iters) * 1.05), 3000)
    print(f"  {name} final rounds: {final_rounds[name]}  (median CV best: {int(np.median(iters))})")


# ── 8. Full Training (5 seeds × 3 models) ──────────────────
print("\n" + "=" * 60)
print("Full Training  →  5 seeds × 3 models")
print("=" * 60)

# Apply target encodings from full training data
train_df = add_hub_encodings(train_df, train_df, GLOBAL_MEAN)
test_df  = add_hub_encodings(train_df, test_df,  GLOBAL_MEAN)

for c in lag_cols:
    train_df[c] = train_df[c].fillna(GLOBAL_MEAN)
    test_df[c]  = test_df[c].fillna(GLOBAL_MEAN)

fm = (train_df['IsOpen'] == 1) & (train_df['OrderVolume'] > 0) & train_df['Lag_42'].notna()
X_all = train_df[fm][features].copy()
y_all = train_df[fm]['LogVol']
X_tst = test_df[features].copy()

N = len(SEEDS)

# ─ LightGBM ×5 seeds ─
X_al, X_tl = X_all.copy(), X_tst.copy()
for c in cat_cols:
    X_al[c] = X_al[c].astype('category')
    X_tl[c] = X_tl[c].astype('category')

print("  LGB: ", end="", flush=True)
preds_lgb = np.zeros(len(X_tst))
for s in SEEDS:
    d = lgb.Dataset(X_al, y_all, categorical_feature=cat_cols)
    m = lgb.train(
        {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
         'learning_rate': 0.02, 'num_leaves': 127, 'max_depth': 10,
         'feature_fraction': 0.7, 'bagging_fraction': 0.8, 'bagging_freq': 1,
         'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
         'verbose': -1, 'n_jobs': -1, 'random_state': s},
        d, final_rounds['LGB'])
    preds_lgb += np.expm1(m.predict(X_tl)) / N
    print(f"s{s}✓ ", end="", flush=True)
print()

# ─ CatBoost ×5 seeds ─
X_ac, X_tc = X_all.copy(), X_tst.copy()
for c in cat_cols:
    X_ac[c] = X_ac[c].astype(str)
    X_tc[c] = X_tc[c].astype(str)

print("  CB : ", end="", flush=True)
preds_cb = np.zeros(len(X_tst))
for s in SEEDS:
    m = cb.CatBoostRegressor(
        iterations=final_rounds['CB'], learning_rate=0.02, depth=8,
        l2_leaf_reg=3.0, random_strength=0.5,
        task_type=CB_TASK, verbose=0, random_seed=s)
    m.fit(X_ac, y_all, cat_features=cat_cols)
    preds_cb += np.expm1(m.predict(X_tc)) / N
    print(f"s{s}✓ ", end="", flush=True)
print()

# ─ XGBoost ×5 seeds ─
X_ax, X_tx = X_all.copy(), X_tst.copy()
for c in cat_cols:
    X_ax[c] = X_ax[c].astype('category')
    X_tx[c] = X_tx[c].astype('category')

print("  XGB: ", end="", flush=True)
preds_xgb = np.zeros(len(X_tst))
for s in SEEDS:
    m = xgb.XGBRegressor(
        n_estimators=final_rounds['XGB'], learning_rate=0.02, max_depth=9,
        colsample_bytree=0.7, subsample=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        tree_method='hist', device=XGB_DEV, enable_categorical=True,
        verbosity=0, random_state=s)
    m.fit(X_ax, y_all)
    preds_xgb += np.expm1(m.predict(X_tx)) / N
    print(f"s{s}✓ ", end="", flush=True)
print()

# ── 9. Ensemble & Post-Processing ──────────────────────────
# Simple 1/3 average — no scipy optimization (avoids overfitting weights)
final = (preds_lgb + preds_cb + preds_xgb) / 3.0
final[test_df['IsOpen'].values == 0] = 0.0
final = np.clip(final, 0, None)

# ── 10. Save ────────────────────────────────────────────────
sub = pd.DataFrame({'Id': test_df['Id'].values, 'OrderVolume': final})
sub.to_csv('submission.csv', index=False)
print(f"\n✅ Saved submission.csv  ({len(sub)} rows)")
print(sub.describe().to_string())
print(sub.head(10).to_string())

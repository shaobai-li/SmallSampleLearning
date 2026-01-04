# -*- coding: utf-8 -*-
"""
Ensemble Stacking for Survival Analysis (YAML-based)
- RSF + EN-Cox base learners
- Single-branch meta: VAL-based robust weighting (cap)
- Fixed evaluation times: [12, 24] months (clipped to train max)
- Safe OOF selection + prune + one-pass fusion
- Multi-seed summary + BCa CI (fallbacks on small/degenerate samples)

Usage:
    python ensemble_pfs_stacking.py --config path/to/config.yaml
"""

from __future__ import annotations
import argparse
import yaml
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import json, warnings, numpy as np, pandas as pd
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from joblib import dump

from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")


# ====== 配置加载 ======
def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_seed_range(seed_cfg) -> List[int]:
    """解析 seed 配置，支持 list 或 range 格式"""
    if isinstance(seed_cfg, list):
        return seed_cfg
    elif isinstance(seed_cfg, dict):
        start = seed_cfg.get("start", 10)
        stop = seed_cfg.get("stop", 60)
        step = seed_cfg.get("step", 1)
        return list(range(start, stop, step))
    else:
        return [seed_cfg]


# ====== 工具 ======
def normalize_seed(s) -> int:
    if isinstance(s, (int, np.integer)): return int(s)
    return 42


def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)


def to_structured_y(df: pd.DataFrame, time_col: str, event_col: str):
    ev = df[event_col].astype(bool).values
    tt = df[time_col].astype(float).values
    return Surv.from_arrays(event=ev, time=tt)


def preprocess_features(df: pd.DataFrame, drop_na_rate=0.40) -> pd.DataFrame:
    x = df.copy().replace([np.inf, -np.inf], np.nan)
    keep = x.isna().mean() <= drop_na_rate
    x = x.loc[:, keep]
    num = x.select_dtypes(include=[np.number])
    zero = num.std(ddof=0)
    x = x.drop(columns=zero[zero == 0].index, errors="ignore")
    return x


def ensure_columns(df_src: pd.DataFrame, df_ref: pd.DataFrame) -> pd.DataFrame:
    need = [c for c in df_ref.columns if c not in df_src.columns]
    if need:
        df_src = df_src.copy()
        for c in need: df_src[c] = np.nan
    return df_src


def safe_feature_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    cols = list(cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        df = df.copy()
        for c in missing: df[c] = np.nan
    return df[cols].to_numpy(dtype=float)


class AdaptivePCA(BaseEstimator, TransformerMixin):
    """自适应 PCA：n_comp = min(k, n_feat-1, n_samp-1)，>=1"""
    def __init__(self, k=6, random_state=42):
        self.k = int(k); self.random_state = random_state
        self._pca = None; self.n_components_ = None
    def fit(self, X, y=None):
        X = np.asarray(X); n, p = X.shape
        nc = max(1, min(self.k, max(1, min(p-1, n-1))))
        self.n_components_ = nc
        self._pca = PCA(n_components=nc, svd_solver="auto", random_state=self.random_state).fit(X)
        return self
    def transform(self, X): return self._pca.transform(X)


def _scores_of(df, res):
    """按当前 res.meta['names'] + weights 生成集成分数"""
    X = preprocess_features(df)
    P = []
    for n in res.meta["names"]:
        br = res.base_models[n]
        Xi = X.reindex(columns=br.feats, fill_value=np.nan)
        P.append(np.asarray(br.pipe.predict(Xi), float))
    if len(P) == 0:
        return np.zeros(len(df), dtype=float)
    P = np.vstack(P)
    w = np.asarray(res.meta["weights"], float); w = w / (w.sum() + 1e-12)
    return (w[:, None] * P).sum(axis=0)


def save_run_outputs(res, df_train, df_val, df_test, time_col, event_col):
    """
    为当前 run 将 训练/验证/测试 的集成分数与 y 同步落盘，
    方便 final_km_report.py 直接读取复现 KM / HR / DCA / IBS。
    """
    outdir = Path(res.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 计算集成分数
    sc_train = _scores_of(df_train, res)
    sc_val   = _scores_of(df_val,   res)
    sc_test  = _scores_of(df_test,  res)

    # 2) 保存分数
    pd.Series(sc_train, name="final_train_score").to_csv(outdir/"final_train_scores.csv", index=False)
    pd.Series(sc_val,   name="final_val_score")  .to_csv(outdir/"final_val_scores.csv",   index=False)
    pd.Series(sc_test,  name="final_test_score") .to_csv(outdir/"final_test_scores.csv",  index=False)

    # 3) 保存 y（time/event）
    def _y(df): 
        return pd.DataFrame({
            "time":  df[time_col].astype(float).values,
            "event": df[event_col].astype(int).values
        })
    _y(df_train).to_csv(outdir/"y_train.csv", index=False)
    _y(df_val).to_csv(outdir/"y_val.csv",     index=False)
    _y(df_test).to_csv(outdir/"y_test.csv",   index=False)

    # 4) 保存阈值（训练分数中位数）
    thr = float(np.median(sc_train))
    with open(outdir/"threshold_median.txt", "w", encoding="utf-8") as f:
        f.write(f"{thr:.10f}\n")

    print(f"[DUMP] train/val/test scores + y + threshold saved to: {outdir}")


# ====== 评估（固定 tAUC@12/@24）======
def fixed_times_from_train(df_train: pd.DataFrame, time_col: str) -> np.ndarray:
    obs = df_train[time_col].astype(float).values
    if obs.size == 0:
        return np.array([12.0, 24.0], dtype=float)

    tmax = float(np.nanmax(obs))
    base = np.array([12.0, 24.0], dtype=float)
    # 裁剪到训练最大随访
    t = np.clip(base, 1e-6, max(tmax, 1e-6))

    # 去重后若只剩一个时间点，补一个与其不同的点（尽量贴近但不同）
    t = np.unique(t)
    if t.size == 1:
        t0 = float(t[0])
        # 取一个稍小的时间点（不少于 1e-3），仍需 <= t0
        t1 = max(1e-3, t0 * 0.8)
        if abs(t1 - t0) < 1e-6:
            t1 = max(1e-3, t0 - 1e-3)
        t = np.array(sorted({t0, t1}), dtype=float)

    return t


def eval_scores(df_eval, time_col, event_col, scores, times_auc, y_train_struct=None, tag="EVAL"):
    y_eval = to_structured_y(df_eval, time_col, event_col)
    scores = np.asarray(scores, dtype=float)

    c = float(concordance_index_censored(
        y_eval["event"], y_eval["time"], scores
    )[0])

    if y_train_struct is None:
        y_train_struct = y_eval

    # 确保 times_auc 为 1D
    times_auc = np.atleast_1d(np.asarray(times_auc, dtype=float))
    aucs, tt = cumulative_dynamic_auc(
        y_train_struct, y_eval, scores, times_auc
    )

    # 统一成 1D，兼容返回标量的实现
    aucs = np.atleast_1d(np.asarray(aucs, dtype=float))
    tt   = np.atleast_1d(np.asarray(tt,   dtype=float))

    # 防御：若 tt 长度为 0（极端异常），直接给 NaN
    if tt.size == 0 or aucs.size == 0:
        auc12 = float("nan"); auc24 = float("nan")
    else:
        # 找最接近 12 / 24 的时间点
        idx12 = int(np.argmin(np.abs(tt - 12.0)))
        idx24 = int(np.argmin(np.abs(tt - 24.0)))
        idx12 = np.clip(idx12, 0, aucs.size - 1)
        idx24 = np.clip(idx24, 0, aucs.size - 1)
        auc12 = float(aucs[idx12])
        auc24 = float(aucs[idx24])

    print(f"[{tag}] C-index={c:.3f}, tAUC@12={auc12:.3f}, tAUC@24={auc24:.3f}")
    return c, auc12, auc24


# ====== 候选器 ======
def candidates(random_state=42) -> Dict[str, Pipeline]:
    r = normalize_seed(random_state); C: Dict[str, Pipeline] = {}

    # EN-Cox （强正则起步）
    for l1r in (0.25, 0.15):
        C[f"EN-Cox(l1={l1r:.2f})"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("std", StandardScaler()),
            ("cox", CoxnetSurvivalAnalysis(alphas=np.logspace(3,-3,40), l1_ratio=l1r, max_iter=200000))
        ])

    # RSF（含自适应PCA一条 + 两条保守 RSF）
    C["RSF(PCA<=6)"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("std", RobustScaler()),
        ("pca", AdaptivePCA(k=6, random_state=r)),
        ("rsf", RandomSurvivalForest(n_estimators=600, min_samples_leaf=16, max_features="sqrt",
                                     n_jobs=-1, random_state=r)),
    ])
    for nm, kw in [
        ("RSF(leaf12,sqrt,600)", dict(n_estimators=600, min_samples_leaf=12, max_features="sqrt")),
        ("RSF(leaf16,0.5,700)",  dict(n_estimators=700, min_samples_leaf=16, max_features=0.5)),
    ]:
        C[nm] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("std", RobustScaler()),
            ("rsf", RandomSurvivalForest(random_state=r, n_jobs=-1, **kw)),
        ])
    return C


# ====== OOF + 单模最优 ======
def kfold_indices(n, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=normalize_seed(seed))
    for tr, va in kf.split(np.arange(n)):
        yield tr, va


@dataclass
class BaseResult:
    name: str
    feats: List[str]
    pipe: Pipeline
    oof: np.ndarray
    ci: float


def oof_for_pipeline(pipe: Pipeline, X: pd.DataFrame, y_struct, feats: List[str],
                     n_splits=5, random_state=42) -> Tuple[np.ndarray, float]:
    oof = np.full(len(X), np.nan, dtype=float)
    ok = 0
    for tr, va in kfold_indices(len(X), n_splits=n_splits, seed=random_state):
        Xi_tr, Xi_va = X.iloc[tr][feats], X.iloc[va][feats]
        yi_tr = Surv.from_arrays(event=y_struct["event"][tr], time=y_struct["time"][tr])
        try:
            m = clone(pipe).fit(Xi_tr, yi_tr)
            oof[va] = np.asarray(m.predict(Xi_va)).reshape(-1)
            ok += 1
        except Exception as e:
            print(f"[WARN] OOF fold failed: {e}")
    if ok == 0: return np.full(len(X), np.nan), float("nan")
    ci = float(concordance_index_censored(y_struct["event"], y_struct["time"], oof)[0])
    return oof, ci


def fit_single_modal_best(X: pd.DataFrame, y_struct, feats: List[str],
                          n_splits=5, seed=42) -> Optional[BaseResult]:
    best = None
    for nm, pipe in candidates(seed).items():
        oof, ci = oof_for_pipeline(pipe, X, y_struct, feats, n_splits=n_splits, random_state=seed)
        if not np.isfinite(ci): continue
        if (best is None) or (ci > best.ci):
            best = BaseResult(name=nm, feats=list(feats), pipe=clone(pipe), oof=oof, ci=float(ci))
    return best


# ====== 剪枝 + VAL 稳健加权 ======
def prune_modalities(ci_map: Dict[str, float], threshold=0.62, min_keep=2, top_k=2) -> List[str]:
    valid = [(nm, float(ci)) for nm, ci in ci_map.items() if np.isfinite(ci)]
    if not valid: raise RuntimeError("[PRUNE] no valid modalities.")
    valid.sort(key=lambda x: x[1], reverse=True)
    kept = [nm for nm,ci in valid if ci >= threshold]
    if len(kept) < min_keep: kept = [nm for nm,_ in valid[:min_keep]]
    if top_k and len(kept) > top_k: kept = kept[:top_k]
    print(f"[PRUNE] keep={kept} | thr={threshold} | min={min_keep} | top_k={top_k}")
    return kept


def robust_weights(val_cis: List[float], oof_cis: List[float], tau=0.06, cap=0.7) -> np.ndarray:
    s_val = np.clip(np.asarray(val_cis)-0.5, 0, None) + 1e-4
    s_oof = np.clip(np.asarray(oof_cis)-0.5, 0, None) + 1e-4
    z = (s_val * s_oof) / max(tau, 1e-6)
    w = np.exp(z - z.max()); w = w / (w.sum() + 1e-12)
    w = np.minimum(w, cap); w = w / (w.sum() + 1e-12)
    return w


@dataclass
class StackResult:
    endpoint: str
    base_models: Dict[str, BaseResult]
    meta: Dict[str, object]  # {"names":[...], "weights":[...]}
    times_auc: np.ndarray
    save_dir: str
    keep_names: List[str]


def train_stacking(df: pd.DataFrame, time_col: str, event_col: str,
                   modality_feats: Dict[str,List[str]], endpoint: str,
                   n_splits=5, save_root: Optional[Path]=None,
                   prune_threshold=0.62, prune_min_keep=2, prune_top_k=2,
                   seed=42) -> StackResult:
    assert time_col in df.columns and event_col in df.columns
    X = preprocess_features(df); y = to_structured_y(df, time_col, event_col)

    # 单模最优
    best_by_modal: Dict[str, BaseResult] = {}
    for mname, feats in modality_feats.items():
        feats = [f for f in feats if f in X.columns]
        if not feats: 
            print(f"[SKIP] {mname} no valid features."); continue
        br = fit_single_modal_best(X, y, feats, n_splits=n_splits, seed=seed)
        if br is None:
            print(f"[SKIP] {mname} all candidates failed."); continue
        # 拟合最终管线
        pipe_f = clone(br.pipe).fit(X[br.feats], y)
        best_by_modal[mname] = BaseResult(name=br.name, feats=br.feats, pipe=pipe_f, oof=br.oof, ci=br.ci)

    if not best_by_modal: raise RuntimeError("[FATAL] no modality trained.")

    # 剪枝
    ci_map = {m: br.ci for m, br in best_by_modal.items()}
    keep = prune_modalities(ci_map, prune_threshold, prune_min_keep, prune_top_k)

    # weights (先用 OOF 做近似，之后可用 VAL 重建)
    names = list(keep)
    oof_cis = [best_by_modal[n].ci for n in names]
    w = robust_weights(oof_cis, oof_cis, tau=0.06, cap=0.7)
    meta = {"names": names, "weights": w.tolist()}

    # times
    times = fixed_times_from_train(df, time_col)

    # 保存
    save_root = Path.cwd() / "stacking_runs" if save_root is None else Path(save_root)
    outdir = save_root / f"{endpoint}_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    outdir.mkdir(parents=True, exist_ok=True)
    for n in names:
        dump(best_by_modal[n].pipe, outdir / f"base_{safe_filename(n)}.joblib")
        pd.Series(best_by_modal[n].feats).to_csv(outdir / f"base_{safe_filename(n)}_features.csv", index=False)
    pd.DataFrame({"name": names, "weight": w, "oof_cindex": oof_cis}).to_csv(outdir/"meta_weights.csv", index=False)
    pd.Series(times).to_csv(outdir/"times_auc.csv", index=False)
    print(f"✅ Saved to: {outdir}")

    return StackResult(endpoint=endpoint, base_models=best_by_modal, meta=meta,
                       times_auc=times, save_dir=str(outdir), keep_names=names)


def predict_stack(df_new: pd.DataFrame, res: StackResult) -> np.ndarray:
    X = preprocess_features(df_new)
    P = []
    for n in res.meta["names"]:
        br = res.base_models[n]
        Xi = X.reindex(columns=br.feats, fill_value=np.nan)
        P.append(np.asarray(br.pipe.predict(Xi), float))
    P = np.vstack(P) if len(P)>0 else np.zeros((0, len(df_new)))
    w = np.asarray(res.meta["weights"], float); w = w / (w.sum() + 1e-12)
    return (w[:,None] * P).sum(axis=0)


# ====== 用 VAL 单模 C-index 重建权重（推荐）======
def rebuild_weights_with_val(df_val: pd.DataFrame, time_col, event_col, res: StackResult,
                             tau=0.04, cap=0.7):
    Xv = preprocess_features(df_val); yv = to_structured_y(df_val, time_col, event_col)
    cis = []
    for n in res.keep_names:
        br = res.base_models[n]
        sv = np.asarray(br.pipe.predict(Xv.reindex(columns=br.feats, fill_value=np.nan)), float)
        ci = float(concordance_index_censored(yv["event"], yv["time"], sv)[0])
        cis.append(ci)
    w = robust_weights(cis, [res.base_models[n].ci for n in res.keep_names], tau=tau, cap=cap)
    res.meta = {"names": list(res.keep_names), "weights": w.tolist()}
    pd.DataFrame({"name": res.keep_names, "weight": w, "val_cindex": cis}).to_csv(
        Path(res.save_dir)/"meta_weights_VAL_based.csv", index=False)
    print(f"[META] Rebuilt with VAL: names={res.keep_names}, weights={np.round(w,3)}")


# ====== Multi-seed + BCa ======
def _med_iqr(arr: List[float]):
    x = np.asarray([v for v in arr if np.isfinite(v)], float)
    if x.size == 0: return float("nan"), (float("nan"), float("nan"))
    return float(np.median(x)), (float(np.percentile(x,25)), float(np.percentile(x,75)))


def _bca_ci(arr: List[float], n_boot=2000, alpha=0.05) -> Tuple[float,float]:
    rng = np.random.default_rng(12345)
    x = np.asarray([v for v in arr if np.isfinite(v)], float)
    n = x.size
    if n == 0: return float("nan"), float("nan")
    if np.allclose(x, x[0]): return float(x[0]), float(x[0])
    if n < 8:  # 百分位退化
        idx = rng.integers(0, n, size=(n_boot, n))
        boots = np.median(x[idx], axis=1)
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    # jackknife
    theta = np.median(x)
    jk = np.array([np.median(np.r_[x[:i], x[i+1:]]) for i in range(n)], float)
    jk_mean = jk.mean()
    num = np.sum((jk_mean - jk)**3); den = 6.0 * (np.sum((jk_mean - jk)**2)**1.5 + 1e-12)
    a = num / (den + 1e-12)
    # bias-correct z0 via quick bootstrap of medians
    idx = rng.integers(0, n, size=(1000, n))
    boots = np.median(x[idx], axis=1); prop = (boots < theta).mean()
    from math import erf, sqrt, log
    def Phi(z): return 0.5*(1+erf(z/sqrt(2)))
    def Phi_inv(p):
        if p<=0: return -np.inf
        if p>=1: return  np.inf
        # simple probit approx
        # (Abramowitz-Stegun; adequate for CI endpoints)
        import math
        a1=-39.6968302866538; a2=220.946098424521; a3=-275.928510446969
        a4=138.357751867269;  a5=-30.6647980661472; a6=2.50662827745924
        b1=-54.4760987982241; b2=161.585836858041; b3=-155.698979859887
        b4=66.8013118877197;  b5=-13.2806815528857
        c1=-0.00778489400243029; c2=-0.322396458041136; c3=-2.40075827716184
        c4=-2.54973253934373;  c5=4.37466414146497;  c6=2.93816398269878
        d1=0.00778469570904146; d2=0.32246712907004;  d3=2.445134137143; d4=3.75440866190742
        plow=0.02425; phigh=1-plow
        if p<plow:
            q=math.sqrt(-2*math.log(p))
            return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
        if p>phigh:
            q=math.sqrt(-2*math.log(1-p))
            return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
        q=p-0.5; r=q*q
        return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q/(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)
    z0 = Phi_inv(prop)
    z_lo, z_hi = Phi_inv(alpha/2), Phi_inv(1-alpha/2)
    adj_lo = Phi(z0 + (z0+z_lo)/(1 - a*(z0+z_lo) + 1e-12))
    adj_hi = Phi(z0 + (z0+z_hi)/(1 - a*(z0+z_hi) + 1e-12))
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.median(x[idx], axis=1)
    return float(np.quantile(boots, adj_lo)), float(np.quantile(boots, adj_hi))


def summarize_many(board: pd.DataFrame, title="[MEDIAN±IQR + 95% BCa CI]"):
    def line(name, arr):
        med, (q1,q3) = _med_iqr(arr); lo, hi = _bca_ci(arr, 2000, 0.05)
        print(f"{name}: {med:.3f} (IQR {q1:.3f}–{q3:.3f}), 95% BCa CI [{lo:.3f}, {hi:.3f}]")
    print("\n"+title)
    line("VAL  C-index",  board["val_c"].tolist())
    line("TEST C-index",  board["test_c"].tolist())
    line("VAL  tAUC@12",  board["val_auc12"].tolist())
    line("VAL  tAUC@24",  board["val_auc24"].tolist())
    line("TEST tAUC@12",  board["test_auc12"].tolist())
    line("TEST tAUC@24",  board["test_auc24"].tolist())


# ====== 特征加载与融合 ======
def load_features_from_config(cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    """从配置文件加载特征列表"""
    feat_dir = Path(cfg["paths"]["feature_dir"])
    features = {}
    
    for name, feat_cfg in cfg["features"].items():
        if not feat_cfg.get("enabled", True):
            continue
        feat_file = feat_dir / feat_cfg["file"]
        if feat_file.exists():
            features[name] = pd.read_csv(feat_file)["feature"].tolist()
        else:
            print(f"[WARN] Feature file not found: {feat_file}")
    
    return features


def build_modality_dict(cfg: Dict[str, Any], features: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    """根据配置构建不同融合策略的模态字典"""
    modalities = {}
    
    # 获取临床特征（如果存在）
    clin_feats = features.get("clinpath", [])
    
    # 前融合辅助函数
    def fuse_with_clin(feats_img, feats_clin):
        return list(dict.fromkeys(list(feats_img) + list(feats_clin)))
    
    for exp_name, exp_cfg in cfg["experiments"].items():
        if not exp_cfg.get("enabled", True):
            continue
        
        fusion_type = exp_cfg.get("fusion_type", "none")
        modality_names = exp_cfg.get("modalities", [])
        
        mod_dict = {}
        for mod_name in modality_names:
            if mod_name not in features:
                print(f"[WARN] Modality '{mod_name}' not found in features, skipping.")
                continue
            
            if fusion_type == "early" and mod_name != "clinpath":
                # 前融合：影像特征 + 临床特征
                mod_dict[mod_name] = fuse_with_clin(features[mod_name], clin_feats)
            else:
                # none 或 late：直接使用原始特征
                mod_dict[mod_name] = features[mod_name]
        
        modalities[exp_name] = {
            "modality": mod_dict,
            "save_subdir": exp_cfg.get("save_subdir", f"stacking_runs_{exp_name}"),
            "fusion_type": fusion_type
        }
    
    return modalities


# ====== 实验运行器 ======
def run_experiment(label: str, modality: Dict[str, List[str]], save_subdir: str,
                   df: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                   time_col: str, event_col: str, y_tr,
                   feat_dir: Path, cfg: Dict[str, Any]):
    """
    运行单个实验（单次 + multi-seed）
    
    Args:
        label:       实验标签（如 'img', 'img_clin_early', 'hybrid'）
        modality:    传给 train_stacking 的模态字典
        save_subdir: 结果保存的子目录名
        df, df_val, df_test: 训练/验证/测试数据
        time_col, event_col: 生存时间和事件列名
        y_tr:        训练集结构化标签
        feat_dir:    特征目录
        cfg:         完整配置
    """
    print(f"\n\n========== RUN EXPERIMENT: {label} ==========")
    save_root = feat_dir / save_subdir
    
    # 获取配置参数
    model_cfg = cfg.get("model", {})
    n_splits = model_cfg.get("n_splits", 5)
    prune_threshold = model_cfg.get("prune_threshold", 0.62)
    prune_min_keep = model_cfg.get("prune_min_keep", 2)
    prune_top_k = model_cfg.get("prune_top_k", 2)
    tau = model_cfg.get("tau", 0.04)
    cap = model_cfg.get("cap", 0.7)
    
    seed_cfg = cfg.get("seeds", {})
    seed_main = seed_cfg.get("main", 42)
    seeds_multi = parse_seed_range(seed_cfg.get("multi", {"start": 10, "stop": 60, "step": 1}))
    
    # 1) 单次运行（主模型）
    res = train_stacking(
        df, time_col, event_col, modality, endpoint=cfg["survival"]["endpoint"],
        n_splits=n_splits,
        save_root=save_root,
        prune_threshold=prune_threshold, prune_min_keep=prune_min_keep, prune_top_k=prune_top_k,
        seed=seed_main
    )
    # 用 VAL 重建 stacking 权重
    rebuild_weights_with_val(df_val, time_col, event_col, res, tau=tau, cap=cap)

    # 预测 + 评估
    sc_val  = predict_stack(df_val,  res)
    sc_test = predict_stack(df_test, res)

    eval_scores(df_val,  time_col, event_col, sc_val,
                res.times_auc, y_tr, tag=f"VAL[{label}, seed={seed_main}]")
    eval_scores(df_test, time_col, event_col, sc_test,
                res.times_auc, y_tr, tag=f"TEST[{label}, seed={seed_main}]")

    # 2) Multi-seed 稳健性分析
    rows = []
    for sd in seeds_multi:
        print(f"\n--- Multi-seed run: label={label}, seed={sd} ---")
        r = train_stacking(
            df, time_col, event_col, modality, endpoint=cfg["survival"]["endpoint"],
            n_splits=n_splits,
            save_root=save_root,
            prune_threshold=prune_threshold, prune_min_keep=prune_min_keep, prune_top_k=prune_top_k,
            seed=sd
        )
        rebuild_weights_with_val(df_val, time_col, event_col, r, tau=tau, cap=cap)
        save_run_outputs(r, df, df_val, df_test, time_col=time_col, event_col=event_col)

        s_val  = predict_stack(df_val,  r)
        s_test = predict_stack(df_test, r)

        c_v, a12_v, a24_v = eval_scores(
            df_val,  time_col, event_col, s_val,  r.times_auc, y_tr,
            tag=f"VAL[{label}, seed={sd}]"
        )
        c_t, a12_t, a24_t = eval_scores(
            df_test, time_col, event_col, s_test, r.times_auc, y_tr,
            tag=f"TEST[{label}, seed={sd}]"
        )

        rows.append(dict(
            seed=sd,
            val_c=c_v,   test_c=c_t,
            val_auc12=a12_v, val_auc24=a24_v,
            test_auc12=a12_t, test_auc24=a24_t
        ))

    board = pd.DataFrame(rows).sort_values("test_c", ascending=False).reset_index(drop=True)
    print(f"\n[SUMMARY top 10 by TEST C-index]  ({label})")
    print(board.head(10))

    summarize_many(board, title=f"[{label}] MEDIAN±IQR + 95% BCa CI")

    out_csv = feat_dir / f"{cfg['survival']['endpoint']}_multi_seed_board_{label}.csv"
    board.to_csv(out_csv, index=False)
    print(f"\nSaved multi-seed summary to: {out_csv}")

    return res, board


# ============================================================
# 主入口
# ============================================================
def main(config_path: str):
    """主函数，基于 YAML 配置运行实验"""
    print(f"Loading config from: {config_path}")
    cfg = load_config(config_path)
    
    # --- 路径 ---
    paths = cfg["paths"]
    data_dir = Path(paths["data_dir"])
    feat_dir = Path(paths["feature_dir"])
    
    # --- 加载数据 ---
    df      = pd.read_csv(data_dir / paths["train_file"])
    df_val  = pd.read_csv(data_dir / paths["val_file"])
    df_test = pd.read_csv(data_dir / paths["test_file"])
    
    # 列对齐（防止 val/test 少列）
    df_val  = ensure_columns(df_val,  df)
    df_test = ensure_columns(df_test, df)
    
    # --- 生存标签 ---
    surv_cfg = cfg["survival"]
    time_col = surv_cfg["time_col"]
    event_col = surv_cfg["event_col"]
    y_tr = to_structured_y(df, time_col, event_col)
    
    # --- 加载特征 ---
    features = load_features_from_config(cfg)
    print(f"Loaded features: {list(features.keys())}")
    
    # --- 构建模态字典 ---
    modalities = build_modality_dict(cfg, features)
    
    # --- 依次运行实验 ---
    results = {}
    for exp_name, exp_data in modalities.items():
        res, board = run_experiment(
            label=exp_name,
            modality=exp_data["modality"],
            save_subdir=exp_data["save_subdir"],
            df=df, df_val=df_val, df_test=df_test,
            time_col=time_col, event_col=event_col, y_tr=y_tr,
            feat_dir=feat_dir, cfg=cfg
        )
        results[exp_name] = {"result": res, "board": board}
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Stacking for Survival Analysis")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to YAML configuration file")
    args = parser.parse_args()
    
    main(args.config)

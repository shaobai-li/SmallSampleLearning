# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from xml.parsers.expat import model

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from joblib import dump

from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis

from statsmodels.stats.multitest import multipletests


# ------------------ 基础工具 ------------------

def make_y_struct(time, event):
    """构造 sksurv 所需的结构化生存对象"""
    return Surv.from_arrays(event.astype(bool), time.astype(float))


def winsorize_df(X: pd.DataFrame, lower_q=0.01, upper_q=0.99):
    """按分位数 Winsorize 连续变量"""
    Xw = X.copy()
    num_cols = Xw.select_dtypes(include=[np.number]).columns
    q_low = Xw[num_cols].quantile(lower_q)
    q_hi = Xw[num_cols].quantile(upper_q)
    Xw[num_cols] = Xw[num_cols].clip(lower=q_low, upper=q_hi, axis=1)
    return Xw


def preprocess_features(X: pd.DataFrame, max_missing=0.4, lower_q=0.01, upper_q=0.99):
    """
    预处理：
    - 去掉全缺失
    - 去掉缺失率 > max_missing
    - Winsorize
    """
    Xc = X.replace([np.inf, -np.inf], np.nan).copy()

    drop_all_nan = Xc.columns[Xc.isna().all()].tolist()
    Xc = Xc.drop(columns=drop_all_nan)

    miss_rate = Xc.isna().mean()
    drop_high_miss = miss_rate[miss_rate > max_missing].index.tolist()
    Xc = Xc.drop(columns=drop_high_miss)

    Xc = winsorize_df(Xc, lower_q=lower_q, upper_q=upper_q)
    dropped = {"all_nan": drop_all_nan, "high_missing": drop_high_miss}
    return Xc, dropped


def fdr_filter(pvals: np.ndarray, alpha: float = 0.05, method: str = "fdr_bh"):
    """FDR 校正，返回 reject 标记和校正后的 p 值"""
    rej, p_adj, *_ = multipletests(pvals, alpha=alpha, method=method)
    return rej, p_adj


def spearman_prune(X: pd.DataFrame, thr: float = 0.9) -> List[str]:
    """
    Spearman 相关性去冗余：
    - 遍历特征，对高度相关对(|rho|>thr)保留方差更大 & 缺失率更小的
    """
    cols = list(X.columns)
    kept, removed = [], set()
    variances = X.var(axis=0).fillna(0.0)
    notnull_rate = 1.0 - X.isna().mean(axis=0)
    corr = X.corr(method="spearman").abs().fillna(0.0)

    for c in cols:
        if c in removed:
            continue
        drop_this = False
        for k in kept:
            if corr.loc[c, k] > thr:
                score_c = (variances[c], notnull_rate[c])
                score_k = (variances[k], notnull_rate[k])
                if score_c > score_k:
                    kept.remove(k)
                    removed.add(k)
                    kept.append(c)
                else:
                    drop_this = True
                break
        if (not drop_this) and (c not in kept) and (c not in removed):
            kept.append(c)
    return kept


def check_epv(df, event_col, n_feats):
    """计算 EPV（events per variable），便于在 JSON meta 中记录"""
    events = int(df[event_col].sum())
    epv = events / max(1, n_feats)
    flag = "OK" if epv >= 10 else ("LOW" if epv >= 5 else "RISK")
    return {"events": events, "features": n_feats, "EPV": epv, "flag": flag}


def cap_final_features_by_k(
    feats: List[str],
    coefs: Optional[pd.Series],
    k_target: int
) -> Tuple[List[str], Optional[pd.Series]]:
    """
    根据系数绝对值，从大到小保留前 k_target 个特征。
    若没有有效系数，就按原顺序截断。
    """
    if k_target is None or k_target <= 0:
        return feats, coefs

    if len(feats) <= k_target:
        return feats, coefs

    if coefs is not None and len(coefs) > 0:
        coef_abs = coefs.abs().reindex(feats).fillna(0.0)
        top_feats = coef_abs.sort_values(ascending=False).index[:k_target].tolist()
        new_coefs = coefs.reindex(top_feats)
        return top_feats, new_coefs

    top_feats = feats[:k_target]
    new_coefs = coefs.reindex(top_feats) if coefs is not None else coefs
    return top_feats, new_coefs


# ------------------ 单变量 Cox ------------------

def univariate_cox_select(
    X: pd.DataFrame,
    y_struct,
    alpha: float = 0.05,
    fdr_method: str = "fdr_bh",
    min_keep: int = 30,
    frac_keep: float = 0.2,
):
    """
    单变量 Cox 快速筛选：
    - 用 C-index 构造一个排序分数 rank_score = 2*(1 - C-index)，仅用于排序；
    - 对 rank_score 做 FDR 过滤；
    - 如通过特征太少，再按 rank_score 排序补足一定数量。
    """
    results = []
    for col in X.columns:
        Xi = X[[col]].copy()
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("std", RobustScaler(with_centering=True, with_scaling=True,
                                 quantile_range=(25, 75))),
            ("cox", CoxPHSurvivalAnalysis())
        ])
        try:
            pipe.fit(Xi, y_struct)
            log_hr = pipe.named_steps["cox"].coef_[0]
            hr = float(np.exp(log_hr))
            ci = concordance_index_censored(
                y_struct["event"], y_struct["time"], pipe.predict(Xi)
            )[0]
            # 越小越“差”的分数，仅用于排序，不当作真实 p 值使用
            rank_score = max(1e-6, min(1.0, 2.0 * (1.0 - ci)))
            results.append((col, hr, log_hr, ci, rank_score))
        except Exception:
            continue

    if len(results) == 0:
        return [], pd.DataFrame(columns=["feature", "HR", "logHR", "ci",
                                         "rank_score", "p_adj"])

    dfu = pd.DataFrame(results, columns=["feature", "HR", "logHR", "ci", "rank_score"])

    # 用 rank_score 做 FDR，仅作为筛选参考
    rej, p_adj = fdr_filter(dfu["rank_score"].values, alpha=alpha, method=fdr_method)
    dfu["p_adj"] = p_adj

    # 通过 FDR 的特征
    sel = dfu.loc[rej, "feature"].tolist()

    # 如果通过的太少，再按 rank_score 补足
    if len(sel) < min_keep:
        dfu_sorted = dfu.sort_values("rank_score", ascending=True)
        n_extra = max(
            min_keep - len(sel),
            int(len(X.columns) * frac_keep)
        )
        fallback = dfu_sorted["feature"].head(n_extra).tolist()
        sel = list(dict.fromkeys(sel + fallback))

    dfu = dfu.sort_values("rank_score", ascending=True).reset_index(drop=True)
    return sel, dfu


# ------------------ LASSO-Cox（CV） ------------------

@dataclass
class LassoCoxResult:
    best_alpha: float
    alphas_tested: List[float]
    cv_scores: List[float]
    nonzero_features: List[str]
    coef_series: pd.Series
    pipeline: Pipeline


def fit_lasso_cox_cv(
    X: pd.DataFrame,
    y_struct,
    l1_ratio: float = 1.0,
    alphas: Optional[np.ndarray] = None,
    n_splits: int = 5,
    random_state: int = 42,
    max_alpha_scale: int = 4
) -> LassoCoxResult:
    """
    使用 CoxnetSurvivalAnalysis 做 LASSO-Cox + KFold C-index CV，
    选择最佳 alpha，并返回非零特征及管线。
    """
    if alphas is None:
        alphas = np.logspace(-2, 2, 25)

    def make_pipe(a):
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("std", RobustScaler()),
            ("coxnet", CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                alphas=[a],
                max_iter=200000
            ))
        ])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scale_pow = 0
    while True:
        try:
            mean_scores = []
            for a in alphas:
                scores = []
                for tr, va in kf.split(X):
                    Xtr, Xva = X.iloc[tr], X.iloc[va]
                    ytr, yva = y_struct[tr], y_struct[va]
                    pipe = make_pipe(a)
                    pipe.fit(Xtr, ytr)
                    pred = pipe.predict(Xva)
                    ci = concordance_index_censored(yva["event"], yva["time"], pred)[0]
                    scores.append(ci)
                mean_scores.append(float(np.mean(scores)))

            best_idx = int(np.argmax(mean_scores))
            best_alpha = float(alphas[best_idx])

            final_pipe = make_pipe(best_alpha)
            final_pipe.fit(X, y_struct)

            coef_mat = final_pipe.named_steps["coxnet"].coef_
            # 注意：coef_.shape == (n_features, n_alphas)
            # alphas=[best_alpha] → 第 0 列对应这个 alpha 的全部特征系数
            coefs = pd.Series(coef_mat[:, 0], index=X.columns, name="coef")
            nonzero = coefs[coefs != 0].sort_values(key=lambda s: s.abs(),
                                                    ascending=False)

            return LassoCoxResult(
                best_alpha=best_alpha,
                alphas_tested=[float(a) for a in alphas],
                cv_scores=mean_scores,
                nonzero_features=nonzero.index.tolist(),
                coef_series=nonzero,
                pipeline=final_pipe
            )

        except ArithmeticError:
            scale_pow += 1
            if scale_pow > max_alpha_scale:
                raise
            alphas = alphas * 10.0
            print(f"[WARN] Coxnet不稳定，α整体×10重试（{scale_pow}） -> "
                  f"[{alphas.min():.3e}, {alphas.max():.3e}]")


# ------------------ 方案A：K-cap LASSO + Ridge 重拟 ------------------

def fit_cox_ridge(X, y_struct, alpha: float = 1.0, tiny_l1: float = 1e-6):
    """
    用极小的 l1_ratio 近似 Ridge（sksurv 不允许 l1_ratio=0）。
    alpha = L2 强度；tiny_l1=1e-6 基本等价于 Ridge。
    """
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("std", RobustScaler()),
        ("cox_en", CoxnetSurvivalAnalysis(
            l1_ratio=float(tiny_l1),
            alphas=[float(alpha)],
            max_iter=200000
        ))
    ])
    pipe.fit(X, y_struct)
    return pipe


def fit_lasso_with_k_cap(
    X: pd.DataFrame,
    y_struct,
    k_cap: int = 8,
    alphas: Optional[np.ndarray] = None,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    在一组 alpha grid 上搜索，使得 LASSO 模型的非零特征数 <= k_cap，
    返回对应的特征和系数（不含 Ridge 重拟）。
    """
    # alpha 网格
    if alphas is None:
        alpha_grid = np.logspace(-3, 1, 50)
    else:
        alpha_grid = np.array(alphas)

    best_alpha = None
    best_coefs = None

    for a in alpha_grid:
        model = CoxnetSurvivalAnalysis(
            alphas=[a],
            l1_ratio=1.0,
            fit_baseline_model=False,
            max_iter=200000,
        )
        model.fit(X, y_struct)
        coef_vec = model.coef_[:, 0]
        nnz = np.sum(coef_vec != 0)


        if 0 < nnz <= k_cap:
            best_alpha = float(a)
            best_coefs = coef_vec
            break

    # 如果所有 alpha 都没压到 k_cap，就用惩罚最弱的那一个作为 fallback
    if best_alpha is None:
        best_alpha = float(alpha_grid[-1])
        model = CoxnetSurvivalAnalysis(
            alphas=[best_alpha],
            l1_ratio=1.0,
            fit_baseline_model=False,
            max_iter=200000,
        )
        model.fit(X, y_struct)
        best_coefs = model.coef_[:, 0]

    coef_series = pd.Series(best_coefs, index=X.columns)
    nonzero = coef_series[coef_series != 0].sort_values(key=lambda s: s.abs(),
                                                        ascending=False)
    selected_features = list(nonzero.index)

    return {
        "alpha": best_alpha,
        "selected_features": selected_features,
        "coef_series": nonzero,
    }


# ------------------ 稳定性选择（备用可选，不在主流程里用） ------------------

def stability_selection_lasso(
    X,
    y_struct,
    n_iter: int = 200,
    sample_frac: float = 0.7,
    l1_ratio: float = 1.0,
    alpha: float = 1.0,
    freq_thresh: float = 0.6,
    random_state: int = 42,
):
    """
    简单稳定性选择：反复子样本 + LASSO，统计非零频率。
    """
    rng = np.random.RandomState(random_state)
    counts = pd.Series(0.0, index=X.columns)

    for _ in range(n_iter):
        idx = rng.choice(
            np.arange(X.shape[0]),
            size=int(X.shape[0] * sample_frac),
            replace=False
        )
        Xi, yi = X.iloc[idx], y_struct[idx]
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("std", RobustScaler()),
            ("coxnet", CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                alphas=[alpha],
                max_iter=200000
            ))
        ])
        try:
            pipe.fit(Xi, yi)
            coef_mat = pipe.named_steps["coxnet"].coef_
            coefs = pd.Series(coef_mat[:, 0], index=X.columns)
            counts.loc[coefs[coefs != 0].index] += 1.0
        except ArithmeticError:
            continue

    freq = counts / n_iter
    keep = freq[freq >= freq_thresh].index.tolist()
    return freq.sort_values(ascending=False), keep


# ------------------ 结果数据结构 ------------------

@dataclass
class ScreeningOutcome:
    endpoint: str
    kept_after_univariate: List[str]
    kept_after_corr: List[str]
    method: str                  # "lasso_cv" | "kcap_ridge" | "stability_ridge"
    selected_features: List[str]
    coef_series: pd.Series
    pipeline: Optional[Pipeline]
    meta: Dict
    save_dir: Optional[str] = None


# ------------------ 主函数：整合一切 ------------------

def screen_and_select_for_endpoint(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    feature_cols: List[str],
    # 基础筛选参数
    univar_alpha: float = 0.05,
    univar_fdr: str = "fdr_bh",
    corr_thr: float = 0.9,
    # LASSO 参数
    l1_ratio: float = 1.0,
    alphas: Optional[np.ndarray] = None,
    n_splits_lasso: int = 5,
    random_state: int = 42,
    # 终点
    endpoint: str = "OS",   # "PFS" or "OS"
    # 保存
    save: bool = True,
    save_root: Optional[str] = None,
) -> ScreeningOutcome:
    """
    在给定终点 (endpoint=OS/PFS) 下对一组特征进行完整筛选：
      0) 预处理
      1) 单变量 Cox + FDR + fallback
      2) Spearman 相关性去冗余
      3) 多变量选择: 统一用 LASSO-CV
      3.5) 最终特征 cap（最多 25 个）
      4) 保存输出 + 返回 ScreeningOutcome
    """

    # ---------------------
    # 0) 预处理
    # ---------------------
    X_raw = df[feature_cols].copy()
    X_full, dropped_cols = preprocess_features(
        X_raw,
        max_missing=0.40,
        lower_q=0.01,
        upper_q=0.99
    )
    y_struct = make_y_struct(df[time_col].values, df[event_col].values)

    
    if endpoint.upper() == "OS" and univar_alpha < 0.10:
        univar_alpha = 0.10

    # ---------------------
    # 1) 单变量 + FDR + fallback
    # ---------------------
    sel_univar, uni_table = univariate_cox_select(
        X_full,
        y_struct,
        alpha=univar_alpha,
        fdr_method=univar_fdr,
        # 保证候选足够多：至少 max(30, 10% 特征数)
        min_keep=max(30, X_full.shape[1] // 10),
        frac_keep=0.2,
    )

    if len(sel_univar) == 0:
        if "rank_score" in uni_table.columns:
            sel_univar = (
                uni_table
                .sort_values("rank_score")
                .head(min(50, max(10, X_full.shape[1] // 5)))
                ["feature"]
                .tolist()
            )
        else:
            sel_univar = list(X_full.columns)

    X_uni = X_full[sel_univar].copy()

    # ---------------------
    # 2) 相关性去冗余
    # ---------------------
    kept_corr = spearman_prune(X_uni, thr=corr_thr)
    X_corr = X_uni[kept_corr].copy()

    # ---------------------
    # 3) 多变量选择：统一 LASSO-CV
    # ---------------------
    method = "lasso_cv"

    if alphas is None:
        # 给一组较宽的 alpha 网格，便于 LASSO 自己决定稀疏程度
        alphas = np.logspace(-3, 2, 40)

    lasso_cv = fit_lasso_cox_cv(
        X_corr,
        y_struct,
        alphas=alphas,
        l1_ratio=l1_ratio,
        n_splits=n_splits_lasso,
        random_state=random_state,
    )
    selected_features = lasso_cv.nonzero_features
    coef_series = lasso_cv.coef_series
    final_pipe = lasso_cv.pipeline

    # 如果极端情况下没选上特征，用相关性去冗余后的前若干个兜底
    if coef_series is None or len(selected_features) == 0:
        selected_features = kept_corr[:min(10, len(kept_corr))]
        coef_series = pd.Series(0.0, index=selected_features)
        final_pipe = None

    # ---------------------
    # 3.5) 最终特征数 cap（只设上限，不设下限）
    # ---------------------
    n_before_cap = len(selected_features)

    # 统一限制最多 25 个
    k_final = 25

    selected_features, coef_series = cap_final_features_by_k(
        selected_features,
        coef_series,
        k_target=k_final,
    )
    n_after_cap = len(selected_features)

    # ---------------------
    # 4) 保存输出
    # ---------------------
    save_dir = None
    if save:
        if save_root is None:
            save_root = "./feature_selection"
        save_dir = Path(save_root)
        save_dir.mkdir(parents=True, exist_ok=True)

        out_csv = save_dir / f"selected_features_{endpoint}.csv"
        df_sel = pd.DataFrame({
            "feature": list(coef_series.index),
            "coef_or_score": list(coef_series.values),
        })
        df_sel.to_csv(out_csv, index=False)

        if final_pipe is not None:
            dump(final_pipe, save_dir / f"pipeline_{endpoint}.joblib")

        meta = {
            "endpoint": endpoint,
            "method": method,
            "univariate_kept": sel_univar,
            "after_corr_kept": kept_corr,
            "dropped_all_nan": dropped_cols.get("all_nan", []),
            "dropped_high_missing": dropped_cols.get("high_missing", []),
            "n_before_cap": int(n_before_cap),
            "n_after_cap": int(n_after_cap),
            "k_final": int(k_final),
            "selected_features": selected_features,
        }
        meta.update(check_epv(df, event_col, len(selected_features)))

        with open(save_dir / "selection_summary.json", "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[INFO] {endpoint} selection saved to: {save_dir.resolve()}")
    else:
        meta = {
            "endpoint": endpoint,
            "method": method,
            "univariate_kept": sel_univar,
            "after_corr_kept": kept_corr,
            "dropped_all_nan": dropped_cols.get("all_nan", []),
            "dropped_high_missing": dropped_cols.get("high_missing", []),
            "n_before_cap": int(n_before_cap),
            "n_after_cap": int(n_after_cap),
            "k_final": int(k_final),
            "selected_features": selected_features,
        }
        meta.update(check_epv(df, event_col, len(selected_features)))
        save_dir = None

    screening_result = ScreeningOutcome(
        endpoint=endpoint,
        kept_after_univariate=sel_univar,
        kept_after_corr=kept_corr,
        method=method,
        selected_features=selected_features,
        coef_series=coef_series,
        pipeline=final_pipe,
        meta=meta,
        save_dir=str(save_dir) if save_dir is not None else None,
    )
    return screening_result


# ------------------ 示例主程序 ------------------

if __name__ == "__main__":
    # 1) 读取 2.5D radiomics 特征
    df = pd.read_csv(
        r"D:\20251104\train\data\train.csv",
        encoding="utf-8"
    )

    # 2) 确保事件列是 0/1 int
    for c in ["PFS_event", "OS_event"]:
        if c in df.columns:
            df[c] = df[c].astype(int)

    # 3) 自动识别特征列（数值型，排除生存/ID 列）
    exclude = {
        "PFS", "PFS_event", "OS", "OS_event",
        "patient_id", "case_id", "ID", "StudyInstanceUID"
    }
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    print("特征数 =", len(feature_cols))

    # PFS：LASSO-CV
    out_pfs = screen_and_select_for_endpoint(
        df=df,
        time_col="PFS",
        event_col="PFS_event",
        feature_cols=feature_cols,
        endpoint="PFS",
        univar_alpha=0.05,               
        alphas=np.logspace(-3, 2, 40),
        save=True,
        save_root="./feature_selection_all_PFS"
    )

    # OS：同样 LASSO-CV
    out_os = screen_and_select_for_endpoint(
        df=df,
        time_col="OS",
        event_col="OS_event",
        feature_cols=feature_cols,
        endpoint="OS",
        univar_alpha=0.05,                
        alphas=np.logspace(-3, 2, 40),
        save=True,
        save_root="./feature_selection_all_OS"
    )


    print("PFS 结果保存目录：", out_pfs.save_dir)
    print("OS  结果保存目录：", out_os.save_dir)

# app_fuzzy.py
# -*- coding: utf-8 -*-
"""
Fuzzy Hybrid TOPSIS · Fuzzy Clustering ECO-Extended Apostle Model ·
Conditional Probability Ratios.

Method follows Indelicato, Moral-Cuadra, Orgaz-Agüera & Martín (2026),
International Journal of Sustainable Transportation, 20(1), 1-15,
https://doi.org/10.1080/15568318.2025.2557324 (Section 2).
"""

import io
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROFILE_NAMES = ["Least", "Intermediate", "Most"]   # internal column order
LEVEL_NAMES = {1: "Least", 2: "Undetermined", 3: "Intermediate", 4: "Most"}


# ============================================================
# 0) Optional password gate
# ============================================================

def gate():
    """
    If [passwords] exists in .streamlit/secrets.toml, require login.
    If it does not exist, the app remains open.
    """
    try:
        has_passwords = "passwords" in st.secrets and len(st.secrets["passwords"]) > 0
    except Exception:
        has_passwords = False

    if not has_passwords:
        return True

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔐 Access")
    st.write("Enter your username and password.")
    user = st.text_input("Username", key="auth_user")
    pwd = st.text_input("Password", type="password", key="auth_pwd")

    if st.button("Log in"):
        try:
            if user in st.secrets["passwords"] and str(st.secrets["passwords"][user]) == str(pwd):
                st.session_state.authenticated = True
                st.success("Access granted ✅")
                st.rerun()
            else:
                st.error("Incorrect username or password.")
        except Exception as e:
            st.error(f"Could not validate access: {e}")
    st.stop()


# ============================================================
# 1) Triangular Fuzzy Numbers (TFN)
# ============================================================

@dataclass(frozen=True)
class TFN:
    a: float
    b: float
    c: float

    def __post_init__(self):
        if not (self.a <= self.b <= self.c):
            raise ValueError("TFN requires a ≤ b ≤ c.")


def defuzz_buckley(x: TFN) -> float:
    """Buckley (1985): (a1 + 2·a2 + a3) / 4  — Eq. (2)."""
    return (x.a + 2.0 * x.b + x.c) / 4.0


def likert_map_1_4() -> Dict[int, TFN]:
    return {
        1: TFN(0, 0, 50),
        2: TFN(30, 50, 70),
        3: TFN(50, 70, 90),
        4: TFN(70, 100, 100),
    }


def likert_map_1_5_paper() -> Dict[int, TFN]:
    """Table 3 of Indelicato et al. (2026)."""
    return {
        1: TFN(0, 0, 30),
        2: TFN(20, 30, 40),
        3: TFN(30, 50, 70),
        4: TFN(60, 70, 80),
        5: TFN(70, 100, 100),
    }


def likert_map_1_5_symmetric() -> Dict[int, TFN]:
    return {
        1: TFN(0, 0, 25),
        2: TFN(10, 25, 40),
        3: TFN(30, 50, 70),
        4: TFN(60, 75, 90),
        5: TFN(75, 100, 100),
    }


def likert_map_1_6() -> Dict[int, TFN]:
    return {
        1: TFN(0, 0, 20),
        2: TFN(10, 20, 35),
        3: TFN(25, 40, 55),
        4: TFN(45, 60, 75),
        5: TFN(65, 80, 90),
        6: TFN(80, 100, 100),
    }


def linear_tfn_map(levels: List[int]) -> Dict[int, TFN]:
    levels = sorted(set(int(x) for x in levels))
    if len(levels) < 2:
        raise ValueError("At least two scale levels are required.")

    centers = np.linspace(0.0, 100.0, len(levels))
    mapping = {}

    for i, lv in enumerate(levels):
        b = float(centers[i])
        if i == 0:
            a = 0.0
            c = float((centers[i] + centers[i + 1]) / 2.0)
        elif i == len(levels) - 1:
            a = float((centers[i - 1] + centers[i]) / 2.0)
            c = 100.0
        else:
            a = float((centers[i - 1] + centers[i]) / 2.0)
            c = float((centers[i] + centers[i + 1]) / 2.0)
        mapping[lv] = TFN(a, b, c)

    return mapping


SCALE_OPTIONS = {
    "Likert 1-5 (paper, Table 3)": likert_map_1_5_paper,
    "Likert 1-5 (symmetric)": likert_map_1_5_symmetric,
    "Likert 1-4": likert_map_1_4,
    "Likert 1-6": likert_map_1_6,
    "Likert 1-7 (linear)": lambda: linear_tfn_map(list(range(1, 8))),
    "Likert 1-10 (linear)": lambda: linear_tfn_map(list(range(1, 11))),
    "Likert 0-10 (linear)": lambda: linear_tfn_map(list(range(0, 11))),
    "Likert 1-11 (linear)": lambda: linear_tfn_map(list(range(1, 12))),
}


# ============================================================
# 2) Validation and TFN conversion
# ============================================================

def item_problems(
    df: pd.DataFrame,
    items: List[str],
    levels_by_item: Dict[str, List[int]],
) -> Tuple[pd.Series, List[str]]:
    """
    Returns a boolean Series (True = row valid for every item) and a list
    of human-readable problems per item.
    """
    valid = pd.Series(True, index=df.index)
    problems = []

    for it in items:
        if it not in df.columns:
            problems.append(f"{it}: column not found.")
            valid &= False
            continue

        s = pd.to_numeric(df[it], errors="coerce")
        levels = set(levels_by_item[it])
        missing = s.isna()
        invalid = s.notna() & ~s.isin(levels)

        if missing.any():
            problems.append(f"{it}: {int(missing.sum())} missing/non-numeric value(s).")
        if invalid.any():
            bad = sorted(s[invalid].unique().tolist())
            problems.append(
                f"{it}: {int(invalid.sum())} value(s) outside {sorted(levels)}: {bad[:10]}"
            )
        valid &= ~(missing | invalid)

    return valid, problems


# ============================================================
# 2b) Missing values: diagnosis and replacement
# ============================================================

IMPUTATION_METHODS = [
    "Listwise deletion (no replacement)",
    "Respondent mean within the latent variable (person-mean)",
    "K-nearest neighbours hot-deck",
    "Item median",
    "Item mode",
]


def clean_items(df: pd.DataFrame, items: List[str],
                levels_by_item: Dict[str, List[int]]) -> pd.DataFrame:
    """Numeric copy of the items; values outside the scale become NaN."""
    out = pd.DataFrame(index=df.index)
    for it in items:
        s = pd.to_numeric(df[it], errors="coerce")
        out[it] = s.where(s.isin(set(levels_by_item[it])))
    return out


def missing_report(df: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(clean)
    for it in clean.columns:
        raw = df[it]
        n_blank = int(raw.isna().sum())
        n_bad = int(clean[it].isna().sum()) - n_blank
        rows.append({
            "Item": it,
            "Empty cells": n_blank,
            "Out-of-scale / non-numeric": n_bad,
            "Total missing": n_blank + n_bad,
            "% missing": 100.0 * (n_blank + n_bad) / n if n else np.nan,
        })
    return pd.DataFrame(rows)


def _snap(values: np.ndarray, levels: List[int]) -> np.ndarray:
    """Round to the nearest valid scale level (ties go to the lower level)."""
    lv = np.asarray(sorted(levels), dtype=float)
    idx = np.abs(values[:, None] - lv[None, :]).argmin(axis=1)
    return lv[idx]


def impute_missing(
    clean: pd.DataFrame,
    latents: Dict[str, List[str]],
    levels_by_item: Dict[str, List[int]],
    method: str,
    max_share: float = 0.34,
    k: int = 5,
):
    """
    Returns (imputed items, keep mask, n imputed per respondent, log table).

    Respondents with a share of missing items above `max_share` in any latent
    variable are dropped whatever the method. Imputed values are always valid
    scale levels, so they can be converted into TFNs.
    """
    data = clean.copy()
    items = list(clean.columns)
    miss = data.isna()

    keep = pd.Series(True, index=data.index)
    for lat_items in latents.values():
        share = miss[lat_items].mean(axis=1)
        keep &= share <= max_share + 1e-12

    if method.startswith("Listwise"):
        keep &= ~miss.any(axis=1)
        n_imp = pd.Series(0, index=data.index)
        log = pd.DataFrame({"Item": items, "Imputed": 0})
        return data, keep, n_imp, log

    work = data.loc[keep].copy()
    wmiss = work.isna()

    if method.startswith("Respondent mean"):
        for lat_items in latents.values():
            pm = work[lat_items].mean(axis=1)
            for it in lat_items:
                m = work[it].isna() & pm.notna()
                if m.any():
                    work.loc[m, it] = _snap(pm[m].to_numpy(), levels_by_item[it])

    elif method.startswith("K-nearest"):
        A = np.array(work[items].to_numpy(dtype=float), copy=True)
        M = np.isnan(A)
        donors = np.where(~M.any(axis=1))[0]
        if len(donors) == 0:
            raise ValueError("KNN needs at least one respondent without missing items.")
        D = A[donors]
        for i in np.where(M.any(axis=1))[0]:
            obs = ~M[i]
            if obs.any():
                d = np.sqrt(np.mean((D[:, obs] - A[i, obs]) ** 2, axis=1))
            else:
                d = np.zeros(len(donors))
            nn = np.argsort(d, kind="stable")[:min(k, len(donors))]
            for j in np.where(M[i])[0]:
                A[i, j] = _snap(np.array([np.median(D[nn, j])]), levels_by_item[items[j]])[0]
        work[items] = A

    elif method == "Item median":
        for it in items:
            med = work[it].median()
            work[it] = work[it].fillna(_snap(np.array([med]), levels_by_item[it])[0])

    elif method == "Item mode":
        for it in items:
            mode = work[it].mode()
            if len(mode):
                work[it] = work[it].fillna(float(mode.min()))

    data.loc[keep, items] = work[items]
    n_imp = (wmiss & work.notna()).sum(axis=1).reindex(data.index, fill_value=0)
    log = pd.DataFrame({
        "Item": items,
        "Imputed": [int((wmiss[it] & work[it].notna()).sum()) for it in items],
    })
    return data, keep, n_imp, log


def df_to_tfn_array(
    df: pd.DataFrame,
    items: List[str],
    tfn_maps: Dict[str, Dict[int, TFN]],
) -> np.ndarray:
    """(n respondents × k items × 3) array of TFNs."""
    n = len(df)
    k = len(items)
    X = np.zeros((n, k, 3), dtype=float)

    for j, it in enumerate(items):
        lut = {lv: (t.a, t.b, t.c) for lv, t in tfn_maps[it].items()}
        vals = pd.to_numeric(df[it], errors="raise").astype(int).to_numpy()
        X[:, j, :] = np.array([lut[int(v)] for v in vals], dtype=float).reshape(n, 3)

    return X


def defuzz_tfn_array(X: np.ndarray) -> np.ndarray:
    return (X[..., 0] + 2.0 * X[..., 1] + X[..., 2]) / 4.0


# ============================================================
# 3) Fuzzy Hybrid TOPSIS — Eqs. (3a)-(5)
# ============================================================

def topsis_closeness(V: np.ndarray, pis: np.ndarray, nis: np.ndarray):
    d_plus = np.sqrt(np.sum((pis[None, :] - V) ** 2, axis=1))    # Eq. (4a)
    d_minus = np.sqrt(np.sum((nis[None, :] - V) ** 2, axis=1))   # Eq. (4b)
    denom = d_plus + d_minus
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.where(denom > 0, d_minus / denom, np.nan)        # Eq. (5)
    return cc, d_plus, d_minus


def fuzzy_hybrid_topsis(V: np.ndarray, is_benefit: List[bool] = None):
    """
    TOPSIS on a matrix of crisp (defuzzified) values, rows = alternatives
    (respondents or groups), columns = items. All items contribute equally.
    PIS_j = max_i V_ij and NIS_j = min_i V_ij for benefit items (Eqs. 3a-3b).
    """
    n, k = V.shape
    if is_benefit is None:
        is_benefit = [True] * k
    if len(is_benefit) != k:
        raise ValueError("is_benefit must have one value per item.")

    ben = np.asarray(is_benefit, dtype=bool)
    pis = np.where(ben, V.max(axis=0), V.min(axis=0))
    nis = np.where(ben, V.min(axis=0), V.max(axis=0))

    cc, d_plus, d_minus = topsis_closeness(V, pis, nis)
    return {"cc": cc, "V": V, "pis": pis, "nis": nis,
            "d_plus": d_plus, "d_minus": d_minus}


def individual_pis_nis_audit(
    df: pd.DataFrame,
    items: List[str],
    tfn_maps: Dict[str, Dict[int, TFN]],
    latent_name: str,
) -> pd.DataFrame:
    rows = []
    for it in items:
        s = pd.to_numeric(df[it], errors="raise").astype(int)
        lo, hi = int(s.min()), int(s.max())
        rows.append({
            "Latent": latent_name,
            "Item": it,
            "Minimum observed response": lo,
            "N at minimum": int((s == lo).sum()),
            "NIS": defuzz_buckley(tfn_maps[it][lo]),
            "Maximum observed response": hi,
            "N at maximum": int((s == hi).sum()),
            "PIS": defuzz_buckley(tfn_maps[it][hi]),
        })
    return pd.DataFrame(rows)


# ============================================================
# 4) Group Fuzzy Hybrid TOPSIS — Eq. (1) aggregation, pooled ideals
# ============================================================

def group_label(var: str, level) -> str:
    return f"{var}: {level}"


def group_matrix(
    df: pd.DataFrame,
    X_tfn: np.ndarray,
    group_cols: List[str],
    include_total: bool = True,
) -> pd.DataFrame:
    """
    For every level of every grouping variable, the average TFN of each item
    (Eq. 1) is computed and defuzzified (Eq. 2). Returns one row per group.
    """
    rows, mats = [], []

    if include_total:
        rows.append({"Variable": "Total", "Group": "Total", "N": len(df)})
        mats.append(defuzz_tfn_array(X_tfn.mean(axis=0)[None, :, :])[0])

    for gcol in group_cols:
        series = df[gcol].reset_index(drop=True)
        levels = sorted(series.dropna().unique().tolist(), key=lambda v: str(v))
        for lv in levels:
            mask = (series == lv).to_numpy()
            if mask.sum() == 0:
                continue
            avg_tfn = X_tfn[mask].mean(axis=0)              # Eq. (1)
            mats.append(defuzz_tfn_array(avg_tfn[None, :, :])[0])
            rows.append({"Variable": gcol, "Group": str(lv), "N": int(mask.sum())})

    info = pd.DataFrame(rows)
    M = np.asarray(mats, dtype=float) if mats else np.empty((0, X_tfn.shape[1]))
    return info, M


def group_topsis(
    df: pd.DataFrame,
    X_tfn: np.ndarray,
    items: List[str],
    group_cols: List[str],
    latent_name: str,
    scope: str = "pooled",
):
    """
    scope = "pooled": PIS/NIS taken across all groups of all grouping
    variables together (Table 4 of the paper), Total row included.
    scope = "within": PIS/NIS taken within each grouping variable.
    Returns (scores table, ideal-solutions table, crisp group matrix).
    """
    info, M = group_matrix(df, X_tfn, group_cols, include_total=True)
    if len(info) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    labels = [
        "Total" if v == "Total" else group_label(v, g)
        for v, g in zip(info["Variable"], info["Group"])
    ]
    crisp = pd.DataFrame(M, columns=items)
    crisp.insert(0, "Group", labels)
    crisp.insert(0, "Latent", latent_name)

    cc = np.full(len(info), np.nan)
    ideal_rows = []

    if scope == "pooled":
        blocks = [("All groups", np.arange(len(info)))]
    else:
        blocks = [
            (v, np.where(info["Variable"].to_numpy() == v)[0])
            for v in ["Total"] + list(group_cols)
        ]
        blocks = [(v, idx) for v, idx in blocks if v != "Total" and len(idx) > 0]
        cc[0] = np.nan  # Total has no comparison set in within-variable mode

    for block_name, idx in blocks:
        res = fuzzy_hybrid_topsis(M[idx])
        cc[idx] = res["cc"]
        for j, it in enumerate(items):
            col = M[idx, j]
            ideal_rows.append({
                "Latent": latent_name,
                "Scope": block_name,
                "Item": it,
                "Group PIS": labels[idx[int(np.argmax(col))]],
                "PIS": float(col.max()),
                "Group NIS": labels[idx[int(np.argmin(col))]],
                "NIS": float(col.min()),
            })

    scores = info.copy()
    scores[f"TOPSIS-{latent_name}"] = cc
    return scores, pd.DataFrame(ideal_rows), crisp


# ============================================================
# 5) Fuzzy clustering for fuzzy data — Eqs. (6a)-(6c)
# ============================================================

def fuzzy_distance_squared(
    X: np.ndarray,
    centers: np.ndarray,
    w1_extremes: float = 0.7,
    w2_centers: float = 0.3,
) -> np.ndarray:
    """
    Squared fuzzy distance (Eq. 6b):
      d² = w2²‖a2 − p2‖² + w1²(‖a1 − p1‖² + ‖a3 − p3‖²),
    with w1 ≥ w2 ≥ 0 and w1 + w2 = 1 (Eq. 6c).
    """
    if w1_extremes < w2_centers:
        raise ValueError("w1 must be greater than or equal to w2.")
    if abs((w1_extremes + w2_centers) - 1.0) > 1e-8:
        raise ValueError("w1 + w2 must equal 1.")

    left = X[:, None, :, 0] - centers[None, :, :, 0]
    mid = X[:, None, :, 1] - centers[None, :, :, 1]
    right = X[:, None, :, 2] - centers[None, :, :, 2]

    d2 = (
        (w2_centers ** 2) * np.sum(mid ** 2, axis=2)
        + (w1_extremes ** 2) * (np.sum(left ** 2, axis=2) + np.sum(right ** 2, axis=2))
    )
    return np.maximum(d2, 0.0)


def memberships_from_distances(dist: np.ndarray, m: float) -> np.ndarray:
    """
    FCM membership update u_ic = 1 / Σ_k (d_ic / d_ik)^(2/(m−1)).
    A respondent identical to one (or several) prototypes gets full
    membership there (split equally among ties).
    """
    n, c = dist.shape
    U = np.zeros((n, c), dtype=float)
    power = 2.0 / (m - 1.0)
    zero = dist <= 1e-12
    has_zero = zero.any(axis=1)

    if has_zero.any():
        U[has_zero] = zero[has_zero] / zero[has_zero].sum(axis=1, keepdims=True)

    nz = ~has_zero
    if nz.any():
        D = dist[nz]
        ratios = (D[:, :, None] / D[:, None, :]) ** power
        U[nz] = 1.0 / ratios.sum(axis=2)

    return U


def representative_profiles(
    X_tfn: np.ndarray,
    cc: np.ndarray,
    tol: float = 1e-9,
):
    """
    Paper, Section 2.2: three representative citizens per latent variable.
      Least        = respondent(s) with the lowest individual TOPSIS
      Intermediate = respondent(s) whose individual TOPSIS is closest to the median
      Most         = respondent(s) with the highest individual TOPSIS
    When several respondents share that TOPSIS value, the prototype is the
    average of their TFN vectors.
    Returns prototypes in the order [Least, Intermediate, Most] and an info table.
    """
    ok = np.isfinite(cc)
    targets = {
        "Least": float(np.nanmin(cc)),
        "Intermediate": float(np.nanmedian(cc)),
        "Most": float(np.nanmax(cc)),
    }

    protos, info = [], []
    for name in PROFILE_NAMES:
        t = targets[name]
        gap = np.where(ok, np.abs(cc - t), np.inf)
        idx = np.where(gap <= gap.min() + tol)[0]
        protos.append(X_tfn[idx].mean(axis=0))
        info.append({
            "Profile": name,
            "Target TOPSIS": t,
            "TOPSIS of representative": float(cc[idx[0]]),
            "N respondents averaged": int(len(idx)),
            "Row index (first)": int(idx[0]),
        })

    return np.asarray(protos), pd.DataFrame(info)


def fuzzy_clustering_fixed_profiles(
    X_tfn: np.ndarray,
    cc: np.ndarray,
    m: float = 2.0,
    w1_extremes: float = 0.7,
    w2_centers: float = 0.3,
):
    """Memberships to the fixed Least / Intermediate / Most prototypes."""
    if m <= 1:
        raise ValueError("Fuzzifier m must be > 1.")
    centers, info = representative_profiles(X_tfn, cc)
    d2 = fuzzy_distance_squared(X_tfn, centers, w1_extremes, w2_centers)
    U = memberships_from_distances(np.sqrt(d2), m)
    n = X_tfn.shape[0]
    return {
        "U": U,
        "centers": centers,
        "d2": d2,
        "profile_info": info,
        "iterations": 0,
        "objective": float(np.sum((U ** m) * d2)),
        "fpc": float(np.sum(U ** 2) / n),
        "partition_entropy": float(-np.sum(U * np.log(U + 1e-15)) / n),
    }


def fuzzy_cmeans_iterative(
    X: np.ndarray,
    m: float = 2.0,
    w1_extremes: float = 0.7,
    w2_centers: float = 0.3,
    max_iter: int = 500,
    tol: float = 1e-7,
    seed: int = 42,
):
    """
    Alternative: free FCM for fuzzy data (D'Urso) with 3 estimated centroids,
    ordered afterwards as Least / Intermediate / Most by mean defuzzified value.
    """
    if m <= 1:
        raise ValueError("Fuzzifier m must be > 1.")
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    U = rng.random((n, 3))
    U = U / U.sum(axis=1, keepdims=True)

    it = 0
    for it in range(max_iter):
        U_old = U.copy()
        Um = U ** m
        centers = np.einsum("nc,nkd->ckd", Um, X) / (Um.sum(axis=0)[:, None, None] + 1e-15)
        d2 = fuzzy_distance_squared(X, centers, w1_extremes, w2_centers)
        U = memberships_from_distances(np.sqrt(d2), m)
        if np.max(np.abs(U - U_old)) < tol:
            break

    Um = U ** m
    centers = np.einsum("nc,nkd->ckd", Um, X) / (Um.sum(axis=0)[:, None, None] + 1e-15)
    d2 = fuzzy_distance_squared(X, centers, w1_extremes, w2_centers)

    order = np.argsort(defuzz_tfn_array(centers).mean(axis=1))
    U = U[:, order]
    centers = centers[order]
    d2 = d2[:, order]

    info = pd.DataFrame({
        "Profile": PROFILE_NAMES,
        "Mean defuzzified centroid": defuzz_tfn_array(centers).mean(axis=1),
    })
    return {
        "U": U,
        "centers": centers,
        "d2": d2,
        "profile_info": info,
        "iterations": it + 1,
        "objective": float(np.sum((U ** m) * d2)),
        "fpc": float(np.sum(U ** 2) / n),
        "partition_entropy": float(-np.sum(U * np.log(U + 1e-15)) / n),
    }


def eco_axis_level(U: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    U columns ordered [Least, Intermediate, Most].
      1 = Least membership > α
      2 = no membership > α (undetermined)
      3 = Intermediate membership > α
      4 = Most membership > α
    """
    out = np.full(U.shape[0], 2, dtype=int)
    out[U[:, 0] > alpha] = 1
    out[U[:, 1] > alpha] = 3
    out[U[:, 2] > alpha] = 4
    return out


def profile_centers_table(fcm: dict, items: List[str], latent_name: str) -> pd.DataFrame:
    rows = []
    for c, pname in enumerate(PROFILE_NAMES):
        for j, it in enumerate(items):
            a, b, cc = fcm["centers"][c, j, :]
            rows.append({
                "Latent": latent_name,
                "Profile": pname,
                "Item": it,
                "TFN_a": round(float(a), 4),
                "TFN_b": round(float(b), 4),
                "TFN_c": round(float(cc), 4),
                "Defuzz": round((a + 2 * b + cc) / 4.0, 4),
            })
    return pd.DataFrame(rows)


# ============================================================
# 6) Classic and ECO-Extended Apostle classification
# ============================================================

def classic_quadrant_codes(x, y, thr_x, thr_y) -> np.ndarray:
    hx = x >= thr_x
    hy = y >= thr_y
    return np.where(hx & hy, "HH", np.where(hx, "HL", np.where(hy, "LH", "LL")))


def macro_codes_from_levels(level_x: np.ndarray, level_y: np.ndarray) -> np.ndarray:
    """Levels 3-4 = high side, 1-2 = low side of each axis."""
    hx = level_x >= 3
    hy = level_y >= 3
    return np.where(hx & hy, "HH", np.where(hx, "HL", np.where(hy, "LH", "LL")))


def pure_codes_from_levels(level_x: np.ndarray, level_y: np.ndarray) -> np.ndarray:
    """Pure categories: both axes at an extreme level (1 or 4)."""
    out = np.full(len(level_x), "", dtype=object)
    out[(level_x == 4) & (level_y == 4)] = "HH"
    out[(level_x == 4) & (level_y == 1)] = "HL"
    out[(level_x == 1) & (level_y == 4)] = "LH"
    out[(level_x == 1) & (level_y == 1)] = "LL"
    return out


def counts_table(series: pd.Series, order: List[str], name: str) -> pd.DataFrame:
    n = len(series)
    rows = [{name: k, "N": int((series == k).sum())} for k in order]
    out = pd.DataFrame(rows)
    out["Percent"] = 100.0 * out["N"] / n if n else np.nan
    return out


# ============================================================
# 7) Conditional probability ratios with bootstrap CIs — Eq. (9)
# ============================================================

def _ratio(A: np.ndarray, B: np.ndarray) -> float:
    pA, pB = A.mean(), B.mean()
    if pA == 0 or pB == 0:
        return np.nan
    return float((A & B).mean() / (pA * pB))


def _ratio_bootstrap_matrix(Aboot: np.ndarray, Bboot: np.ndarray) -> np.ndarray:
    pA = Aboot.mean(axis=1)
    pB = Bboot.mean(axis=1)
    pAB = (Aboot & Bboot).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where((pA > 0) & (pB > 0), pAB / (pA * pB), np.nan)
    return r


def probability_ratios(
    df: pd.DataFrame,
    target_col: str,
    targets: List[str],
    covar_cols: List[str],
    max_levels: int = 30,
    n_boot: int = 1000,
    seed: int = 42,
    chunk: int = 250,
) -> pd.DataFrame:
    """
    R_AB = P(A∩B) / [P(A)P(B)], A = membership in a target category,
    B = membership in a covariate group. 95% percentile CIs from n_boot
    bootstrap resamples (the same resamples are used for every A-B pair).
    Driver: CI_low > 1. Barrier: CI_high < 1. Otherwise independent.
    """
    N = len(df)
    rng = np.random.default_rng(seed)
    idx_all = rng.integers(0, N, size=(n_boot, N))

    tgt = df[target_col].astype(str).to_numpy()
    rows = []

    for cov in covar_cols:
        series = df[cov]
        levels = sorted(series.dropna().unique().tolist(), key=lambda v: str(v))
        if len(levels) > max_levels:
            st.warning(f"'{cov}' has {len(levels)} levels; only the first {max_levels} are used.")
            levels = levels[:max_levels]

        for lv in levels:
            B = (series == lv).to_numpy()
            for t in targets:
                A = tgt == t
                vals = []
                for s in range(0, n_boot, chunk):
                    idx = idx_all[s:s + chunk]
                    vals.append(_ratio_bootstrap_matrix(A[idx], B[idx]))
                vals = np.concatenate(vals)
                vals = vals[np.isfinite(vals)]

                point = _ratio(A, B)
                if len(vals):
                    lo, hi = np.percentile(vals, [2.5, 97.5])
                else:
                    lo = hi = np.nan

                if np.isfinite(lo) and lo > 1:
                    assoc = "Driver (+)"
                elif np.isfinite(hi) and hi < 1:
                    assoc = "Barrier (−)"
                elif np.isfinite(lo):
                    assoc = "Independent"
                else:
                    assoc = "Not estimable"

                rows.append({
                    "Covariate": cov,
                    "Group": str(lv),
                    "Target": t,
                    "N group": int(B.sum()),
                    "N target ∩ group": int((A & B).sum()),
                    "Ratio": round(point, 4) if np.isfinite(point) else np.nan,
                    "CI_low (2.5%)": round(float(lo), 4) if np.isfinite(lo) else np.nan,
                    "CI_high (97.5%)": round(float(hi), 4) if np.isfinite(hi) else np.nan,
                    "Association": assoc,
                    "Note": "no cases in A∩B" if (A & B).sum() == 0 else "",
                })

    return pd.DataFrame(rows)


def drivers_barriers_table(ratios: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    """Table 5 layout: significant drivers and barriers per target."""
    cols = {}
    for t in targets:
        sub = ratios[ratios["Target"] == t]
        drv = sub[sub["Association"] == "Driver (+)"].sort_values("CI_low (2.5%)", ascending=False)
        bar = sub[sub["Association"] == "Barrier (−)"].sort_values("CI_high (97.5%)")
        cols[t] = (
            [f"{r.Covariate}: {r.Group}" for r in drv.itertuples()],
            [f"{r.Covariate}: {r.Group}" for r in bar.itertuples()],
        )

    n_d = max([len(v[0]) for v in cols.values()] + [1])
    n_b = max([len(v[1]) for v in cols.values()] + [1])
    out = {"Type": ["Drivers"] * n_d + ["Barriers"] * n_b}
    for t, (d, b) in cols.items():
        out[t] = d + [""] * (n_d - len(d)) + b + [""] * (n_b - len(b))
    return pd.DataFrame(out)


# ============================================================
# 8) Figures
# ============================================================

def fig_group_topsis(scores: pd.DataFrame, name_x: str, name_y: str):
    lab = [
        "Total" if v == "Total" else group_label(v, g)
        for v, g in zip(scores["Variable"], scores["Group"])
    ]
    n = len(lab)
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.28 * n + 1.2)))
    ypos = np.arange(n)
    h = 0.4
    ax.barh(ypos - h / 2, scores[f"TOPSIS-{name_x}"], height=h, color="#2b59c3", label=name_x)
    ax.barh(ypos + h / 2, scores[f"TOPSIS-{name_y}"], height=h, color="#2e8b57", label=name_y)
    ax.set_yticks(ypos)
    ax.set_yticklabels(lab, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("TOPSIS (0 to 1)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_ternary(U: np.ndarray, title: str, color: str):
    # vertices: Least (0,0), Intermediate (1,0), Most (0.5, √3/2)
    h = np.sqrt(3) / 2
    px = U[:, 1] * 1.0 + U[:, 2] * 0.5
    py = U[:, 2] * h
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color="#555", lw=1)
    ax.scatter(px, py, s=10, alpha=0.5, color=color, edgecolors="none")
    ax.text(0, -0.05, "Least", ha="center", va="top", fontsize=9)
    ax.text(1, -0.05, "Intermediate", ha="center", va="top", fontsize=9)
    ax.text(0.5, h + 0.03, "Most", ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, h + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_heatmap_4x4(level_x, level_y, name_x: str, name_y: str):
    grid = np.zeros((4, 4))
    for a, b in zip(level_x, level_y):
        grid[b - 1, a - 1] += 1
    pct = 100.0 * grid / max(len(level_x), 1)

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(pct, origin="lower", cmap="Blues", vmin=0)
    for r in range(4):
        for c in range(4):
            ax.text(c, r, f"{pct[r, c]:.2f}%", ha="center", va="center", fontsize=9,
                    color="white" if pct[r, c] > pct.max() * 0.6 else "black")
    ticks = [f"{k}\n{LEVEL_NAMES[k]}" for k in range(1, 5)]
    ax.set_xticks(range(4))
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_yticks(range(4))
    ax.set_yticklabels([t.replace("\n", " ") for t in ticks], fontsize=8)
    ax.axhline(1.5, color="#333", lw=1.2)
    ax.axvline(1.5, color="#333", lw=1.2)
    ax.set_xlabel(name_x)
    ax.set_ylabel(name_y)
    fig.colorbar(im, ax=ax, label="Percentage (%)")
    fig.tight_layout()
    return fig


def fig_classic(x, y, thr_x, thr_y, name_x, name_y, labels):
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(x, y, s=12, alpha=0.45, color="#2b59c3", edgecolors="none")
    ax.axvline(thr_x, color="#c0392b", lw=1)
    ax.axhline(thr_y, color="#c0392b", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"{name_x} (TOPSIS)")
    ax.set_ylabel(f"{name_y} (TOPSIS)")
    kw = dict(fontsize=8, color="#444", ha="center")
    ax.text((1 + thr_x) / 2, 0.97, labels["HH"], va="top", **kw)
    ax.text(thr_x / 2, 0.97, labels["LH"], va="top", **kw)
    ax.text((1 + thr_x) / 2, 0.02, labels["HL"], va="bottom", **kw)
    ax.text(thr_x / 2, 0.02, labels["LL"], va="bottom", **kw)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    return buf.getvalue()


# ============================================================
# 9) Streamlit UI helpers
# ============================================================

def parse_tfn_table(edited: pd.DataFrame) -> Tuple[Dict[int, TFN], List[str]]:
    """Validate a manually edited TFN table (Level, a1, a2, a3)."""
    errors, mapping = [], {}
    d = edited.dropna(how="all")
    for r in d.itertuples(index=False):
        lv, a, b, c = r.Level, r.a1, r.a2, r.a3
        if any(pd.isna(v) for v in (lv, a, b, c)):
            errors.append(f"Level {lv}: incomplete row.")
            continue
        if float(lv) != int(lv):
            errors.append(f"Level {lv}: levels must be integers.")
            continue
        if not (0 <= a <= b <= c <= 100):
            errors.append(f"Level {int(lv)}: need 0 ≤ a1 ≤ a2 ≤ a3 ≤ 100 (got {a}, {b}, {c}).")
            continue
        if int(lv) in mapping:
            errors.append(f"Level {int(lv)} appears twice.")
            continue
        mapping[int(lv)] = TFN(float(a), float(b), float(c))
    if len(mapping) < 2:
        errors.append("At least two levels are required.")
    else:
        lv = sorted(mapping)
        dz = [defuzz_buckley(mapping[k]) for k in lv]
        if any(d2 <= d1 for d1, d2 in zip(dz, dz[1:])):
            errors.append("Defuzzified values must increase with the level "
                          "(1 = lowest, K = highest).")
    return mapping, errors


def tfn_editor(key: str, title: str) -> Tuple[Dict[int, TFN], List[str]]:
    """Editable TFN table; a preset only fills the starting values."""
    st.markdown(f"**{title}**")
    preset = st.selectbox("Starting values", list(SCALE_OPTIONS.keys()), key=f"preset_{key}")
    base = tfn_table(SCALE_OPTIONS[preset]())[["Level", "a1", "a2", "a3"]]
    edited = st.data_editor(
        base,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=f"tfn_editor_{key}_{preset}",
        column_config={
            "Level": st.column_config.NumberColumn("Level", step=1, format="%d"),
            "a1": st.column_config.NumberColumn("a1 (left)", min_value=0.0, max_value=100.0),
            "a2": st.column_config.NumberColumn("a2 (centre)", min_value=0.0, max_value=100.0),
            "a3": st.column_config.NumberColumn("a3 (right)", min_value=0.0, max_value=100.0),
        },
    )
    mapping, errors = parse_tfn_table(edited)
    if errors:
        for e in errors:
            st.error(e)
    else:
        st.caption("Defuzzified (a1 + 2·a2 + a3)/4: " + ", ".join(
            f"{lv} → {defuzz_buckley(t):.2f}" for lv, t in sorted(mapping.items())))
    return mapping, errors


def build_scale_ui(items_x: List[str], items_y: List[str], name_x: str, name_y: str):
    """Main-area TFN tables, entered manually. Returns maps, levels and errors."""
    items = list(dict.fromkeys(items_x + items_y))
    mode = st.radio(
        "TFN tables",
        ["One table for all items", "One table per latent variable", "One table per item"],
        horizontal=True, key="tfn_mode",
    )
    tfn_maps, errors = {}, []

    if mode == "One table for all items":
        mp, er = tfn_editor("ALL", "All items")
        errors += er
        tfn_maps = {it: mp for it in items}
    elif mode == "One table per latent variable":
        c1, c2 = st.columns(2)
        with c1:
            mx, ex = tfn_editor("X", name_x)
        with c2:
            my, ey = tfn_editor("Y", name_y)
        errors += ex + ey
        for it in items_x:
            tfn_maps[it] = mx
        for it in items_y:
            if it in tfn_maps and tfn_maps[it] != my:
                errors.append(f"{it} is in both latent variables with different TFN tables.")
            tfn_maps[it] = my
    else:
        cols = st.columns(min(3, max(1, len(items))))
        for i, it in enumerate(items):
            with cols[i % len(cols)]:
                mp, er = tfn_editor(f"item_{it}", it)
                errors += [f"{it}: {e}" for e in er]
                tfn_maps[it] = mp

    levels_by_item = {it: sorted(mp.keys()) for it, mp in tfn_maps.items()}
    return tfn_maps, levels_by_item, errors


def tfn_table(mapping: Dict[int, TFN]) -> pd.DataFrame:
    return pd.DataFrame([
        {"Level": lv, "a1": t.a, "a2": t.b, "a3": t.c, "Defuzzified": defuzz_buckley(t)}
        for lv, t in sorted(mapping.items())
    ])


def extended_labels_ui(name_x: str, name_y: str):
    labels = {}
    for xl in range(1, 5):
        for yl in range(1, 5):
            default = f"{LEVEL_NAMES[xl]} {name_x} | {LEVEL_NAMES[yl]} {name_y}"
            labels[(xl, yl)] = st.text_input(
                f"({xl},{yl}) {LEVEL_NAMES[xl]} X · {LEVEL_NAMES[yl]} Y",
                value=default, key=f"ext_label_{xl}_{yl}",
            )
    return labels


def show_table(title: str, df: pd.DataFrame, file_name: str, caption: str = None):
    if title:
        st.subheader(title)
    if caption:
        st.caption(caption)
    st.dataframe(df, use_container_width=True)
    st.download_button(
        f"⬇️ {file_name}",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        key=f"dl_{file_name}",
    )


def show_fig(fig, file_name: str):
    st.pyplot(fig, use_container_width=False)
    st.download_button(f"⬇️ {file_name}", data=fig_to_png(fig), file_name=file_name,
                       mime="image/png", key=f"dl_{file_name}")
    plt.close(fig)


def excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            if d is not None and not d.empty:
                d.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


METHOD_TEXT = r"""
Implementation of Section 2 of Indelicato, Moral-Cuadra, Orgaz-Agüera & Martín (2026),
*International Journal of Sustainable Transportation*, 20(1), 1–15.

**1 · Fuzzy Hybrid TOPSIS**
1. Each response becomes a triangular fuzzy number (TFN) on [0, 100]. Default 1–5 table (paper, Table 3):
   1 → (0,0,30), 2 → (20,30,40), 3 → (30,50,70), 4 → (60,70,80), 5 → (70,100,100).
2. Groups: average TFN per item and group (Eq. 1). Individuals: their own TFNs.
3. Buckley defuzzification $(a_1 + 2a_2 + a_3)/4$ (Eq. 2).
4. All items are benefits. $PIS_j = \max_i$, $NIS_j = \min_i$ of the crisp values (Eqs. 3a–3b).
   For groups, the maximum and minimum are taken across **all groups of all grouping variables together**
   (plus the Total), as in Table 4 of the paper.
5. Euclidean distances to PIS and NIS (Eqs. 4a–4b); $TOPSIS = S^-/(S^+ + S^-)$ (Eq. 5). Items are unweighted.

**2 · Fuzzy Clustering ECO-Extended Apostle Model**
- Classic Apostle Model: individual TOPSIS of X and Y split by a threshold into four quadrants.
- Three representative profiles per latent variable: *Least* (respondent with the lowest individual TOPSIS),
  *Intermediate* (respondent at the median TOPSIS) and *Most* (highest TOPSIS). Tied respondents are averaged.
- Memberships to the three profiles use the fuzzy distance
  $d^2 = w_2^2\lVert a_2-p_2\rVert^2 + w_1^2(\lVert a_1-p_1\rVert^2+\lVert a_3-p_3\rVert^2)$, $w_1 \ge w_2$, $w_1+w_2=1$ (Eqs. 6a–6c),
  and the FCM rule $u_{ic} = 1/\sum_k (d_{ic}/d_{ik})^{2/(m-1)}$.
- Level per axis with threshold α: 1 = Least > α, 2 = undetermined, 3 = Intermediate > α, 4 = Most > α.
  This gives the 4×4 grid (16 categories). Levels 3–4 are the high side and 1–2 the low side, which gives
  the four ECO-Extended quadrants. **Pure** categories are the corner cells (1,1), (1,4), (4,1), (4,4).

**Missing values**
Out-of-scale or non-numeric answers count as missing. Respondents above the chosen share of
missing items in either latent variable are dropped; the rest are imputed with the selected technique
(person-mean within the latent variable, KNN hot-deck, item median or item mode), always rounded to a
valid scale level. Listwise deletion drops every incomplete respondent. The column `n_imputed_items`
flags imputed respondents so results can be compared with and without them.

**3 · Conditional probability ratios**
- $R_{AB} = P(A\cap B)/[P(A)P(B)]$ (Eq. 9), A = a Pure category, B = a covariate group.
- 95% percentile intervals from bootstrap resamples (1,000 in the paper).
- Driver: lower bound > 1. Barrier: upper bound < 1. Independent: the interval contains 1.
"""


# ============================================================
# 10) Streamlit App
# ============================================================

def main():
    st.set_page_config(page_title="Fuzzy Quadrant Mapper", page_icon="📊", layout="wide")
    gate()

    st.title("Fuzzy Hybrid TOPSIS · ECO-Extended Apostle Model")
    st.caption("Input scales must run from the lowest level (1) to the highest level (K). "
               "All items are treated as benefits.")

    with st.expander("Method used in this app", expanded=False):
        st.markdown(METHOD_TEXT)

    # --------------------------
    # Upload
    # --------------------------
    up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if up is None:
        st.info("Upload a dataset to start.")
        st.stop()

    try:
        # Only true blanks count as missing: answer labels such as "None" stay as categories.
        na = ["", "NA", "N/A", "n/a", "NaN", "nan", "#N/A", "NULL", "null"]
        if up.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(up, sep=None, engine="python",
                                 keep_default_na=False, na_values=na)
        else:
            df_raw = pd.read_excel(up, keep_default_na=False, na_values=na)
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        st.stop()

    st.subheader("Uploaded data")
    st.write(f"Rows: **{len(df_raw):,}** · Columns: **{len(df_raw.columns):,}**")
    st.dataframe(df_raw.head(), use_container_width=True)
    all_cols = list(df_raw.columns)

    # --------------------------
    # 1 · Latent variables
    # --------------------------
    st.sidebar.header("1 · Latent variables")
    lname_x = st.sidebar.text_input("Latent X name (horizontal axis)", value="EA")
    items_x = st.sidebar.multiselect("Items for X", all_cols, key="items_x_widget")
    lname_y = st.sidebar.text_input("Latent Y name (vertical axis)", value="STA")
    items_y = st.sidebar.multiselect("Items for Y", all_cols, key="items_y_widget")

    if lname_x.strip() == lname_y.strip():
        st.sidebar.error("The two latent variables need different names.")

    overlap = sorted(set(items_x) & set(items_y))
    if overlap:
        st.sidebar.warning("Items included in both latent variables: " + ", ".join(overlap))

    selected_unique = list(dict.fromkeys(items_x + items_y))

    if not items_x or not items_y:
        st.info("Select the items of both latent variables in the sidebar.")
        st.stop()

    # --------------------------
    # 2 · TFN conversion (manual)
    # --------------------------
    st.header("Step 1 · TFN conversion")
    st.caption("Type the triangular fuzzy number of every scale level. The preset only fills "
               "the starting values; the default reproduces Table 3 of the paper. "
               "Rows can be added or deleted.")
    tfn_map_by_item, levels_by_item, tfn_errors = build_scale_ui(
        items_x, items_y, lname_x, lname_y)

    # --------------------------
    # 3 · Missing values
    # --------------------------
    st.header("Step 2 · Missing values")
    if tfn_errors:
        st.warning("Fix the TFN table first.")
        st.stop()

    clean = clean_items(df_raw, selected_unique, levels_by_item)
    rep = missing_report(df_raw, clean)
    n_incomplete = int(clean.isna().any(axis=1).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Respondents", f"{len(clean):,}")
    c2.metric("With ≥1 missing item", f"{n_incomplete:,}")
    c3.metric("Missing cells", f"{int(clean.isna().sum().sum()):,}")

    impute_method = IMPUTATION_METHODS[0]
    max_share, k_nn = 0.34, 5
    if n_incomplete:
        st.dataframe(rep[rep["Total missing"] > 0], use_container_width=True, hide_index=True)
        impute_method = st.selectbox(
            "Replacement technique", IMPUTATION_METHODS, index=1,
            help=(
                "Person-mean: average of the respondent's own answers to the other items of the "
                "same latent variable, rounded to the nearest scale level. "
                "KNN hot-deck: median answer of the k most similar respondents with complete data. "
                "Item median / mode: the same value for everyone; simple but shrinks variance. "
                "Listwise: drop every respondent with a missing item."
            ),
        )
        c1, c2 = st.columns(2)
        max_share = c1.slider(
            "Maximum share of missing items per latent variable that can be replaced",
            0.0, 1.0, 0.34, 0.01,
            help="Respondents above this share in X or Y are dropped instead of imputed.",
            disabled=impute_method.startswith("Listwise"),
        )
        if impute_method.startswith("K-nearest"):
            k_nn = int(c2.number_input("k (neighbours)", 1, 50, 5, 1))
        try:
            _, keep_prev, nimp_prev, _ = impute_missing(
                clean, {lname_x: items_x, lname_y: items_y}, levels_by_item,
                impute_method, max_share, k_nn)
            st.caption(f"Preview: {int(keep_prev.sum()):,} respondents kept, "
                       f"{int((~keep_prev).sum()):,} dropped, "
                       f"{int((nimp_prev > 0).sum()):,} with imputed values "
                       f"({int(nimp_prev.sum()):,} cells).")
        except ValueError as e:
            st.error(str(e))
    else:
        st.success("No missing or out-of-scale values in the selected items.")

    # --------------------------
    # 3 · Classic quadrants
    # --------------------------
    st.sidebar.header("3 · Quadrant names and classic threshold")
    q4_labels = {
        "HH": st.sidebar.text_input("High X · High Y", value="Consistent", key="q4_hh"),
        "HL": st.sidebar.text_input("High X · Low Y", value="Conscientious", key="q4_hl"),
        "LH": st.sidebar.text_input("Low X · High Y", value="Inconsistent", key="q4_lh"),
        "LL": st.sidebar.text_input("Low X · Low Y", value="Negationist", key="q4_ll"),
    }
    thr_rule = st.sidebar.selectbox(
        "Classic threshold rule", ["Fixed value", "Sample median", "Sample mean"],
    )
    thr_x_fixed = thr_y_fixed = 0.5
    if thr_rule == "Fixed value":
        thr_x_fixed = st.sidebar.slider(f"{lname_x} threshold", 0.0, 1.0, 0.5, 0.01)
        thr_y_fixed = st.sidebar.slider(f"{lname_y} threshold", 0.0, 1.0, 0.5, 0.01)

    with st.sidebar.expander("Name the 16 extended cells (optional)", expanded=False):
        ext_labels = extended_labels_ui(lname_x, lname_y)

    # --------------------------
    # 4 · Fuzzy clustering
    # --------------------------
    st.sidebar.header("4 · Fuzzy clustering")
    cluster_mode = st.sidebar.selectbox(
        "Representative profiles",
        ["Fixed: min / median / max TOPSIS (paper)", "Estimated: iterative fuzzy C-means"],
    )
    with st.sidebar.expander("Clustering parameters", expanded=False):
        m_fuzzy = st.slider("Fuzzifier m", 1.1, 3.0, 2.0, 0.1)
        w1_extremes = st.slider("w1 — weight of TFN extremes (a1, a3)", 0.50, 1.00, 0.70, 0.05)
        w2_centers = round(1.0 - w1_extremes, 10)
        st.write(f"w2 — weight of TFN centre (a2) = **{w2_centers:.2f}**")
        alpha = st.slider("α (membership threshold)", 0.30, 0.80, 0.50, 0.05)

    # --------------------------
    # 5 · Groups and covariates
    # --------------------------
    st.sidebar.header("5 · Groups and covariates")
    group_cols = st.sidebar.multiselect("Group TOPSIS by", all_cols, key="group_cols_widget")
    scope_label = st.sidebar.radio(
        "Group ideal solutions (PIS/NIS)",
        ["Across all groups (paper)", "Within each grouping variable"],
    )
    scope = "pooled" if scope_label.startswith("Across") else "within"

    ratio_cols = st.sidebar.multiselect("Covariates for probability ratios", all_cols,
                                        key="ratio_cols_widget")
    ratio_targets = st.sidebar.multiselect(
        "Probability ratios for",
        ["Pure categories (paper)", "ECO-Extended quadrants", "Classic quadrants"],
        default=["Pure categories (paper)"],
    )
    n_boot = int(st.sidebar.number_input("Bootstrap replications", 100, 10000, 1000, 100))
    cov_missing_group = st.sidebar.checkbox(
        "Keep missing covariate values as a 'Missing' group", value=False,
        help="Otherwise respondents with a missing covariate are left out of that "
             "covariate's groups only.",
    )

    # ============================================================
    # Run
    # ============================================================
    st.header("Step 3 · Run")
    if st.button("Run analysis", type="primary"):
        if lname_x.strip() == lname_y.strip():
            st.error("The two latent variables need different names.")
            st.stop()

        try:
            imputed, keep, n_imp, imp_log = impute_missing(
                clean, {lname_x: items_x, lname_y: items_y}, levels_by_item,
                impute_method, max_share, k_nn)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        df = df_raw.loc[keep].copy()
        df[selected_unique] = imputed.loc[keep, selected_unique].astype(int)
        df["n_imputed_items"] = n_imp.loc[keep].astype(int)
        df = df.reset_index(drop=True)
        if cov_missing_group:
            for c in set(group_cols) | set(ratio_cols):
                df[c] = df[c].astype(object).where(df[c].notna(), "Missing")
        n_dropped = int((~keep).sum())
        imp_summary = rep.merge(imp_log, on="Item", how="left")
        imp_summary.insert(0, "Method", impute_method)

        if len(df) < 3:
            st.error("At least 3 valid respondents are required.")
            st.stop()

        X_tfn = df_to_tfn_array(df, items_x, tfn_map_by_item)
        Y_tfn = df_to_tfn_array(df, items_y, tfn_map_by_item)

        # Individual TOPSIS
        tx = fuzzy_hybrid_topsis(defuzz_tfn_array(X_tfn))
        ty = fuzzy_hybrid_topsis(defuzz_tfn_array(Y_tfn))
        x, y = tx["cc"], ty["cc"]
        if np.isnan(x).any() or np.isnan(y).any():
            st.error("TOPSIS is undefined because every respondent gave the same answers "
                     "to all items of a latent variable.")
            st.stop()

        audit = pd.concat([
            individual_pis_nis_audit(df, items_x, tfn_map_by_item, lname_x),
            individual_pis_nis_audit(df, items_y, tfn_map_by_item, lname_y),
        ], ignore_index=True)

        # Fuzzy clustering
        if cluster_mode.startswith("Fixed"):
            fx = fuzzy_clustering_fixed_profiles(X_tfn, x, m_fuzzy, w1_extremes, w2_centers)
            fy = fuzzy_clustering_fixed_profiles(Y_tfn, y, m_fuzzy, w1_extremes, w2_centers)
        else:
            fx = fuzzy_cmeans_iterative(X_tfn, m_fuzzy, w1_extremes, w2_centers, seed=42)
            fy = fuzzy_cmeans_iterative(Y_tfn, m_fuzzy, w1_extremes, w2_centers, seed=84)
        Ux, Uy = fx["U"], fy["U"]
        level_x = eco_axis_level(Ux, alpha)
        level_y = eco_axis_level(Uy, alpha)

        # Classic
        if thr_rule == "Sample median":
            thr_x, thr_y = float(np.median(x)), float(np.median(y))
        elif thr_rule == "Sample mean":
            thr_x, thr_y = float(np.mean(x)), float(np.mean(y))
        else:
            thr_x, thr_y = thr_x_fixed, thr_y_fixed

        classic_code = classic_quadrant_codes(x, y, thr_x, thr_y)
        macro_code = macro_codes_from_levels(level_x, level_y)
        pure_code = pure_codes_from_levels(level_x, level_y)

        quad_names = [q4_labels[k] for k in ["HH", "HL", "LH", "LL"]]
        pure_names = [f"Pure {q4_labels[k]}" for k in ["HH", "HL", "LH", "LL"]]

        res = pd.DataFrame({
            f"TOPSIS_{lname_x}": x,
            f"TOPSIS_{lname_y}": y,
            "ClassicQuadrant": [q4_labels[c] for c in classic_code],
            f"Level_{lname_x}": level_x,
            f"Level_{lname_y}": level_y,
            "Extended4x4": [ext_labels[(int(a), int(b))] for a, b in zip(level_x, level_y)],
            "ECOExtendedQuadrant": [q4_labels[c] for c in macro_code],
            "PureCategory": [f"Pure {q4_labels[c]}" if c else "Not pure" for c in pure_code],
        })
        for j, p in enumerate(PROFILE_NAMES):
            res[f"{lname_x}_u_{p}"] = Ux[:, j]
        for j, p in enumerate(PROFILE_NAMES):
            res[f"{lname_y}_u_{p}"] = Uy[:, j]

        full = pd.concat([df, res], axis=1)

        # Group TOPSIS
        if group_cols:
            gx, ix, cx = group_topsis(df, X_tfn, items_x, group_cols, lname_x, scope)
            gy, iy, cy = group_topsis(df, Y_tfn, items_y, group_cols, lname_y, scope)
            g_scores = gx.merge(gy, on=["Variable", "Group", "N"], how="outer", sort=False)
            g_ideal = pd.concat([ix, iy], ignore_index=True)
            g_crisp = pd.concat([cx, cy], ignore_index=True)
        else:
            g_scores = g_ideal = g_crisp = pd.DataFrame()

        # Probability ratios
        ratio_tables = {}
        if ratio_cols:
            spec = {
                "Pure categories (paper)": ("PureCategory", pure_names),
                "ECO-Extended quadrants": ("ECOExtendedQuadrant", quad_names),
                "Classic quadrants": ("ClassicQuadrant", quad_names),
            }
            with st.spinner("Bootstrapping probability ratios..."):
                for key in ratio_targets:
                    col, targets = spec[key]
                    r = probability_ratios(full, col, targets, ratio_cols, n_boot=n_boot)
                    ratio_tables[key] = (r, drivers_barriers_table(r, targets))

        st.session_state["results"] = {
            "full": full, "res": res, "audit": audit,
            "fx": fx, "fy": fy, "items_x": items_x, "items_y": items_y,
            "lname_x": lname_x, "lname_y": lname_y,
            "thr_x": thr_x, "thr_y": thr_y, "thr_rule": thr_rule,
            "q4_labels": q4_labels, "quad_names": quad_names, "pure_names": pure_names,
            "level_x": level_x, "level_y": level_y,
            "g_scores": g_scores, "g_ideal": g_ideal, "g_crisp": g_crisp, "scope": scope,
            "ratio_tables": ratio_tables, "alpha": alpha, "cluster_mode": cluster_mode,
            "tfn_map": tfn_map_by_item, "n_dropped": n_dropped,
            "imp_summary": imp_summary, "impute_method": impute_method,
        }
        st.success("Analysis completed.")

    # ============================================================
    # Results
    # ============================================================
    R = st.session_state.get("results")
    if R is None:
        return

    lx, ly = R["lname_x"], R["lname_y"]
    res, full = R["res"], R["full"]

    tabs = st.tabs([
        "1 · TOPSIS", "2 · Group TOPSIS", "3 · Fuzzy clustering",
        "4 · Apostle models", "5 · Probability ratios", "⬇️ Export",
    ])

    # 1 · Individual TOPSIS
    with tabs[0]:
        st.header("Individual Fuzzy Hybrid TOPSIS")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Mean {lx}", f"{res[f'TOPSIS_{lx}'].mean():.4f}")
        c2.metric(f"Mean {ly}", f"{res[f'TOPSIS_{ly}'].mean():.4f}")
        c3.metric("Respondents analysed", f"{len(res):,}",
                  delta=f"-{R['n_dropped']} dropped" if R["n_dropped"] else None,
                  delta_color="off")

        first_map = next(iter(R["tfn_map"].values()))
        with st.expander("TFN conversion used (first item)"):
            st.dataframe(tfn_table(first_map), use_container_width=True)

        st.caption(f"Missing values: {R['impute_method']}. "
                   f"{R['n_dropped']} respondent(s) dropped, "
                   f"{int((full['n_imputed_items'] > 0).sum())} with imputed items.")
        with st.expander("Missing values and replacement by item"):
            st.dataframe(R["imp_summary"], use_container_width=True, hide_index=True)

        show_table("Individual results", full, "topsis_individual.csv")
        show_table(
            "Individual PIS/NIS", R["audit"], "pis_nis_individual.csv",
            "PIS and NIS are the largest and smallest observed defuzzified value of each item "
            "across respondents.",
        )

    # 2 · Group TOPSIS
    with tabs[1]:
        st.header("Group Fuzzy Hybrid TOPSIS")
        if R["g_scores"].empty:
            st.info("Select one or more grouping variables in the sidebar and run the analysis again.")
        else:
            show_table("TOPSIS by group", R["g_scores"], "topsis_group.csv")
            st.subheader("Figure: TOPSIS by group")
            show_fig(fig_group_topsis(R["g_scores"], lx, ly), "fig_group_topsis.png")
            show_table(
                "Ideal solutions (Table 4 layout)", R["g_ideal"], "group_ideal_solutions.csv",
                "PIS = highest group mean of the item, NIS = lowest, with the group that reaches it. "
                + ("Taken across all groups of all grouping variables."
                   if R["scope"] == "pooled" else "Taken within each grouping variable."),
            )
            show_table("Defuzzified group means per item", R["g_crisp"], "group_crisp_means.csv")

    # 3 · Fuzzy clustering
    with tabs[2]:
        st.header("Fuzzy clustering")
        st.caption(R["cluster_mode"])
        metrics = pd.DataFrame([
            {"Latent": lx, "FPC": R["fx"]["fpc"], "Partition entropy": R["fx"]["partition_entropy"],
             "Objective": R["fx"]["objective"], "Iterations": R["fx"]["iterations"]},
            {"Latent": ly, "FPC": R["fy"]["fpc"], "Partition entropy": R["fy"]["partition_entropy"],
             "Objective": R["fy"]["objective"], "Iterations": R["fy"]["iterations"]},
        ])
        st.dataframe(metrics, use_container_width=True)

        pinfo = pd.concat([
            R["fx"]["profile_info"].assign(Latent=lx),
            R["fy"]["profile_info"].assign(Latent=ly),
        ], ignore_index=True)
        show_table("Representative profiles", pinfo, "representative_profiles.csv")

        # Dominant profile shares (paper reports % in each cluster)
        dom = []
        for name, U in [(lx, R["fx"]["U"]), (ly, R["fy"]["U"])]:
            arg = np.argmax(U, axis=1)
            for j, p in enumerate(PROFILE_NAMES):
                dom.append({"Latent": name, "Profile": p,
                            "Mean membership": float(U[:, j].mean()),
                            "% dominant": 100.0 * float((arg == j).mean())})
        show_table("Profile shares", pd.DataFrame(dom), "profile_shares.csv")

        centers = pd.concat([
            profile_centers_table(R["fx"], R["items_x"], lx),
            profile_centers_table(R["fy"], R["items_y"], ly),
        ], ignore_index=True)
        show_table("Profile TFNs per item", centers, "profile_tfns.csv")

        st.subheader("Figure: ternary plots of memberships")
        c1, c2 = st.columns(2)
        with c1:
            show_fig(fig_ternary(R["fx"]["U"], lx, "#2b59c3"), f"fig_ternary_{lx}.png")
        with c2:
            show_fig(fig_ternary(R["fy"]["U"], ly, "#e67e22"), f"fig_ternary_{ly}.png")

    # 4 · Apostle models
    with tabs[3]:
        st.header("Classic Apostle Model")
        st.caption(f"Threshold rule: {R['thr_rule']} — {lx} ≥ {R['thr_x']:.4f}, {ly} ≥ {R['thr_y']:.4f}")
        classic_counts = counts_table(res["ClassicQuadrant"], R["quad_names"], "Quadrant")
        show_table(None, classic_counts, "classic_quadrants.csv")
        show_fig(fig_classic(res[f"TOPSIS_{lx}"].to_numpy(), res[f"TOPSIS_{ly}"].to_numpy(),
                             R["thr_x"], R["thr_y"], lx, ly, R["q4_labels"]),
                 "fig_classic_apostle.png")

        st.header("Fuzzy Clustering ECO-Extended Apostle Model")
        st.caption(f"α = {R['alpha']:.2f}. Levels: 1 Least · 2 Undetermined · 3 Intermediate · 4 Most. "
                   "Quadrants: levels 3–4 = high side, 1–2 = low side.")
        show_fig(fig_heatmap_4x4(R["level_x"], R["level_y"], lx, ly), "fig_eco_extended_4x4.png")

        ext_quad = counts_table(res["ECOExtendedQuadrant"], R["quad_names"], "Quadrant")
        show_table("ECO-Extended quadrants", ext_quad, "eco_extended_quadrants.csv")

        pure = counts_table(res["PureCategory"], R["pure_names"] + ["Not pure"], "Category")
        show_table("Pure categories", pure, "pure_categories.csv")

        cells = (res.groupby([f"Level_{lx}", f"Level_{ly}", "Extended4x4"])
                 .size().reset_index(name="N"))
        cells["Percent"] = 100.0 * cells["N"] / len(res)
        show_table("All 16 cells", cells, "eco_extended_16_cells.csv")

        st.subheader("Classic vs ECO-Extended (counts)")
        st.dataframe(pd.crosstab(res["ClassicQuadrant"], res["ECOExtendedQuadrant"], margins=True),
                     use_container_width=True)

    # 5 · Probability ratios
    with tabs[4]:
        st.header("Conditional probability ratios")
        st.caption("R = P(A∩B) / [P(A)P(B)]. Driver if the 95% interval lies above 1, "
                   "barrier if it lies below 1. Groups with no respondents in the target "
                   "category give R = 0 with interval (0, 0) and are flagged in the Note column; "
                   "with small groups, read these as absence of cases rather than a measured effect.")
        if not R["ratio_tables"]:
            st.info("Select covariates (and a target set) in the sidebar and run the analysis again.")
        for key, (r, db) in R["ratio_tables"].items():
            st.subheader(key)
            slug = key.split(" (")[0].lower().replace(" ", "_").replace("-", "_")
            show_table("Drivers and barriers (Table 5 layout)", db, f"drivers_barriers_{slug}.csv")
            wide = r.pivot_table(index=["Covariate", "Group"], columns="Target",
                                 values=["CI_low (2.5%)", "CI_high (97.5%)"], sort=False,
                                 dropna=False)
            wide.columns = [f"{t} {c.split(' ')[0]}" for c, t in wide.columns]
            ordered = [f"{t} {b}" for t in r["Target"].unique() for b in ["CI_low", "CI_high"]]
            wide = wide[[c for c in ordered if c in wide.columns]].reset_index()
            show_table("Confidence intervals (Table A1 layout)", wide, f"ratios_ci_{slug}.csv")
            show_table("All ratios (long format)", r, f"ratios_long_{slug}.csv")

    # Export
    with tabs[5]:
        sheets = {
            "Individual": full,
            "PIS_NIS_individual": R["audit"],
            "Missing_values": R["imp_summary"],
            "Group_TOPSIS": R["g_scores"],
            "Group_ideal_solutions": R["g_ideal"],
            "Group_crisp_means": R["g_crisp"],
            "Classic": counts_table(res["ClassicQuadrant"], R["quad_names"], "Quadrant"),
            "ECO_quadrants": counts_table(res["ECOExtendedQuadrant"], R["quad_names"], "Quadrant"),
            "Pure": counts_table(res["PureCategory"], R["pure_names"] + ["Not pure"], "Category"),
        }
        for key, (r, db) in R["ratio_tables"].items():
            tag = {"Pure categories (paper)": "Pure", "ECO-Extended quadrants": "ECO",
                   "Classic quadrants": "Classic"}[key]
            sheets[f"Ratios_{tag}"] = r
            sheets[f"DriversBarriers_{tag}"] = db
        st.download_button(
            "⬇️ Download all tables (Excel)",
            data=excel_bytes(sheets),
            file_name="fuzzy_quadrant_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()

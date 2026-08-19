# app_fuzzy.py
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


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
    return (x.a + 2.0 * x.b + x.c) / 4.0


def likert_map_1_4() -> Dict[int, TFN]:
    return {
        1: TFN(0, 0, 50),
        2: TFN(30, 50, 70),
        3: TFN(50, 70, 90),
        4: TFN(70, 100, 100),
    }


def likert_map_1_5() -> Dict[int, TFN]:
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


def likert_map_1_7():
    return linear_tfn_map(list(range(1, 8)))


def likert_map_1_10():
    return linear_tfn_map(list(range(1, 11)))


def likert_map_1_11():
    return linear_tfn_map(list(range(1, 12)))


# ============================================================
# 2) Validation and TFN conversion
# ============================================================

def validate_items(
    df: pd.DataFrame,
    items: List[str],
    levels_by_item: Dict[str, List[int]],
) -> Tuple[bool, List[str]]:
    problems = []

    for it in items:
        if it not in df.columns:
            problems.append(f"{it}: column not found.")
            continue

        s = pd.to_numeric(df[it], errors="coerce")
        n_missing = int(s.isna().sum())

        levels = set(levels_by_item[it])
        invalid_mask = s.notna() & ~s.isin(levels)
        n_invalid = int(invalid_mask.sum())

        if n_missing:
            problems.append(f"{it}: {n_missing} missing/non-numeric value(s).")
        if n_invalid:
            bad = sorted(s[invalid_mask].unique().tolist())
            problems.append(
                f"{it}: {n_invalid} value(s) outside {sorted(levels)}: {bad[:10]}"
            )

    return len(problems) == 0, problems


def df_to_tfn_array(
    df: pd.DataFrame,
    items: List[str],
    tfn_maps: Dict[str, Dict[int, TFN]],
) -> np.ndarray:
    n = len(df)
    k = len(items)
    X = np.zeros((n, k, 3), dtype=float)

    for j, it in enumerate(items):
        mapping = tfn_maps[it]
        vals = pd.to_numeric(df[it], errors="raise").astype(int).to_numpy()

        for i, v in enumerate(vals):
            t = mapping[int(v)]
            X[i, j, :] = [t.a, t.b, t.c]

    return X


def defuzz_tfn_array(X: np.ndarray) -> np.ndarray:
    return (X[:, :, 0] + 2.0 * X[:, :, 1] + X[:, :, 2]) / 4.0


# ============================================================
# 3) Fuzzy-Hybrid TOPSIS
# ============================================================

def fuzzy_hybrid_topsis(
    X_tfn: np.ndarray,
    is_benefit: List[bool] = None,
):
    """
    Equal contribution of all items.

    For each item:
      PIS = maximum defuzzified value observed among all firms
      NIS = minimum defuzzified value observed among all firms

    Then:
      D+ = sqrt(sum_j (Vij - PISj)^2)
      D- = sqrt(sum_j (Vij - NISj)^2)
      TOPSIS = D- / (D+ + D-)

    No weight normalization is applied.
    """
    V = defuzz_tfn_array(X_tfn)
    n, k = V.shape

    if is_benefit is None:
        is_benefit = [True] * k

    if len(is_benefit) != k:
        raise ValueError("is_benefit must have one value per item.")

    pis = np.zeros(k, dtype=float)
    nis = np.zeros(k, dtype=float)

    for j in range(k):
        if is_benefit[j]:
            pis[j] = np.max(V[:, j])
            nis[j] = np.min(V[:, j])
        else:
            pis[j] = np.min(V[:, j])
            nis[j] = np.max(V[:, j])

    d_plus = np.sqrt(np.sum((V - pis[None, :]) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((V - nis[None, :]) ** 2, axis=1))

    cc = d_minus / (d_plus + d_minus + 1e-15)

    return {
        "cc": np.clip(cc, 0.0, 1.0),
        "V": V,
        "pis": pis,
        "nis": nis,
        "d_plus": d_plus,
        "d_minus": d_minus,
    }


def individual_pis_nis_audit(
    df: pd.DataFrame,
    items: List[str],
    tfn_maps: Dict[str, Dict[int, TFN]],
    latent_name: str,
) -> pd.DataFrame:
    """
    Transparent audit table showing why each PIS/NIS has its value.
    """
    rows = []

    for it in items:
        s = pd.to_numeric(df[it], errors="raise").astype(int)
        min_response = int(s.min())
        max_response = int(s.max())
        n_min = int((s == min_response).sum())
        n_max = int((s == max_response).sum())

        min_tfn = tfn_maps[it][min_response]
        max_tfn = tfn_maps[it][max_response]

        rows.append({
            "Latent": latent_name,
            "Item": it,
            "Minimum observed response": min_response,
            "N at minimum": n_min,
            "NIS": defuzz_buckley(min_tfn),
            "Maximum observed response": max_response,
            "N at maximum": n_max,
            "PIS": defuzz_buckley(max_tfn),
        })

    return pd.DataFrame(rows)


# ============================================================
# 4) Group TOPSIS and group PIS/NIS
# ============================================================

def group_defuzz_matrix(
    df: pd.DataFrame,
    items: List[str],
    tfn_maps: Dict[str, Dict[int, TFN]],
    group_col: str,
):
    groups = list(df.groupby(group_col, dropna=True))
    if not groups:
        return [], np.empty((0, len(items)))

    group_names = []
    M = []

    for g, dfg in groups:
        X = df_to_tfn_array(dfg, items, tfn_maps)
        V = defuzz_tfn_array(X)
        M.append(V.mean(axis=0))
        group_names.append(g)

    return group_names, np.asarray(M, dtype=float)


def group_topsis_one_latent(
    df: pd.DataFrame,
    items: List[str],
    tfn_maps: Dict[str, Dict[int, TFN]],
    group_col: str,
) -> pd.DataFrame:
    group_names, M = group_defuzz_matrix(df, items, tfn_maps, group_col)

    if len(group_names) == 0:
        return pd.DataFrame(columns=["Variable", "Item", "TOPSIS"])

    pis = M.max(axis=0)
    nis = M.min(axis=0)

    d_plus = np.sqrt(np.sum((M - pis[None, :]) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((M - nis[None, :]) ** 2, axis=1))
    cc = d_minus / (d_plus + d_minus + 1e-15)

    return pd.DataFrame({
        "Variable": group_col,
        "Item": [str(g) for g in group_names],
        "TOPSIS": np.round(cc, 6),
    })


def unified_group_topsis_table(
    df: pd.DataFrame,
    items_x: List[str],
    items_y: List[str],
    tfn_maps: Dict[str, Dict[int, TFN]],
    group_cols: List[str],
    name_x: str,
    name_y: str,
) -> pd.DataFrame:
    rows = []

    for gcol in group_cols:
        tx = group_topsis_one_latent(df, items_x, tfn_maps, gcol)
        ty = group_topsis_one_latent(df, items_y, tfn_maps, gcol)

        merged = tx.merge(
            ty,
            on=["Variable", "Item"],
            how="outer",
            suffixes=(f"_{name_x}", f"_{name_y}"),
        )

        for _, r in merged.iterrows():
            rows.append({
                "Variable": r["Variable"],
                "Item": r["Item"],
                f"TOPSIS-{name_x}": r.get(f"TOPSIS_{name_x}", np.nan),
                f"TOPSIS-{name_y}": r.get(f"TOPSIS_{name_y}", np.nan),
            })

    return pd.DataFrame(rows)


def group_pis_nis_table(
    df: pd.DataFrame,
    items: List[str],
    tfn_maps: Dict[str, Dict[int, TFN]],
    group_cols: List[str],
    latent_name: str,
) -> pd.DataFrame:
    """
    For each grouping variable and each latent item:
      group PIS = largest group mean
      group NIS = smallest group mean
    """
    rows = []

    for gcol in group_cols:
        group_names, M = group_defuzz_matrix(df, items, tfn_maps, gcol)
        if len(group_names) == 0:
            continue

        for j, it in enumerate(items):
            col = M[:, j]
            i_pis = int(np.argmax(col))
            i_nis = int(np.argmin(col))

            rows.append({
                "Latent": latent_name,
                "Grouping variable": gcol,
                "Item": it,
                "Group PIS": float(col[i_pis]),
                "PIS group": str(group_names[i_pis]),
                "Group NIS": float(col[i_nis]),
                "NIS group": str(group_names[i_nis]),
            })

    return pd.DataFrame(rows)


# ============================================================
# 5) Fuzzy C-Means on TFN item vectors
# ============================================================

def fuzzy_distance_squared(
    X: np.ndarray,
    centers: np.ndarray,
    w1_extremes: float = 0.7,
    w2_centers: float = 0.3,
) -> np.ndarray:
    if w1_extremes < w2_centers:
        raise ValueError("w1 must be greater than or equal to w2.")
    if abs((w1_extremes + w2_centers) - 1.0) > 1e-8:
        raise ValueError("w1 + w2 must equal 1.")

    left = X[:, None, :, 0] - centers[None, :, :, 0]
    mid = X[:, None, :, 1] - centers[None, :, :, 1]
    right = X[:, None, :, 2] - centers[None, :, :, 2]

    d2 = (
        (w2_centers ** 2) * np.sum(mid ** 2, axis=2)
        + (w1_extremes ** 2)
        * (np.sum(left ** 2, axis=2) + np.sum(right ** 2, axis=2))
    )

    return np.maximum(d2, 0.0)


def _update_memberships_from_distances(dist: np.ndarray, m: float) -> np.ndarray:
    n, c = dist.shape
    U = np.zeros((n, c), dtype=float)
    power = 2.0 / (m - 1.0)

    for i in range(n):
        zero_idx = np.where(dist[i] <= 1e-14)[0]

        if len(zero_idx) > 0:
            U[i, zero_idx] = 1.0 / len(zero_idx)
            continue

        for j in range(c):
            ratios = (dist[i, j] / dist[i, :]) ** power
            U[i, j] = 1.0 / np.sum(ratios)

    return U


def fuzzy_cmeans_fuzzy_data(
    X: np.ndarray,
    n_clusters: int = 3,
    m: float = 2.0,
    w1_extremes: float = 0.7,
    w2_centers: float = 0.3,
    max_iter: int = 500,
    tol: float = 1e-7,
    seed: int = 42,
):
    if m <= 1:
        raise ValueError("FCM fuzzifier m must be > 1.")
    if n_clusters < 2:
        raise ValueError("At least 2 clusters are required.")

    n = X.shape[0]
    rng = np.random.default_rng(seed)

    U = rng.random((n, n_clusters))
    U = U / U.sum(axis=1, keepdims=True)

    objective_history = []

    for iteration in range(max_iter):
        U_old = U.copy()
        Um = U ** m

        denom = Um.sum(axis=0)[:, None, None] + 1e-15
        centers = np.einsum("nc,nkd->ckd", Um, X) / denom

        d2 = fuzzy_distance_squared(
            X,
            centers,
            w1_extremes=w1_extremes,
            w2_centers=w2_centers,
        )
        dist = np.sqrt(d2 + 1e-30)

        objective = float(np.sum(Um * d2))
        objective_history.append(objective)

        U = _update_memberships_from_distances(dist, m)

        if np.max(np.abs(U - U_old)) < tol:
            break

    Um = U ** m
    denom = Um.sum(axis=0)[:, None, None] + 1e-15
    centers = np.einsum("nc,nkd->ckd", Um, X) / denom

    d2 = fuzzy_distance_squared(
        X,
        centers,
        w1_extremes=w1_extremes,
        w2_centers=w2_centers,
    )

    fpc = float(np.sum(U ** 2) / n)
    pe = float(-np.sum(U * np.log(U + 1e-15)) / n)

    return {
        "U": U,
        "centers": centers,
        "d2": d2,
        "iterations": iteration + 1,
        "objective_history": objective_history,
        "fpc": fpc,
        "partition_entropy": pe,
    }


def order_fcm_low_intermediate_high(fcm_result: dict):
    centers = fcm_result["centers"]

    if centers.shape[0] != 3:
        raise ValueError("The extended 4×4 classification requires exactly 3 clusters.")

    center_crisp = (
        centers[:, :, 0] + 2.0 * centers[:, :, 1] + centers[:, :, 2]
    ) / 4.0

    profile_score = center_crisp.mean(axis=1)
    order = np.argsort(profile_score)

    low_idx, mid_idx, high_idx = order.tolist()

    return {
        **fcm_result,
        "U_ordered": fcm_result["U"][:, [low_idx, mid_idx, high_idx]],
        "centers_ordered": centers[[low_idx, mid_idx, high_idx], :, :],
        "profile_scores_ordered": profile_score[[low_idx, mid_idx, high_idx]],
    }


def eco_axis_level(U_low_mid_high: np.ndarray, alpha: float = 0.5):
    """
    Internal 4-level classification:
      1 = Low
      2 = Hybrid / transition
      3 = Intermediate
      4 = High
    """
    out = np.full(U_low_mid_high.shape[0], 2, dtype=int)

    low = U_low_mid_high[:, 0] > alpha
    mid = U_low_mid_high[:, 1] > alpha
    high = U_low_mid_high[:, 2] > alpha

    out[low] = 1
    out[mid] = 3
    out[high] = 4

    return out


def fcm_centers_table(
    fcm_ordered: dict,
    items: List[str],
    latent_name: str,
) -> pd.DataFrame:
    centers = fcm_ordered["centers_ordered"]
    profile_names = ["Low", "Intermediate", "High"]
    rows = []

    for c, pname in enumerate(profile_names):
        for j, it in enumerate(items):
            t = TFN(*centers[c, j, :])
            rows.append({
                "Latent": latent_name,
                "Profile": pname,
                "Item": it,
                "TFN_a": round(t.a, 4),
                "TFN_b": round(t.b, 4),
                "TFN_c": round(t.c, 4),
                "Defuzz": round(defuzz_buckley(t), 4),
            })

    return pd.DataFrame(rows)


# ============================================================
# 6) Classic and extended quadrant classification
# ============================================================

def classic_quadrant_labels(
    x: np.ndarray,
    y: np.ndarray,
    thr_x: float,
    thr_y: float,
    labels: Dict[str, str],
) -> List[str]:
    out = []

    for xi, yi in zip(x, y):
        if xi >= thr_x and yi >= thr_y:
            out.append(labels["HH"])
        elif xi >= thr_x and yi < thr_y:
            out.append(labels["HL"])
        elif xi < thr_x and yi >= thr_y:
            out.append(labels["LH"])
        else:
            out.append(labels["LL"])

    return out


def extended_16_labels(
    level_x: np.ndarray,
    level_y: np.ndarray,
    labels_16: Dict[Tuple[int, int], str],
) -> List[str]:
    return [
        labels_16[(int(a), int(b))]
        for a, b in zip(level_x, level_y)
    ]


# ============================================================
# 7) Probability Ratios with bootstrap confidence intervals
# ============================================================

def _prob_ratio_bootstrap(
    A: np.ndarray,
    B: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    N = len(A)
    vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        As = A[idx]
        Bs = B[idx]

        pA = As.mean()
        pB = Bs.mean()
        pAB = (As & Bs).mean()

        if pA > 0 and pB > 0:
            vals.append(pAB / (pA * pB))

    if not vals:
        return np.nan, np.nan, np.nan

    return (
        float(np.mean(vals)),
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    )


def probability_ratios_for_target(
    df: pd.DataFrame,
    target_col: str,
    covar_cols: List[str],
    max_levels: int = 20,
    n_boot: int = 1000,
) -> pd.DataFrame:
    rows = []

    for cov in covar_cols:
        series = df[cov]
        unique = list(series.dropna().unique())
        unique_sorted = sorted(unique, key=lambda x: str(x))

        levels = [(str(v), v) for v in unique_sorted[:max_levels]]

        if len(unique_sorted) > max_levels:
            levels.append(("OTHER", None))

        targets = list(df[target_col].dropna().unique())

        for target in targets:
            A = (df[target_col] == target).astype(int).to_numpy()

            for group_name, val in levels:
                if val is None:
                    explicit_vals = [v for _, v in levels[:-1]]
                    B = (~series.isin(explicit_vals)).astype(int).to_numpy()
                else:
                    B = (series == val).astype(int).to_numpy()

                mean, lo, hi = _prob_ratio_bootstrap(
                    A,
                    B,
                    n_boot=n_boot,
                    seed=42,
                )

                rows.append({
                    "Covariate": cov,
                    "Group": group_name,
                    "Target": str(target),
                    "Ratio": round(mean, 4) if np.isfinite(mean) else np.nan,
                    "CI_low": round(lo, 4) if np.isfinite(lo) else np.nan,
                    "CI_high": round(hi, 4) if np.isfinite(hi) else np.nan,
                    "Association": (
                        "Positive"
                        if np.isfinite(mean) and mean > 1
                        else "Negative"
                        if np.isfinite(mean) and mean < 1
                        else "Independent/≈1"
                        if np.isfinite(mean)
                        else ""
                    ),
                })

    return pd.DataFrame(rows)


# ============================================================
# 8) Streamlit UI helpers
# ============================================================

def build_scale_ui_for_items(items: List[str]):
    tfn_maps = {}
    levels_by_item = {}

    for it in items:
        sc_choice = st.sidebar.selectbox(
            f"Scale for {it}",
            [
                "Likert1-4",
                "Likert1-5",
                "Likert1-6",
                "Likert1-7",
                "Likert1-10",
                "Likert1-11",
                "Linear custom",
                "Manual TFN",
            ],
            key=f"sc_{it}",
        )

        if sc_choice == "Likert1-4":
            mapping = likert_map_1_4()

        elif sc_choice == "Likert1-5":
            mapping = likert_map_1_5()

        elif sc_choice == "Likert1-6":
            mapping = likert_map_1_6()

        elif sc_choice == "Likert1-7":
            mapping = likert_map_1_7()

        elif sc_choice == "Likert1-10":
            mapping = likert_map_1_10()

        elif sc_choice == "Likert1-11":
            mapping = likert_map_1_11()

        elif sc_choice == "Linear custom":
            txt = st.sidebar.text_input(
                f"Levels for {it}",
                value="1,2,3,4",
                key=f"lv_{it}",
            )
            levels = [int(x.strip()) for x in txt.split(",") if x.strip()]
            mapping = linear_tfn_map(levels)

        else:
            txt = st.sidebar.text_input(
                f"Levels for {it}",
                value="1,2,3,4",
                key=f"lvman_{it}",
            )
            levels = [int(x.strip()) for x in txt.split(",") if x.strip()]
            mapping = {}

            for lv in levels:
                abc = st.sidebar.text_input(
                    f"{it} — level {lv} TFN",
                    value="0,0,25",
                    key=f"tfn_{it}_{lv}",
                )
                a, b, c = [float(x.strip()) for x in abc.split(",")]
                mapping[lv] = TFN(a, b, c)

        tfn_maps[it] = mapping
        levels_by_item[it] = list(mapping.keys())

    return tfn_maps, levels_by_item


def extended_labels_ui(name_x: str, name_y: str):
    axis_names = {
        1: "Low",
        2: "Hybrid",
        3: "Intermediate",
        4: "High",
    }

    labels = {}

    for xlevel in range(1, 5):
        for ylevel in range(1, 5):
            default = (
                f"{axis_names[xlevel]} {name_x} | "
                f"{axis_names[ylevel]} {name_y}"
            )
            labels[(xlevel, ylevel)] = st.sidebar.text_input(
                f"{axis_names[xlevel]} X · {axis_names[ylevel]} Y",
                value=default,
                key=f"ext_label_{xlevel}_{ylevel}",
            )

    return labels


# ============================================================
# 9) Streamlit App
# ============================================================

st.set_page_config(
    page_title="Fuzzy-Hybrid TOPSIS Suite",
    page_icon="📊",
    layout="wide",
)

gate()

st.title("Fuzzy-Hybrid TOPSIS · Fuzzy C-Means · Eco-Apostle")
st.caption(
    "Input scales must run from 1 = lower level to K = higher level."
)

with st.expander("Method used in this app", expanded=False):
    st.markdown(
        """
        **Fuzzy-Hybrid TOPSIS**
        1. Convert ordinal responses into triangular fuzzy numbers.
        2. Defuzzify each triangular fuzzy number using `(a + 2b + c) / 4`.
        3. For each item, identify the largest observed value as PIS and the smallest observed value as NIS.
        4. Calculate Euclidean distance to PIS and NIS across all items of the latent dimension.
        5. Calculate `TOPSIS = D- / (D+ + D-)`.

        Every selected item contributes equally. There is no normalization of item weights.

        **Fuzzy C-Means**
        - Three fuzzy clusters are estimated separately for each latent variable.
        - Clustering uses the triangular fuzzy item vectors.
        - Profiles are ordered as Low, Intermediate and High.

        **Probability ratios**
        - `R = P(A∩B) / [P(A)P(B)]`
        - Bootstrap 95% confidence intervals are calculated.
        """
    )

# --------------------------
# Upload
# --------------------------

up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if up is None:
    st.info("Upload a dataset to start.")
    st.stop()

try:
    if up.name.lower().endswith(".csv"):
        df = pd.read_csv(up)
    else:
        df = pd.read_excel(up)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

st.subheader("Uploaded data")
st.write(f"Rows: **{len(df):,}** · Columns: **{len(df.columns):,}**")
st.dataframe(df.head(), use_container_width=True)

all_cols = list(df.columns)

# --------------------------
# Latent variables
# --------------------------

st.sidebar.header("1 · Latent variables")

lname_x = st.sidebar.text_input("Latent X name", value="Latent X")
items_x = st.sidebar.multiselect("Items for X", all_cols, key="items_x_widget")

lname_y = st.sidebar.text_input("Latent Y name", value="Latent Y")
items_y = st.sidebar.multiselect("Items for Y", all_cols, key="items_y_widget")

overlap = sorted(set(items_x) & set(items_y))
if overlap:
    st.sidebar.warning(
        "Some items are included in both latent variables: "
        + ", ".join(overlap)
    )

selected_unique = list(dict.fromkeys(items_x + items_y))

# --------------------------
# Scales
# --------------------------

st.sidebar.header("2 · TFN scale per item")

if selected_unique:
    tfn_map_by_item, levels_by_item = build_scale_ui_for_items(selected_unique)
else:
    tfn_map_by_item, levels_by_item = {}, {}

# --------------------------
# Classic quadrants
# --------------------------

st.sidebar.header("3 · Classic 4 quadrants")

thr_x = st.sidebar.slider(
    f"{lname_x} threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
)

thr_y = st.sidebar.slider(
    f"{lname_y} threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
)

q4_labels = {
    "HH": st.sidebar.text_input(
        "High X · High Y",
        value="High X | High Y",
        key="q4_hh",
    ),
    "HL": st.sidebar.text_input(
        "High X · Low Y",
        value="High X | Low Y",
        key="q4_hl",
    ),
    "LH": st.sidebar.text_input(
        "Low X · High Y",
        value="Low X | High Y",
        key="q4_lh",
    ),
    "LL": st.sidebar.text_input(
        "Low X · Low Y",
        value="Low X | Low Y",
        key="q4_ll",
    ),
}

# --------------------------
# Extended 16 names
# --------------------------

st.sidebar.header("4 · Extended 16 quadrants")

with st.sidebar.expander("Name all 16 extended classes", expanded=False):
    ext_labels = extended_labels_ui(lname_x, lname_y)

# --------------------------
# Fuzzy C-Means advanced settings
# --------------------------

st.sidebar.header("5 · Fuzzy C-Means")

with st.sidebar.expander("Advanced clustering settings", expanded=False):
    m_fuzzy = st.slider(
        "Fuzzifier m",
        min_value=1.1,
        max_value=3.0,
        value=2.0,
        step=0.1,
    )

    w1_extremes = st.slider(
        "w1 — TFN extremes",
        min_value=0.50,
        max_value=1.00,
        value=0.70,
        step=0.05,
    )

    w2_centers = 1.0 - w1_extremes
    st.write(f"w2 — TFN centres = **{w2_centers:.2f}**")

    alpha = st.slider(
        "Extended Eco-Apostle α",
        min_value=0.30,
        max_value=0.80,
        value=0.50,
        step=0.05,
    )

# --------------------------
# Grouping and probability ratios
# --------------------------

st.sidebar.header("6 · Analysis variables")

group_cols = st.sidebar.multiselect(
    "Group TOPSIS and group PIS/NIS by",
    all_cols,
    key="group_cols_widget",
)

ratio_cols = st.sidebar.multiselect(
    "Covariates for Probability Ratios",
    all_cols,
    key="ratio_cols_widget",
)

n_boot = st.sidebar.number_input(
    "Bootstrap replications",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
)

# ============================================================
# Run
# ============================================================

run = st.button("Run analysis", type="primary")

if run:
    if not items_x or not items_y:
        st.error("Select at least one item for both Latent X and Latent Y.")
        st.stop()

    valid, problems = validate_items(
        df,
        selected_unique,
        levels_by_item,
    )

    if not valid:
        st.error(
            "Latent-item validation failed. "
            "Missing or invalid values must be corrected before the analysis."
        )

        for p in problems:
            st.write(f"- {p}")

        st.stop()

    # Fuzzy matrices
    X_tfn = df_to_tfn_array(df, items_x, tfn_map_by_item)
    Y_tfn = df_to_tfn_array(df, items_y, tfn_map_by_item)

    # TOPSIS
    topsis_x = fuzzy_hybrid_topsis(
        X_tfn,
        is_benefit=[True] * len(items_x),
    )

    topsis_y = fuzzy_hybrid_topsis(
        Y_tfn,
        is_benefit=[True] * len(items_y),
    )

    x = topsis_x["cc"]
    y = topsis_y["cc"]

    # Individual PIS/NIS audit
    audit_x = individual_pis_nis_audit(
        df,
        items_x,
        tfn_map_by_item,
        lname_x,
    )

    audit_y = individual_pis_nis_audit(
        df,
        items_y,
        tfn_map_by_item,
        lname_y,
    )

    pis_nis_audit = pd.concat(
        [audit_x, audit_y],
        ignore_index=True,
    )

    # Fuzzy C-Means
    fcm_x = fuzzy_cmeans_fuzzy_data(
        X_tfn,
        n_clusters=3,
        m=m_fuzzy,
        w1_extremes=w1_extremes,
        w2_centers=w2_centers,
        seed=42,
    )

    fcm_y = fuzzy_cmeans_fuzzy_data(
        Y_tfn,
        n_clusters=3,
        m=m_fuzzy,
        w1_extremes=w1_extremes,
        w2_centers=w2_centers,
        seed=84,
    )

    fcm_x = order_fcm_low_intermediate_high(fcm_x)
    fcm_y = order_fcm_low_intermediate_high(fcm_y)

    Ux = fcm_x["U_ordered"]
    Uy = fcm_y["U_ordered"]

    level_x = eco_axis_level(Ux, alpha=alpha)
    level_y = eco_axis_level(Uy, alpha=alpha)

    # Named classifications
    classic = classic_quadrant_labels(
        x,
        y,
        thr_x=thr_x,
        thr_y=thr_y,
        labels=q4_labels,
    )

    extended = extended_16_labels(
        level_x,
        level_y,
        labels_16=ext_labels,
    )

    profile_names = np.array(["Low", "Intermediate", "High"])

    dominant_x = profile_names[np.argmax(Ux, axis=1)]
    dominant_y = profile_names[np.argmax(Uy, axis=1)]

    res = pd.DataFrame({
        lname_x: x,
        lname_y: y,
        "ClassicQuadrant": classic,
        "Extended4x4": extended,
        f"{lname_x}_FCM_Dominant": dominant_x,
        f"{lname_y}_FCM_Dominant": dominant_y,
        f"{lname_x}_FCM_Low": Ux[:, 0],
        f"{lname_x}_FCM_Intermediate": Ux[:, 1],
        f"{lname_x}_FCM_High": Ux[:, 2],
        f"{lname_y}_FCM_Low": Uy[:, 0],
        f"{lname_y}_FCM_Intermediate": Uy[:, 1],
        f"{lname_y}_FCM_High": Uy[:, 2],
    })

    full = pd.concat(
        [df.reset_index(drop=True), res.reset_index(drop=True)],
        axis=1,
    )

    # Group outputs
    if group_cols:
        group_topsis = unified_group_topsis_table(
            df,
            items_x,
            items_y,
            tfn_map_by_item,
            group_cols,
            lname_x,
            lname_y,
        )

        gp_x = group_pis_nis_table(
            df,
            items_x,
            tfn_map_by_item,
            group_cols,
            lname_x,
        )

        gp_y = group_pis_nis_table(
            df,
            items_y,
            tfn_map_by_item,
            group_cols,
            lname_y,
        )

        group_pis_nis = pd.concat(
            [gp_x, gp_y],
            ignore_index=True,
        )
    else:
        group_topsis = pd.DataFrame()
        group_pis_nis = pd.DataFrame()

    # Save
    st.session_state["analysis_done"] = True
    st.session_state["df_input"] = df
    st.session_state["full_results"] = full
    st.session_state["res"] = res
    st.session_state["pis_nis_audit"] = pis_nis_audit
    st.session_state["group_topsis"] = group_topsis
    st.session_state["group_pis_nis"] = group_pis_nis
    st.session_state["fcm_x"] = fcm_x
    st.session_state["fcm_y"] = fcm_y
    st.session_state["items_x"] = items_x
    st.session_state["items_y"] = items_y
    st.session_state["lname_x"] = lname_x
    st.session_state["lname_y"] = lname_y
    st.session_state["group_cols"] = group_cols
    st.session_state["ratio_cols"] = ratio_cols
    st.session_state["n_boot"] = int(n_boot)

    st.success("Analysis completed.")


# ============================================================
# Results
# ============================================================

if st.session_state.get("analysis_done", False):

    df_input = st.session_state["df_input"]
    full = st.session_state["full_results"]
    res = st.session_state["res"]
    pis_nis_audit = st.session_state["pis_nis_audit"]
    group_topsis = st.session_state["group_topsis"]
    group_pis_nis = st.session_state["group_pis_nis"]
    fcm_x = st.session_state["fcm_x"]
    fcm_y = st.session_state["fcm_y"]
    items_x = st.session_state["items_x"]
    items_y = st.session_state["items_y"]
    lname_x = st.session_state["lname_x"]
    lname_y = st.session_state["lname_y"]
    group_cols = st.session_state["group_cols"]
    ratio_cols = st.session_state["ratio_cols"]
    n_boot = st.session_state["n_boot"]

    # --------------------------------------------------------
    # 1. Individual TOPSIS
    # --------------------------------------------------------

    st.header("📊 1. Individual Fuzzy-Hybrid TOPSIS")

    c1, c2 = st.columns(2)

    c1.metric(
        f"Mean {lname_x}",
        f"{res[lname_x].mean():.4f}",
    )

    c2.metric(
        f"Mean {lname_y}",
        f"{res[lname_y].mean():.4f}",
    )

    st.dataframe(
        full,
        use_container_width=True,
        height=430,
    )

    st.download_button(
        "⬇️ Download individual results",
        data=full.to_csv(index=False).encode("utf-8"),
        file_name="topsis_individual.csv",
        mime="text/csv",
    )

    # --------------------------------------------------------
    # 2. Individual PIS/NIS audit
    # --------------------------------------------------------

    st.header("🌐 2. Individual PIS/NIS audit")

    st.caption(
        "For each item, PIS is the largest observed defuzzified value "
        "and NIS is the smallest observed defuzzified value across all firms."
    )

    st.dataframe(
        pis_nis_audit,
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Download individual PIS/NIS audit",
        data=pis_nis_audit.to_csv(index=False).encode("utf-8"),
        file_name="pis_nis_audit.csv",
        mime="text/csv",
    )

    # --------------------------------------------------------
    # 3. Fuzzy C-Means
    # --------------------------------------------------------

    st.header("🔺 3. Fuzzy C-Means")

    fcm_metrics = pd.DataFrame([
        {
            "Latent": lname_x,
            "Iterations": fcm_x["iterations"],
            "FPC": fcm_x["fpc"],
            "PartitionEntropy": fcm_x["partition_entropy"],
            "LowProfileScore": fcm_x["profile_scores_ordered"][0],
            "IntermediateProfileScore": fcm_x["profile_scores_ordered"][1],
            "HighProfileScore": fcm_x["profile_scores_ordered"][2],
        },
        {
            "Latent": lname_y,
            "Iterations": fcm_y["iterations"],
            "FPC": fcm_y["fpc"],
            "PartitionEntropy": fcm_y["partition_entropy"],
            "LowProfileScore": fcm_y["profile_scores_ordered"][0],
            "IntermediateProfileScore": fcm_y["profile_scores_ordered"][1],
            "HighProfileScore": fcm_y["profile_scores_ordered"][2],
        },
    ])

    st.dataframe(
        fcm_metrics,
        use_container_width=True,
    )

    centers = pd.concat([
        fcm_centers_table(
            fcm_x,
            items_x,
            lname_x,
        ),
        fcm_centers_table(
            fcm_y,
            items_y,
            lname_y,
        ),
    ], ignore_index=True)

    st.subheader("Fuzzy cluster profiles")

    st.dataframe(
        centers,
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Download fuzzy cluster profiles",
        data=centers.to_csv(index=False).encode("utf-8"),
        file_name="fuzzy_cluster_profiles.csv",
        mime="text/csv",
    )

    # --------------------------------------------------------
    # 4. Classic quadrants
    # --------------------------------------------------------

    st.header("◼️ 4. Classic 4-quadrant model")

    classic_counts = (
        res["ClassicQuadrant"]
        .value_counts(dropna=False)
        .rename_axis("Quadrant")
        .reset_index(name="N")
    )

    classic_counts["Percent"] = (
        100.0 * classic_counts["N"] / len(res)
    )

    st.dataframe(
        classic_counts,
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Download classic quadrants",
        data=classic_counts.to_csv(index=False).encode("utf-8"),
        file_name="classic_quadrants.csv",
        mime="text/csv",
    )

    # --------------------------------------------------------
    # 5. Extended 16 categories
    # --------------------------------------------------------

    st.header("🧩 5. Extended Eco-Apostle 4×4")

    ext_counts = (
        res["Extended4x4"]
        .value_counts(dropna=False)
        .rename_axis("ExtendedClass")
        .reset_index(name="N")
    )

    ext_counts["Percent"] = (
        100.0 * ext_counts["N"] / len(res)
    )

    st.dataframe(
        ext_counts,
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Download extended model",
        data=ext_counts.to_csv(index=False).encode("utf-8"),
        file_name="eco_extended_apostle_model.csv",
        mime="text/csv",
    )

    # --------------------------------------------------------
    # 6. Group TOPSIS
    # --------------------------------------------------------

    st.header("🧮 6. Group TOPSIS")

    if group_cols and not group_topsis.empty:
        st.dataframe(
            group_topsis,
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Download Group TOPSIS",
            data=group_topsis.to_csv(index=False).encode("utf-8"),
            file_name="topsis_group.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "Select one or more grouping variables in the sidebar."
        )

    # --------------------------------------------------------
    # 7. Group PIS/NIS
    # --------------------------------------------------------

    st.header("🎯 7. Group PIS/NIS")

    st.caption(
        "For each selected grouping variable and each latent item, "
        "the group PIS is the largest group mean and the group NIS "
        "is the smallest group mean."
    )

    if group_cols and not group_pis_nis.empty:
        st.dataframe(
            group_pis_nis,
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Download Group PIS/NIS",
            data=group_pis_nis.to_csv(index=False).encode("utf-8"),
            file_name="group_pis_nis.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "Select one or more grouping variables in the sidebar."
        )

    # --------------------------------------------------------
    # 8. Probability Ratios — Classic
    # --------------------------------------------------------

    st.header("📈 8. Probability Ratios — Classic quadrants")

    if ratio_cols:
        ratios_classic = probability_ratios_for_target(
            full,
            "ClassicQuadrant",
            ratio_cols,
            max_levels=20,
            n_boot=n_boot,
        )

        st.dataframe(
            ratios_classic,
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Download Classic Probability Ratios",
            data=ratios_classic.to_csv(index=False).encode("utf-8"),
            file_name="probability_ratios_classic.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "Select covariates for Probability Ratios in the sidebar."
        )

    # --------------------------------------------------------
    # 9. Probability Ratios — Extended
    # --------------------------------------------------------

    st.header("📈 9. Probability Ratios — Extended classes")

    if ratio_cols:
        ratios_extended = probability_ratios_for_target(
            full,
            "Extended4x4",
            ratio_cols,
            max_levels=20,
            n_boot=n_boot,
        )

        st.dataframe(
            ratios_extended,
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Download Extended Probability Ratios",
            data=ratios_extended.to_csv(index=False).encode("utf-8"),
            file_name="probability_ratios_extended.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "Select covariates for Probability Ratios in the sidebar."
        )

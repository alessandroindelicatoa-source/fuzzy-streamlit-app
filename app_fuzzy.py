# app_fuzzy.py
# -*- coding: utf-8 -*-
import math
from io import BytesIO
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =========================================
# Triangular Fuzzy Numbers (TFN) + utilities
# =========================================
@dataclass(frozen=True)
class TFN:
    a: float
    b: float
    c: float
    def __post_init__(self):
        if not (self.a <= self.b <= self.c):
            raise ValueError("TFN requires a ≤ b ≤ c.")
    def scale(self, w: float) -> "TFN":
        return TFN(self.a*w, self.b*w, self.c*w)

def _ensure_tuple_abc(x) -> Tuple[float,float,float]:
    if isinstance(x, TFN): return (x.a, x.b, x.c)
    a,b,c = x
    return float(a), float(b), float(c)

# =========================================
# Fuzzy-Hybrid TOPSIS core
# =========================================
def _normalize_fuzzy_matrix(matrix: List[List[TFN]], is_benefit: List[bool]) -> List[List[TFN]]:
    m = len(matrix); n = len(matrix[0])
    c_max = [max(matrix[i][j].c for i in range(m)) for j in range(n)]
    a_min = [min(matrix[i][j].a for i in range(m)) for j in range(n)]
    out: List[List[TFN]] = []
    for i in range(m):
        row: List[TFN] = []
        for j in range(n):
            x = matrix[i][j]
            if is_benefit[j]:
                denom = c_max[j] if c_max[j] != 0 else 1.0
                row.append(TFN(x.a/denom, x.b/denom, x.c/denom))
            else:
                amin = a_min[j] if a_min[j] != 0 else 1.0
                row.append(TFN(amin/x.c, amin/(x.b if x.b!=0 else 1e-9), amin/(x.a if x.a!=0 else 1e-9)))
        out.append(row)
    return out

def _apply_weights(matrix: List[List[TFN]], weights: List[float]) -> List[List[TFN]]:
    m, n = len(matrix), len(matrix[0])
    wsum = sum(weights)
    if wsum <= 0:
        raise ValueError("Weights must sum > 0.")
    w = [wi/wsum for wi in weights]
    return [[matrix[i][j].scale(w[j]) for j in range(n)] for i in range(m)]

def _fuzzy_distance(x: TFN, y: TFN) -> float:
    return math.sqrt((x.a - y.a)**2 + (x.b - y.b)**2 + (x.c - y.c)**2)

def fuzzy_topsis_cc(matrix: List[List[TFN]], is_benefit: List[bool], weights: Optional[List[float]] = None) -> np.ndarray:
    """Return closeness coefficients in [0,1] for each row (alternative)."""
    m = len(matrix); n = len(matrix[0])
    if weights is None:
        weights = [1.0/n]*n
    norm = _normalize_fuzzy_matrix(matrix, is_benefit)
    vw = _apply_weights(norm, weights)
    fpis: List[TFN] = []; fnis: List[TFN] = []
    for j in range(n):
        col = [vw[i][j] for i in range(m)]
        fpis.append(TFN(max(x.a for x in col), max(x.b for x in col), max(x.c for x in col)))
        fnis.append(TFN(min(x.a for x in col), min(x.b for x in col), min(x.c for x in col)))
    d_plus = np.zeros(m); d_minus = np.zeros(m)
    for i in range(m):
        for j in range(n):
            d_plus[i]  += _fuzzy_distance(vw[i][j], fpis[j])**2
            d_minus[i] += _fuzzy_distance(vw[i][j], fnis[j])**2
        d_plus[i]  = math.sqrt(d_plus[i])
        d_minus[i] = math.sqrt(d_minus[i])
    cc = d_minus / (d_plus + d_minus + 1e-12)
    return np.clip(cc, 0, 1)

# Build fuzzy matrix when EACH item has its own TFN mapping
def df_to_tfn_matrix_per_item(
    df: pd.DataFrame,
    cols: List[str],
    tfn_map_by_item: Dict[str, Dict[int, TFN]],
    levels_by_item: Dict[str, List[int]]
) -> List[List[TFN]]:
    m = df.shape[0]
    mat: List[List[TFN]] = []
    for i in range(m):
        row: List[TFN] = []
        for c in cols:
            if c not in tfn_map_by_item or c not in levels_by_item:
                raise ValueError(f"Missing TFN mapping for column '{c}'.")
            v = df.iloc[i][c]
            if pd.isna(v):
                raise ValueError(f"NaN at row {i}, column '{c}'. Please impute/remove NA.")
            iv = int(v)
            if iv not in set(levels_by_item[c]):
                raise ValueError(f"Column '{c}' value {v} not in its scale {sorted(levels_by_item[c])}.")
            row.append(tfn_map_by_item[c][iv])
        mat.append(row)
    return mat

# =========================================
# ECO-Extended 4×4 fuzzy sets & labels
# =========================================
def eco_fuzzy_sets_4(val: float) -> Tuple[float,float,float,float]:
    low, medlow, medhigh, high = 0,0,0,0
    if val <= 0.33: low = 1 - val/0.33
    if val >= 0.66: high = (val-0.66)/0.34 if val <=1 else 1
    if 0 <= val <= 0.66: medlow = 1 - abs(val-0.33)/0.33
    if 0.33 <= val <= 1: medhigh = 1 - abs(val-0.66)/0.34
    return (max(low,0), max(medlow,0), max(medhigh,0), max(high,0))

def eco_extended_labels_4x4(x: np.ndarray, y: np.ndarray,
                            x_names=("LowX","MedLowX","MedHighX","HighX"),
                            y_names=("LowY","MedLowY","MedHighY","HighY")) -> List[str]:
    labels = []
    for xi, yi in zip(x,y):
        lx = np.array(eco_fuzzy_sets_4(xi))
        ly = np.array(eco_fuzzy_sets_4(yi))
        cx = x_names[lx.argmax()]
        cy = y_names[ly.argmax()]
        labels.append(f"{cx}|{cy}")
    return labels

# =========================================
# Classic Apostle (2×2)
# =========================================
def apostle_quadrants(x: np.ndarray, y: np.ndarray, x_thr: float, y_thr: float,
                      q_AA: str, q_AB: str, q_BA: str, q_BB: str) -> List[str]:
    out = []
    for xi, yi in zip(x, y):
        if   xi >= x_thr and yi >= y_thr: out.append(q_AA)
        elif xi >= x_thr and yi <  y_thr: out.append(q_AB)
        elif xi <  x_thr and yi >= y_thr: out.append(q_BA)
        else:                              out.append(q_BB)
    return out

# =========================================
# Streamlit UI
# =========================================
st.set_page_config(page_title="Fuzzy-Hybrid TOPSIS + ECO-Extended 4×4", layout="wide")
st.title("Latent Variables via Fuzzy-Hybrid TOPSIS → Apostle Classic (2×2) & ECO-Extended (4×4)")

# ---- Upload data ----
df = None
up = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
sep = st.text_input("CSV separator (if CSV)", value=",")
sheet = st.text_input("Excel sheet name (optional)", value="")
if up is not None:
    try:
        if up.name.lower().endswith(".csv"):
            df = pd.read_csv(up, sep=sep)
        else:
            df = pd.read_excel(up, sheet_name=sheet if sheet else 0)
        st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns.")
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Read error: {e}")

# ---- Step 1: choose items for the TWO latents ----
st.header("1) Choose items for the TWO latent variables")
latents: Dict[str, List[str]] = {}
is_benefit_by: Dict[str, List[bool]] = {}
weights_by: Dict[str, List[float]] = {}
selected_items_all: List[str] = []

if df is not None and len(df.columns) > 0:
    all_cols = list(df.columns)

    cL, cR = st.columns(2)
    with cL:
        lname_x = st.text_input("Latent X name", value="LatX")
        items_x = st.multiselect(f"Items for {lname_x}", all_cols, key="items_x")
    with cR:
        lname_y = st.text_input("Latent Y name", value="LatY")
        items_y = st.multiselect(f"Items for {lname_y}", all_cols, key="items_y")

    # Per-item benefit/cost + weight
    def _bcw(side_name: str, items: List[str], key_prefix: str):
        bc_list, w_list = [], []
        for c in items:
            c1, c2 = st.columns(2)
            with c1:
                bc_list.append(st.selectbox(f"{side_name} — {c}: Benefit/Cost",
                                            ["Benefit","Cost"], key=f"bc_{key_prefix}_{c}") == "Benefit")
            with c2:
                w_list.append(float(st.number_input(f"{side_name} — {c}: Weight",
                                                    value=1.0, step=0.1, key=f"w_{key_prefix}_{c}")))
        return bc_list, w_list

    if items_x:
        bc_x, w_x = _bcw(lname_x, items_x, "x")
        is_benefit_by[lname_x] = bc_x
        weights_by[lname_x] = w_x
        latents[lname_x] = items_x
    if items_y:
        bc_y, w_y = _bcw(lname_y, items_y, "y")
        is_benefit_by[lname_y] = bc_y
        weights_by[lname_y] = w_y
        latents[lname_y] = items_y

    selected_items_all = list(dict.fromkeys((items_x or []) + (items_y or [])))  # unique order

# ---- Step 2: for each selected item, define its OWN scale and TFNs ----
st.header("2) Define per-item TFN conversion (manual, point-by-point)")
st.caption("For every selected item, specify its discrete scale points (e.g., 1–10) and a triangular fuzzy number (a,b,c) for each point.")

# these two hold the definitions
tfn_map_by_item: Dict[str, Dict[int, TFN]] = {}
levels_by_item: Dict[str, List[int]] = {}

if selected_items_all:
    for it in selected_items_all:
        with st.expander(f"TFN mapping for item: {it}", expanded=False):
            lvtxt = st.text_input(f"Levels for {it} (comma-separated integers)", value="1,2,3,4,5", key=f"levels_{it}")
            lvls: List[int] = []
            try:
                lvls = sorted(list({int(x.strip()) for x in lvtxt.split(",") if x.strip() != ""}))
            except:
                st.warning(f"Invalid levels for {it}. Please use integers like 1,2,3,...")
            levels_by_item[it] = lvls

            tmap: Dict[int, TFN] = {}
            for lv in lvls:
                abctxt = st.text_input(f"{it} — level {lv} TFN (a,b,c)", value="0,0,25", key=f"tfn_{it}_{lv}")
                try:
                    a,b,c = [float(x.strip()) for x in abctxt.split(",")]
                    tmap[lv] = TFN(a,b,c)
                except:
                    st.warning(f"Invalid (a,b,c) for {it} level {lv}.")
            if tmap:
                tfn_map_by_item[it] = tmap

# ---- Step 3: custom quadrant names & ECO labels ----
st.header("3) Quadrant naming")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Classic (2×2)")
    AA = st.text_input("x≥Xthr & y≥Ythr", value="Apostles")
    AB = st.text_input("x≥Xthr & y<Ythr",  value="Mercenaries")
    BA = st.text_input("x<Xthr & y≥Ythr",  value="Hostages")
    BB = st.text_input("x<Xthr & y<Ythr",  value="Defectors")
with c2:
    st.subheader("ECO-Extended (4×4) axis labels")
    x_names = [
        st.text_input("X Low", value="LowX"),
        st.text_input("X MedLow", value="MedLowX"),
        st.text_input("X MedHigh", value="MedHighX"),
        st.text_input("X High", value="HighX"),
    ]
    y_names = [
        st.text_input("Y Low", value="LowY"),
        st.text_input("Y MedLow", value="MedLowY"),
        st.text_input("Y MedHigh", value="MedHighY"),
        st.text_input("Y High", value="HighY"),
    ]

st.subheader("Optional: rename the 16 ECO cells")
use_custom_16 = st.checkbox("Manually name the 16 cells", value=False)
custom16: Dict[Tuple[int,int], str] = {}
if use_custom_16:
    for iy in range(4):
        row_cols = st.columns(4)
        for ix in range(4):
            default_name = f"{x_names[ix]}|{y_names[iy]}"
            custom16[(ix,iy)] = row_cols[ix].text_input(f"Cell X{ix+1}-Y{iy+1}", value=default_name, key=f"cell_{ix}_{iy}")

# ---- Step 4: clustering/thresholds & aggregation columns ----
st.header("4) Thresholds & grouping")
thr_mode = st.radio("Classic thresholds (computed from indices)", ["Mean", "Median"], index=0, horizontal=True)
agg_cols_indep: List[str] = []
if df is not None:
    agg_cols_indep = st.multiselect("Aggregate Fuzzy-Hybrid TOPSIS independently by these categorical columns",
                                    options=list(df.columns), key="agg_cols_indep")

# ---- Step 5: RUN ----
run_btn = st.button("▶ Run analysis")

if run_btn:
    try:
        if df is None:
            st.error("Please upload a dataset first."); st.stop()
        if len(latents) != 2:
            st.error("Please choose items for TWO latent variables (X and Y)."); st.stop()
        # Verify every selected item has TFNs defined
        for it in selected_items_all:
            if it not in tfn_map_by_item or it not in levels_by_item or not levels_by_item[it]:
                st.error(f"Missing TFN mapping for item '{it}'. Expand its panel and define (a,b,c) for each scale point.")
                st.stop()

        # ==== 5a) TOPSIS indices for both latents ====
        idx: Dict[str, np.ndarray] = {}
        for nm, items in latents.items():
            bene = is_benefit_by.get(nm, [True]*len(items))
            w    = weights_by.get(nm, [1.0]*len(items))
            mat  = df_to_tfn_matrix_per_item(df, items, tfn_map_by_item, levels_by_item)
            z    = fuzzy_topsis_cc(mat, is_benefit=bene, weights=w)
            idx[nm] = np.clip(z, 0, 1)

        names = list(latents.keys())
        xnm, ynm = names[0], names[1]
        x, y = idx[xnm], idx[ynm]

        res = pd.DataFrame({f"idx_{xnm}": x, f"idx_{ynm}": y})

        # thresholds
        x_thr = float(np.mean(x)) if thr_mode == "Mean" else float(np.median(x))
        y_thr = float(np.mean(y)) if thr_mode == "Mean" else float(np.median(y))

        # labels
        res["Apostle_Classic"] = apostle_quadrants(x, y, x_thr, y_thr, AA, AB, BA, BB)
        if use_custom_16:
            named = []
            for xi, yi in zip(x, y):
                lx = np.array(eco_fuzzy_sets_4(xi)); ix = int(lx.argmax())
                ly = np.array(eco_fuzzy_sets_4(yi)); iy = int(ly.argmax())
                named.append(custom16.get((ix,iy), f"{x_names[ix]}|{y_names[iy]}"))
            res["Apostle_Extended4x4"] = named
        else:
            res["Apostle_Extended4x4"] = eco_extended_labels_4x4(x, y, x_names, y_names)

        st.header("Results (individual indices & labels)")
        st.dataframe(res, use_container_width=True)
        st.download_button("⬇ Download full results (CSV)",
                           data=res.to_csv(index=False).encode("utf-8"),
                           file_name="fuzzy_apostle_results.csv",
                           mime="text/csv")

        # ==== Plots ====
        st.header("Plots")
        # 2×2
        st.subheader("Apostle Classic (2×2)")
        fig1, ax1 = plt.subplots()
        ax1.scatter(x, y, alpha=0.75)
        ax1.axvline(x_thr, linestyle="--"); ax1.axhline(y_thr, linestyle="--")
        ax1.set_xlabel(f"TOPSIS index ({xnm})"); ax1.set_ylabel(f"TOPSIS index ({ynm})")
        ax1.set_title("Classic 2×2")
        st.pyplot(fig1)
        buf1 = BytesIO(); fig1.savefig(buf1, format="png", dpi=300, bbox_inches="tight")
        st.download_button("⬇ Download 2×2 plot (PNG)", data=buf1.getvalue(),
                           file_name="apostle_classic_2x2.png", mime="image/png")
        plt.close(fig1)

        # 4×4
        st.subheader("ECO-Extended (4×4)")
        fig2, ax2 = plt.subplots()
        ax2.scatter(x, y, alpha=0.75)
        for thr in [0.25, 0.5, 0.75]:
            ax2.axvline(thr, linestyle="--", alpha=0.7)
            ax2.axhline(thr, linestyle="--", alpha=0.7)
        ax2.set_xlabel(f"TOPSIS index ({xnm})"); ax2.set_ylabel(f"TOPSIS index ({ynm})")
        ax2.set_title("ECO-Extended 4×4")
        st.pyplot(fig2)
        buf2 = BytesIO(); fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
        st.download_button("⬇ Download 4×4 plot (PNG)", data=buf2.getvalue(),
                           file_name="eco_extended_4x4.png", mime="image/png")
        plt.close(fig2)

        # ==== Aggregation by selected categorical columns (independent) ====
        st.header("Aggregated Fuzzy-Hybrid TOPSIS by categories (independent)")
        if agg_cols_indep:
            for gcol in agg_cols_indep:
                st.subheader(f"Grouped by {gcol}")
                rows: List[Dict[str, Any]] = []
                for cat, subdf in df.groupby(gcol, dropna=False):
                    row: Dict[str, Any] = {gcol: cat}

                    # recompute indices in subgroup using the SAME per-item mappings
                    gx_items = latents[xnm]; gy_items = latents[ynm]
                    bene_x = is_benefit_by.get(xnm, [True]*len(gx_items))
                    bene_y = is_benefit_by.get(ynm, [True]*len(gy_items))
                    w_x = weights_by.get(xnm, [1.0]*len(gx_items))
                    w_y = weights_by.get(ynm, [1.0]*len(gy_items))

                    z_x = fuzzy_topsis_cc(df_to_tfn_matrix_per_item(subdf, gx_items, tfn_map_by_item, levels_by_item),
                                          is_benefit=bene_x, weights=w_x)
                    z_y = fuzzy_topsis_cc(df_to_tfn_matrix_per_item(subdf, gy_items, tfn_map_by_item, levels_by_item),
                                          is_benefit=bene_y, weights=w_y)

                    gx, gy = float(np.mean(z_x)), float(np.mean(z_y))
                    row[f"idx_{xnm}"] = gx
                    row[f"idx_{ynm}"] = gy

                    # Labels of subgroup centroid with global thresholds
                    if gx >= x_thr and gy >= y_thr:
                        row["Classic_Label"] = AA
                    elif gx >= x_thr and gy < y_thr:
                        row["Classic_Label"] = AB
                    elif gx < x_thr and gy >= y_thr:
                        row["Classic_Label"] = BA
                    else:
                        row["Classic_Label"] = BB

                    lx = np.array(eco_fuzzy_sets_4(gx)); ix = int(lx.argmax())
                    ly = np.array(eco_fuzzy_sets_4(gy)); iy = int(ly.argmax())
                    row["Extended4x4_Label"] = (custom16.get((ix,iy), f"{x_names[ix]}|{y_names[iy]}")
                                                if use_custom_16 else f"{x_names[ix]}|{y_names[iy]}")
                    rows.append(row)

                agg_one = pd.DataFrame(rows)
                st.dataframe(agg_one, use_container_width=True)
                st.download_button(f"⬇ Download aggregated by {gcol} (CSV)",
                                   data=agg_one.to_csv(index=False).encode("utf-8"),
                                   file_name=f"fuzzy_topsis_aggregated_by_{gcol}.csv",
                                   mime="text/csv")
        else:
            st.info("No columns selected for aggregation.")

        # ==== Overall counts by quadrant ====
        st.header("Overall counts by quadrant")

        classic_order = [AA, AB, BA, BB]
        cnt_classic = res["Apostle_Classic"].value_counts().reindex(classic_order, fill_value=0)
        tbl_classic = pd.DataFrame({"count": cnt_classic, "percent": (cnt_classic / max(1, cnt_classic.sum()) * 100).round(2)})
        st.subheader("Classic (2×2)")
        st.dataframe(tbl_classic, use_container_width=True)
        st.download_button("⬇ Download Classic counts (CSV)",
                           data=tbl_classic.to_csv().encode("utf-8"),
                           file_name="overall_counts_classic.csv",
                           mime="text/csv")

        all16 = [f"{x_names[ix]}|{y_names[iy]}" for iy in range(4) for ix in range(4)]
        cnt_ext = res["Apostle_Extended4x4"].value_counts().reindex(all16, fill_value=0)
        tbl_ext = pd.DataFrame({"count": cnt_ext, "percent": (cnt_ext / max(1, cnt_ext.sum()) * 100).round(2)})
        st.subheader("ECO-Extended (4×4)")
        st.dataframe(tbl_ext, use_container_width=True)
        st.download_button("⬇ Download Extended counts (CSV)",
                           data=tbl_ext.to_csv().encode("utf-8"),
                           file_name="overall_counts_extended.csv",
                           mime="text/csv")

    except Exception as e:
        st.error(f"Error: {e}")

# app_fuzzy.py
# -*- coding: utf-8 -*-
import math
from io import BytesIO
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ==============================
# Triangular fuzzy numbers (TFN)
# ==============================
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

# Default Likert mappings
def likert4_default_map() -> Dict[int, TFN]:
    return {1: TFN(0,0,50), 2: TFN(30,50,70), 3: TFN(50,70,90), 4: TFN(70,100,100)}

def likert5_default_map() -> Dict[int, TFN]:
    return {1: TFN(0,0,25), 2: TFN(15,30,45), 3: TFN(40,50,60), 4: TFN(55,70,85), 5: TFN(75,100,100)}

def linear_tfn_map(levels: List[int]) -> Dict[int, TFN]:
    """Linearly spaced TFNs over [0,100] for arbitrary integer levels."""
    K = len(levels)
    if K < 2:
        raise ValueError("Need at least 2 levels for linear TFNs.")
    centers = np.linspace(0, 100, K)
    mapping: Dict[int, TFN] = {}
    for i, lv in enumerate(levels):
        b = float(centers[i])
        if i == 0:
            a = 0.0
            c = float((centers[i] + centers[i+1]) / 2.0)
        elif i == K-1:
            a = float((centers[i-1] + centers[i]) / 2.0)
            c = 100.0
        else:
            a = float((centers[i-1] + centers[i]) / 2.0)
            c = float((centers[i]   + centers[i+1]) / 2.0)
        a = max(0.0, min(a, b))
        c = min(100.0, max(c, b))
        mapping[lv] = TFN(a, b, c)
    return mapping

# ==============================
# Fuzzy-Hybrid TOPSIS
# ==============================
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
    """Return closeness coefficients in [0,1]."""
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

def df_to_tfn_matrix(df: pd.DataFrame, cols: List[str], tfn_map: Dict[int, TFN], levels: List[int]) -> List[List[TFN]]:
    level_set = set(levels)
    for c in cols:
        vals = pd.unique(df[c].dropna())
        if not set(map(int, vals)).issubset(level_set):
            raise ValueError(f"Column '{c}' has values outside scale {sorted(level_set)}.")
    m = df.shape[0]; mat: List[List[TFN]] = []
    for i in range(m):
        row=[]
        for c in cols:
            v = df.iloc[i][c]
            if pd.isna(v):
                raise ValueError(f"NaN at row {i}, column '{c}'. Please impute/remove NA.")
            row.append(tfn_map[int(v)])
        mat.append(row)
    return mat

# ==============================
# ECO-Extended 4×4 fuzzy sets
# ==============================
def eco_fuzzy_sets_4(val: float) -> Tuple[float,float,float,float]:
    """Memberships to (Low, MedLow, MedHigh, High) over [0,1]. Triangular/shoulder MFs."""
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

# ==============================
# Classic Apostle (2×2)
# ==============================
def apostle_quadrants(x: np.ndarray, y: np.ndarray, x_thr: float, y_thr: float,
                      q_AA: str, q_AB: str, q_BA: str, q_BB: str) -> List[str]:
    out = []
    for xi, yi in zip(x, y):
        if   xi >= x_thr and yi >= y_thr: out.append(q_AA)
        elif xi >= x_thr and yi <  y_thr: out.append(q_AB)
        elif xi <  x_thr and yi >= y_thr: out.append(q_BA)
        else:                              out.append(q_BB)
    return out

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Fuzzy-Hybrid TOPSIS + ECO-Extended 4×4", layout="wide")
st.title("Latent Variables with Fuzzy-Hybrid TOPSIS → Apostle Classic (2×2) & ECO-Extended (4×4)")

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

# ---- Scale & TFNs ----
st.header("1) Scale & TFN conversion")
scale_choice = st.selectbox(
    "Scale type",
    ["Likert 1–4 (preset)", "Likert 1–5 (preset)", "Arbitrary (linear TFNs)", "Arbitrary (manual TFNs)"]
)
levels: List[int] = []
tfn_map: Dict[int, TFN] = {}

if scale_choice.startswith("Likert 1–4"):
    levels = [1,2,3,4]; tfn_map = likert4_default_map()
elif scale_choice.startswith("Likert 1–5"):
    levels = [1,2,3,4,5]; tfn_map = likert5_default_map()
else:
    levels_str = st.text_input("Levels (e.g., 1,2,3,...,10)", value="1,2,3,4,5")
    try:
        levels = sorted(list({int(x.strip()) for x in levels_str.split(",") if x.strip() != ""}))
    except:
        levels = []; st.warning("Check the levels format.")
    if levels:
        if "linear" in scale_choice.lower():
            try:
                tfn_map = linear_tfn_map(levels)
            except Exception as e:
                st.error(f"Could not generate linear TFNs: {e}")
        else:
            st.caption("Define manual TFNs (a,b,c) for each level:")
            tfn_map = {}
            for lv in levels:
                abctxt = st.text_input(f"Level {lv} TFN (a,b,c)", value="0,0,25" if lv == levels[0] else "", key=f"tfn_{lv}")
                try:
                    a,b,c = [float(x.strip()) for x in abctxt.split(",")]
                    tfn_map[lv] = TFN(a,b,c)
                except:
                    pass

if tfn_map:
    tfndf = pd.DataFrame([{"level": lv, "a": t.a, "b": t.b, "c": t.c} for lv, t in sorted(tfn_map.items())])
    with st.expander("Show TFN mapping"):
        st.dataframe(tfndf, use_container_width=True)
    st.download_button("⬇ Download TFN mapping (CSV)",
                       data=tfndf.to_csv(index=False).encode("utf-8"),
                       file_name="tfn_mapping.csv", mime="text/csv")

# ---- Latents ----
st.header("2) Select items for the TWO latent variables")
latents: Dict[str, List[str]] = {}
is_benefit_by: Dict[str, List[bool]] = {}
weights_by: Dict[str, List[float]] = {}

if df is not None and len(df.columns) > 0 and levels and tfn_map:
    all_cols = list(df.columns)
    st.info("Pick items for exactly two latents (X and Y).")
    # Latent X
    lname_x = st.text_input("Latent X name", value="LatX")
    sel_x = st.multiselect(f"Items for {lname_x}", all_cols, key="sel_lat_x")
    bc_x, w_x = [], []
    for c in sel_x:
        c1, c2 = st.columns(2)
        with c1:
            bc_x.append(st.selectbox(f"{c}: Benefit/Cost (X)", ["Benefit","Cost"], key=f"bc_x_{c}") == "Benefit")
        with c2:
            w_x.append(st.number_input(f"Weight for {c} (X)", value=1.0, step=0.1, key=f"w_x_{c}"))
    # Latent Y
    lname_y = st.text_input("Latent Y name", value="LatY")
    sel_y = st.multiselect(f"Items for {lname_y}", all_cols, key="sel_lat_y")
    bc_y, w_y = [], []
    for c in sel_y:
        c1, c2 = st.columns(2)
        with c1:
            bc_y.append(st.selectbox(f"{c}: Benefit/Cost (Y)", ["Benefit","Cost"], key=f"bc_y_{c}") == "Benefit")
        with c2:
            w_y.append(st.number_input(f"Weight for {c} (Y)", value=1.0, step=0.1, key=f"w_y_{c}"))
    if sel_x and sel_y:
        latents = {lname_x: sel_x, lname_y: sel_y}
        is_benefit_by = {lname_x: bc_x, lname_y: bc_y}
        weights_by = {lname_x: w_x, lname_y: w_y}

# ---- Manual quadrant naming ----
st.header("3) Manual quadrant naming")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Classic (2×2) names")
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

st.subheader("Optional: custom names for each of the 16 ECO cells")
use_custom_16 = st.checkbox("Manually name the 16 cells", value=False)
custom16: Dict[Tuple[int,int], str] = {}
if use_custom_16:
    for iy in range(4):
        row_cols = st.columns(4)
        for ix in range(4):
            default_name = f"{x_names[ix]}|{y_names[iy]}"
            custom16[(ix,iy)] = row_cols[ix].text_input(f"Cell X{ix+1}-Y{iy+1}", value=default_name, key=f"cell_{ix}_{iy}")

# ---- Run ----
st.header("4) Run Analysis")
thr_mode = st.radio("Classic thresholds", ["Mean", "Median"], horizontal=True, index=0)
run_btn = st.button("▶ Run analysis")

if run_btn:
    try:
        if df is None:
            st.error("Please upload a dataset first.")
            st.stop()
        if not levels or not tfn_map:
            st.error("Please define a scale and TFNs.")
            st.stop()
        if len(latents) != 2:
            st.error("Please choose items for TWO latent variables (X and Y).")
            st.stop()

        # Compute indices with Fuzzy-Hybrid TOPSIS
        idx: Dict[str, np.ndarray] = {}
        for nm, items in latents.items():
            bene = is_benefit_by.get(nm, [True]*len(items))
            w    = weights_by.get(nm, [1.0]*len(items))
            mat  = df_to_tfn_matrix(df, items, tfn_map, levels)
            z    = fuzzy_topsis_cc(mat, is_benefit=bene, weights=w)
            idx[nm] = z

        names = list(latents.keys())
        xnm, ynm = names[0], names[1]
        x, y = idx[xnm], idx[ynm]

        # Latent indices table
        res = pd.DataFrame({f"idx_{xnm}": x, f"idx_{ynm}": y})
        st.subheader("Fuzzy-Hybrid TOPSIS results (latent indices)")
        st.dataframe(res, use_container_width=True)
        st.download_button("⬇ Download latent indices (CSV)",
                           data=res.to_csv(index=False).encode("utf-8"),
                           file_name="latent_indices_fuzzy_topsis.csv",
                           mime="text/csv")

        # Classic thresholds (mean or median)
        x_thr = float(np.mean(x)) if thr_mode == "Mean" else float(np.median(x))
        y_thr = float(np.mean(y)) if thr_mode == "Mean" else float(np.median(y))

        # Classic 2×2 labels (manual names)
        classic = apostle_quadrants(x, y, x_thr, y_thr, AA, AB, BA, BB)
        res["Apostle_Classic"] = classic

        # ECO-Extended 4×4 labels
        base16 = eco_extended_labels_4x4(x, y, x_names, y_names)
        if use_custom_16:
            named = []
            for xi, yi in zip(x, y):
                lx = np.array(eco_fuzzy_sets_4(xi)); ix = int(lx.argmax())
                ly = np.array(eco_fuzzy_sets_4(yi)); iy = int(ly.argmax())
                named.append(custom16.get((ix,iy), f"{x_names[ix]}|{y_names[iy]}"))
            res["Apostle_Extended4x4"] = named
        else:
            res["Apostle_Extended4x4"] = base16

        st.subheader("Final classifications")
        st.dataframe(res, use_container_width=True)
        st.download_button("⬇ Download full results (CSV)",
                           data=res.to_csv(index=False).encode("utf-8"),
                           file_name="fuzzy_apostle_results.csv",
                           mime="text/csv")

        # ---- Plots ----
        st.header("5) Plots")

        # Classic 2×2
        st.subheader("Apostle Classic (2×2)")
        fig1, ax1 = plt.subplots()
        ax1.scatter(x, y, alpha=0.7)      # no explicit colors/styles
        ax1.axvline(x_thr); ax1.axhline(y_thr)
        ax1.set_xlabel(f"TOPSIS index ({xnm})"); ax1.set_ylabel(f"TOPSIS index ({ynm})")
        ax1.set_title("Apostle Classic")
        st.pyplot(fig1)
        buf1 = BytesIO()
        fig1.savefig(buf1, format="png", dpi=300, bbox_inches="tight")
        st.download_button("⬇ Download 2×2 plot (PNG)", data=buf1.getvalue(),
                           file_name="apostle_2x2.png", mime="image/png")
        plt.close(fig1)

        # ECO-Extended 4×4
        st.subheader("ECO-Extended (4×4)")
        fig2, ax2 = plt.subplots()
        ax2.scatter(x, y, alpha=0.7)
        for thr in [0.25, 0.5, 0.75]:
            ax2.axvline(thr, linestyle="--", alpha=0.7)
            ax2.axhline(thr, linestyle="--", alpha=0.7)
        ax2.set_xlabel(f"TOPSIS index ({xnm})"); ax2.set_ylabel(f"TOPSIS index ({ynm})")
        ax2.set_title("ECO-Extended 4×4")
        st.pyplot(fig2)
        buf2 = BytesIO()
        fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
        st.download_button("⬇ Download 4×4 plot (PNG)", data=buf2.getvalue(),
                           file_name="eco_extended_4x4.png", mime="image/png")
        plt.close(fig2)

    except Exception as e:
        st.error(f"Error: {e}")

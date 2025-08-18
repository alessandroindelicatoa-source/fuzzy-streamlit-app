# app_fuzzy.py
# -*- coding: utf-8 -*-
import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ==============================
# Fuzzy numbers and utilities
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

def likert4_default_map() -> Dict[int, TFN]:
    return {1: TFN(0,0,50), 2: TFN(30,50,70), 3: TFN(50,70,90), 4: TFN(70,100,100)}

def likert5_default_map() -> Dict[int, TFN]:
    return {1: TFN(0,0,25), 2: TFN(15,30,45), 3: TFN(40,50,60), 4: TFN(55,70,85), 5: TFN(75,100,100)}

def linear_tfn_map(levels: List[int]) -> Dict[int, TFN]:
    """Generate linearly spaced TFNs over [0,100] for arbitrary integer levels."""
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
                row.append(TFN(amin/x.c, amin/x.b if x.b != 0 else amin/1e-9, amin/x.a if x.a != 0 else amin/1e-9))
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

# ==============================
# Fuzzy C-Means (1D convenience)
# ==============================
def fuzzy_c_means(X: np.ndarray, c: int = 3, m: float = 2.0, max_iter: int = 200, tol: float = 1e-5, seed: int = 42):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    U = rng.uniform(0, 1, size=(c, n))
    U /= U.sum(axis=0, keepdims=True)
    for _ in range(max_iter):
        um = U ** m
        centers = (um @ X) / (um.sum(axis=1, keepdims=True) + 1e-12)
        dist = np.zeros((c, n))
        for k in range(c):
            diff = X - centers[k]
            dist[k] = np.sqrt((diff**2).sum(axis=1))
        dist = np.fmax(dist, 1e-12)
        U_new = 1.0 / (dist ** 2)
        U_new /= U_new.sum(axis=0, keepdims=True)
        if np.linalg.norm(U_new - U) < tol:
            U = U_new
            break
        U = U_new
    return U, centers

# ==============================
# Labels: Apostle & Extended
# ==============================
def apostle_quadrants(x: np.ndarray, y: np.ndarray, x_thr: float, y_thr: float,
                      AA: str, AB: str, BA: str, BB: str) -> List[str]:
    out = []
    for xi, yi in zip(x, y):
        if   xi >= x_thr and yi >= y_thr: out.append(AA)
        elif xi >= x_thr and yi <  y_thr: out.append(AB)
        elif xi <  x_thr and yi >= y_thr: out.append(BA)
        else:                             out.append(BB)
    return out

def extended_3x3_labels(u_x: np.ndarray, u_y: np.ndarray,
                        names_x=("LowX","MidX","HighX"),
                        names_y=("LowY","MidY","HighY")) -> List[str]:
    ix = u_x.argmax(axis=0); iy = u_y.argmax(axis=0)
    mapx = {0:names_x[0], 1:names_x[1], 2:names_x[2]}
    mapy = {0:names_y[0], 1:names_y[1], 2:names_y[2]}
    return [f"{mapx[i]}|{mapy[j]}" for i, j in zip(ix, iy)]

# ==============================
# Pipeline helpers
# ==============================
def df_to_tfn_matrix(df: pd.DataFrame, cols: List[str], tfn_map: Dict[int, TFN], levels: List[int]) -> List[List[TFN]]:
    level_set = set(levels)
    for c in cols:
        vals = pd.unique(df[c].dropna())
        if not set(map(int, vals)).issubset(level_set):
            raise ValueError(f"Column '{c}' has values outside scale {sorted(level_set)}.")
    m = df.shape[0]
    mat: List[List[TFN]] = []
    for i in range(m):
        row: List[TFN] = []
        for c in cols:
            v = df.iloc[i][c]
            if pd.isna(v):
                raise ValueError(f"NaN at row {i}, column '{c}'. Please impute/remove NA.")
            row.append(tfn_map[int(v)])
        mat.append(row)
    return mat

def run_latent_index(df: pd.DataFrame, items: List[str], tfn_map: Dict[int, TFN], levels: List[int],
                     is_benefit: Optional[List[bool]] = None, weights: Optional[List[float]] = None) -> np.ndarray:
    if is_benefit is None:
        is_benefit = [True] * len(items)
    matrix = df_to_tfn_matrix(df, items, tfn_map, levels)
    return fuzzy_topsis_cc(matrix, is_benefit=is_benefit, weights=weights)

def run_pipeline(df: pd.DataFrame,
                 latents: Dict[str, List[str]],
                 tfn_map: Dict[int, TFN], levels: List[int],
                 is_benefit_by_latent: Optional[Dict[str, List[bool]]] = None,
                 weights_by_latent: Optional[Dict[str, List[float]]] = None,
                 fcm_c: int = 3, fcm_m: float = 2.0,
                 thresholds: Tuple[float, float] = (0.5, 0.5),
                 quad_labels: Tuple[str, str, str, str] = ("Apostles","Mercenaries","Hostages","Defectors"),
                 seed: int = 42) -> Dict[str, Any]:

    names = list(latents.keys())
    idx: Dict[str, np.ndarray] = {}
    U_by: Dict[str, np.ndarray] = {}
    C_by: Dict[str, np.ndarray] = {}

    for nm in names:
        items = latents[nm]
        bene = None if (is_benefit_by_latent is None or nm not in is_benefit_by_latent) else is_benefit_by_latent[nm]
        w    = None if (weights_by_latent     is None or nm not in weights_by_latent)     else weights_by_latent[nm]
        z = run_latent_index(df, items, tfn_map, levels, bene, w)
        z = np.clip(z, 0, 1)
        idx[nm] = z
        U, C = fuzzy_c_means(z.reshape(-1,1), c=fcm_c, m=fcm_m, seed=seed)
        U_by[nm] = U
        C_by[nm] = C

    classic = None; extended = None
    if len(names) >= 2:
        xnm, ynm = names[0], names[1]
        AA, AB, BA, BB = quad_labels
        classic = apostle_quadrants(idx[xnm], idx[ynm], thresholds[0], thresholds[1], AA, AB, BA, BB)
        if U_by[xnm].shape[0] == 3 and U_by[ynm].shape[0] == 3:
            def reorder(u: np.ndarray, c: np.ndarray) -> np.ndarray:
                order = np.argsort(c[:,0]); return u[order]
            ux = reorder(U_by[xnm], C_by[xnm])
            uy = reorder(U_by[ynm], C_by[ynm])
            extended = extended_3x3_labels(ux, uy)

    return {"idx": idx, "U": U_by, "C": C_by, "classic": classic, "extended": extended}

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Fuzzy-Hybrid TOPSIS + ECO-Apostle", layout="wide")
st.title("Latent Variables via Fuzzy-Hybrid TOPSIS + ECO-Extended Apostle")

# ---- Upload
central_up = st.file_uploader("Upload CSV/Excel here", type=["csv","xlsx"], key="central_up")

with st.sidebar:
    st.header("1) Data")
    up = st.file_uploader("CSV/Excel (sidebar)", type=["csv","xlsx"], key="sidebar_up")
    sep = st.text_input("CSV separator", value=",")
    sheet = st.text_input("Excel sheet name (optional)", value="")

df = None
try:
    if up is not None:
        if up.name.lower().endswith(".csv"):
            df = pd.read_csv(up, sep=sep)
        else:
            df = pd.read_excel(up, sheet_name=sheet if sheet else 0)
    elif central_up is not None:
        if central_up.name.lower().endswith(".csv"):
            df = pd.read_csv(central_up)
        else:
            df = pd.read_excel(central_up, sheet_name=0)
except Exception as e:
    st.error(f"Read error: {e}")

if df is not None:
    st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols.")
    st.dataframe(df.head(), use_container_width=True)
else:
    st.info("Upload a file to continue.")

# ---- Scale
with st.sidebar:
    st.header("2) Scale & TFNs")
    sc_choice = st.selectbox("Scale type", ["Likert 1–4","Likert 1–5","Arbitrary (linear)","Arbitrary (manual)"])

levels: List[int] = []; tfn_map: Dict[int, TFN] = {}
if sc_choice.startswith("Likert 1–4"): levels=[1,2,3,4]; tfn_map=likert4_default_map()
elif sc_choice.startswith("Likert 1–5"): levels=[1,2,3,4,5]; tfn_map=likert5_default_map()
else:
    levels_str = st.sidebar.text_input("Levels", value="1,2,3,4,5")
    try: levels = sorted(list({int(x.strip()) for x in levels_str.split(",") if x.strip()}))
    except: levels=[]
    if levels:
        if "linear" in sc_choice: tfn_map = linear_tfn_map(levels)
        else:
            for lv in levels:
                abctxt = st.sidebar.text_input(f"Level {lv} TFN", value="", key=f"tfn_{lv}")
                try:
                    a,b,c=[float(x) for x in abctxt.split(",")]
                    tfn_map[lv]=TFN(a,b,c)
                except: pass

# ---- Latents + Group vars
with st.sidebar:
    st.header("3) Latent variables")
    latents: Dict[str,List[str]] = {}
    is_benefit_by: Dict[str,List[bool]] = {}
    weights_by: Dict[str,List[float]] = {}

    group_vars = []
    if df is not None:
        all_cols = list(df.columns)
        n_lat = st.number_input("How many latent vars?",1,6,2)
        for i in range(int(n_lat)):
            lname = st.text_input(f"Latent name #{i+1}", f"Lat{i+1}")
            sel = st.multiselect(f"Items for {lname}", all_cols, key=f"lcols_{i}")
            latents[lname]=sel
            bc_list=[]; w_list=[]
            for c in sel:
                c1,c2=st.columns(2)
                with c1: bc = st.selectbox(f"{c}: type",["Benefit","Cost"],key=f"bc_{i}_{c}"); bc_list.append(bc=="Benefit")
                with c2: w = st.number_input(f"Weight {c}",1.0,10.0,1.0,0.1,key=f"w_{i}_{c}"); w_list.append(w)
            is_benefit_by[lname]=bc_list
            weights_by[lname]=w_list
        group_vars = st.multiselect("Group results by (categorical)", all_cols)

# ---- Clustering & thresholds
with st.sidebar:
    st.header("4) Clustering & thresholds")
    fcm_c = st.number_input("FCM clusters",2,6,3)
    fcm_m = st.number_input("Fuzziness m",1.1,5.0,2.0,0.1)
    x_thr = st.number_input("X threshold",0.0,1.0,0.5,0.01)
    y_thr = st.number_input("Y threshold",0.0,1.0,0.5,0.01)
    st.header("5) Quadrant labels")
    AA = st.text_input("x≥thr & y≥thr", "Apostles")
    AB = st.text_input("x≥thr & y<thr", "Mercenaries")
    BA = st.text_input("x<thr & y≥thr", "Hostages")
    BB = st.text_input("x<thr & y<thr", "Defectors")

run_btn =

# app_fuzzy.py
# -*- coding: utf-8 -*-
import math
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

# ==============================
# Fuzzy-Hybrid TOPSIS Functions
# ==============================
def normalize_fuzzy_matrix(matrix: List[List[TFN]], is_benefit: List[bool]) -> List[List[TFN]]:
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

def apply_weights(matrix: List[List[TFN]], weights: List[float]) -> List[List[TFN]]:
    m, n = len(matrix), len(matrix[0])
    wsum = sum(weights)
    w = [wi/wsum for wi in weights]
    return [[matrix[i][j].scale(w[j]) for j in range(n)] for i in range(m)]

def fuzzy_distance(x: TFN, y: TFN) -> float:
    return math.sqrt((x.a - y.a)**2 + (x.b - y.b)**2 + (x.c - y.c)**2)

def fuzzy_topsis_cc(matrix: List[List[TFN]], is_benefit: List[bool], weights: Optional[List[float]] = None) -> np.ndarray:
    m = len(matrix); n = len(matrix[0])
    if weights is None:
        weights = [1.0/n]*n
    norm = normalize_fuzzy_matrix(matrix, is_benefit)
    vw = apply_weights(norm, weights)
    fpis: List[TFN] = []; fnis: List[TFN] = []
    for j in range(n):
        col = [vw[i][j] for i in range(m)]
        fpis.append(TFN(max(x.a for x in col), max(x.b for x in col), max(x.c for x in col)))
        fnis.append(TFN(min(x.a for x in col), min(x.b for x in col), min(x.c for x in col)))
    d_plus = np.zeros(m); d_minus = np.zeros(m)
    for i in range(m):
        for j in range(n):
            d_plus[i]  += fuzzy_distance(vw[i][j], fpis[j])**2
            d_minus[i] += fuzzy_distance(vw[i][j], fnis[j])**2
        d_plus[i]  = math.sqrt(d_plus[i])
        d_minus[i] = math.sqrt(d_minus[i])
    cc = d_minus / (d_plus + d_minus + 1e-12)
    return np.clip(cc, 0, 1)

def df_to_tfn_matrix(df, cols, tfn_map):
    m = df.shape[0]; mat=[]
    for i in range(m):
        row=[]
        for c in cols: row.append(tfn_map[int(df.iloc[i][c])])
        mat.append(row)
    return mat

# ==============================
# Classification models
# ==============================
def apostle_quadrants(x, y, x_thr, y_thr,
                      AA="Apostles", AB="Mercenaries", BA="Hostages", BB="Defectors"):
    out = []
    for xi, yi in zip(x, y):
        if   xi >= x_thr and yi >= y_thr: out.append(AA)
        elif xi >= x_thr and yi <  y_thr: out.append(AB)
        elif xi <  x_thr and yi >= y_thr: out.append(BA)
        else:                             out.append(BB)
    return out

def eco_fuzzy_sets_4(val: float) -> Tuple[float,float,float,float]:
    low, medlow, medhigh, high = 0,0,0,0
    if val <= 0.33: low = 1 - val/0.33
    if val >= 0.66: high = (val-0.66)/0.34 if val <=1 else 1
    if 0 <= val <= 0.66: medlow = 1 - abs(val-0.33)/0.33
    if 0.33 <= val <= 1: medhigh = 1 - abs(val-0.66)/0.34
    return (max(low,0), max(medlow,0), max(medhigh,0), max(high,0))

def eco_extended_labels_4x4(x: np.ndarray, y: np.ndarray,
                            names_x=("LowX","MedLowX","MedHighX","HighX"),
                            names_y=("LowY","MedLowY","MedHighY","HighY")) -> List[str]:
    labels = []
    for xi, yi in zip(x,y):
        lx = np.array(eco_fuzzy_sets_4(xi))
        ly = np.array(eco_fuzzy_sets_4(yi))
        cx = names_x[lx.argmax()]
        cy = names_y[ly.argmax()]
        labels.append(f"{cx}|{cy}")
    return labels

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Fuzzy-Hybrid TOPSIS + ECO-Apostle", layout="wide")
st.title("Latent Variables via Fuzzy-Hybrid TOPSIS + Apostle Models")

# Upload
df=None
file=st.file_uploader("Upload CSV/XLSX",type=["csv","xlsx"])
if file:
    if file.name.endswith(".csv"): df=pd.read_csv(file)
    else: df=pd.read_excel(file)
if df is not None: st.dataframe(df.head())

# Scale
sc=st.selectbox("Scale type",["Likert 1–4","Likert 1–5"])
tfn_map=likert4_default_map() if sc.startswith("Likert 1–4") else likert5_default_map()

# Latents
latents={}
if df is not None:
    st.subheader("Latent Variable Selection")
    n_lat=2  # we need exactly 2 for Apostle
    for i in range(n_lat):
        name=st.text_input(f"Latent #{i+1} name",f"Lat{i+1}")
        items=st.multiselect(f"Items for {name}",df.columns)
        if items: latents[name]=items

if df is not None and len(latents)==2:
    st.subheader("Run Analysis")
    idx={}
    for nm, items in latents.items():
        mat = df_to_tfn_matrix(df, items, tfn_map)
        z = fuzzy_topsis_cc(mat, is_benefit=[True]*len(items))
        idx[nm]=z
    xnm, ynm = list(latents.keys())
    x, y = idx[xnm], idx[ynm]

    # Apostle Classic (2×2)
    classic = apostle_quadrants(x, y, np.mean(x), np.mean(y))
    df["Apostle_Classic"]=classic

    # ECO-Extended (4×4)
    extended = eco_extended_labels_4x4(x, y)
    df["Apostle_Extended4x4"]=extended

    st.dataframe(df.head())

    # Plots
    st.subheader("Apostle Classic (2×2)")
    fig, ax = plt.subplots()
    ax.scatter(x, y, c="blue", alpha=0.6)
    ax.axvline(np.mean(x), color="red", linestyle="--")
    ax.axhline(np.mean(y), color="red", linestyle="--")
    ax.set_xlabel(xnm); ax.set_ylabel(ynm)
    st.pyplot(fig)

    st.subheader("ECO-Extended (4×4)")
    fig, ax = plt.subplots()
    ax.scatter(x, y, c="green", alpha=0.6)
    for thr in [0.25,0.5,0.75]:
        ax.axvline(thr, color="grey", linestyle="--", alpha=0.7)
        ax.axhline(thr, color="grey", linestyle="--", alpha=0.7)
    ax.set_xlabel(xnm); ax.set_ylabel(ynm)
    st.pyplot(fig)

    # Download
    st.download_button("Download Results", df.to_csv(index=False).encode("utf-8"), "results.csv")

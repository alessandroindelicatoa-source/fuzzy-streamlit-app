# app_fuzzy.py
# -*- coding: utf-8 -*-
"""
Fuzzy-Hybrid TOPSIS Suite (SEPARATED LOGICS)
--------------------------------------------
1) INDIVIDUAL: Fuzzy–Hybrid TOPSIS per person (TFN -> normalize -> weights -> fuzzy PIS/NIS -> distances -> index).
2) GROUP: Crisp TOPSIS per group (Table 4 style):
      • Aggregate TFN by group (mean of a,b,c per item)
      • Defuzzify (Buckley ĉ = (a + 2b + c)/4)
      • PIS_j = max over groups (crisp); NIS_j = min over groups (crisp)
      • Distances & group TOPSIS index
      • PLUS: Ideal table (which group attains PIS/NIS for each item)
3) Classic Apostle (4 quadrants) using FIXED thresholds on indices (no means).
4) Extended Apostle (16 classes) using Fuzzy C-Means (memberships most/least/intermediate + α-rule).
5) Probability ratios using EXACTLY the chosen group columns.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# Fuzzy C-Means (for Extended Apostle ONLY)
import skfuzzy as fuzz

# ==============================
# TFN
# ==============================
@dataclass(frozen=True)
class TFN:
    a: float; b: float; c: float
    def __post_init__(self):
        if not (self.a <= self.b <= self.c):
            raise ValueError("TFN requires a ≤ b ≤ c.")
    def scale(self, w: float) -> "TFN":
        return TFN(self.a*w, self.b*w, self.c*w)

def likert_map_1_4(): return {1: TFN(0,0,20), 2: TFN(16,33,60), 3: TFN(49,66,83), 4: TFN(80,100,100)}
def likert_map_1_5(): return {1: TFN(0,0,30), 2: TFN(20,30,40), 3: TFN(30,50,70), 4: TFN(60,70,80), 5: TFN(70,100,100)}
def likert_map_1_6(): return {1: TFN(0,0,15), 2: TFN(25,40,55), 3: TFN(45,60,75), 4: TFN(70,80,90), 5: TFN(85,100,100), 6: TFN(90,100,100)}
def likert_map_1_7(): return {i: TFN((i-1)*15,(i-1)*15+10,(i-1)*15+20) for i in range(1,8)}
def likert_map_1_10():
    return {
        1:  TFN(0, 0, 10),
        2:  TFN(0, 10, 20),
        3:  TFN(10, 20, 30),
        4:  TFN(20, 30, 40),
        5:  TFN(30, 40, 50),
        6:  TFN(50, 60, 70),
        7:  TFN(60, 70, 80),
        8:  TFN(70, 80, 90),
        9:  TFN(80, 90, 100),
        10: TFN(90, 100, 100)
    }
def likert_map_1_11(): return {i: TFN((i-1)*10,(i-1)*10+10,(i-1)*10+20) for i in range(1,12)}

def linear_tfn_map(levels: List[int]) -> Dict[int, TFN]:
    K = len(levels)
    centers = np.linspace(0, 100, K)
    mapping = {}
    for i, lv in enumerate(levels):
        b = float(centers[i])
        if i == 0: a, c = 0.0, (centers[i]+centers[i+1])/2
        elif i == K-1: a, c = (centers[i-1]+centers[i])/2, 100.0
        else: a, c = (centers[i-1]+centers[i])/2, (centers[i]+centers[i+1])/2
        mapping[lv] = TFN(a, b, c)
    return mapping

def ensure_tfn(x):
    if isinstance(x, TFN): return x
    if isinstance(x, (int, float)): return TFN(x, x, x)
    if isinstance(x, (list, tuple)) and len(x)==3: return TFN(*x)
    raise ValueError(f"Cannot convert {x} to TFN")

def defuzz_buckley(x: TFN) -> float:
    return (x.a + 2*x.b + x.c) / 4.0

# ==============================
# (1) INDIVIDUAL Fuzzy–Hybrid TOPSIS
# ==============================
def _normalize_fuzzy_matrix(matrix, is_benefit):
    m,n=len(matrix),len(matrix[0])
    matrix=[[ensure_tfn(x) for x in row] for row in matrix]
    c_max=[max(matrix[i][j].c for i in range(m)) for j in range(n)]
    a_min=[min(matrix[i][j].a for i in range(m)) for j in range(n)]
    out=[]
    for i in range(m):
        row=[]
        for j in range(n):
            x=matrix[i][j]
            if is_benefit[j]:
                denom=c_max[j] if c_max[j]!=0 else 1
                row.append(TFN(x.a/denom,x.b/denom,x.c/denom))
            else:
                amin=a_min[j] if a_min[j]!=0 else 1
                row.append(TFN(amin/x.c, amin/(x.b if x.b else 1e-9), amin/(x.a if x.a else 1e-9)))
        out.append(row)
    return out

def _apply_weights(matrix,weights):
    wsum=sum(weights); w=[wi/wsum for wi in weights]
    return [[matrix[i][j].scale(w[j]) for j in range(len(matrix[0]))] for i in range(len(matrix))]

def _fuzzy_distance(x,y): return math.sqrt((x.a-y.a)**2+(x.b-y.b)**2+(x.c-y.c)**2)

def fuzzy_topsis_cc(matrix,is_benefit,weights=None):
    m,n=len(matrix),len(matrix[0])
    if weights is None: weights=[1/n]*n
    norm=_normalize_fuzzy_matrix(matrix,is_benefit)
    vw=_apply_weights(norm,weights)
    fpis=[]; fnis=[]
    for j in range(n):
        col=[vw[i][j] for i in range(m)]
        fpis.append(TFN(max(x.a for x in col),max(x.b for x in col),max(x.c for x in col)))
        fnis.append(TFN(min(x.a for x in col),min(x.b for x in col),min(x.c for x in col)))
    d_plus=np.zeros(m); d_minus=np.zeros(m)
    for i in range(m):
        for j in range(n):
            d_plus[i]+=_fuzzy_distance(vw[i][j],fpis[j])**2
            d_minus[i]+=_fuzzy_distance(vw[i][j],fnis[j])**2
        d_plus[i]=math.sqrt(d_plus[i]); d_minus[i]=math.sqrt(d_minus[i])
    return np.clip(d_minus/(d_plus+d_minus+1e-12),0,1)

def df_to_tfn_matrix(df: pd.DataFrame, cols: List[str], tfn_map: Dict[int,TFN], levels: List[int]):
    m=df.shape[0]; mat=[]; median=int(np.median(levels)); fallback=tfn_map[median]
    for i in range(m):
        row=[]
        for c in cols:
            v=df.iloc[i][c]
            try: iv=int(v)
            except: iv=None
            if iv not in levels: iv=median
            row.append(tfn_map.get(iv,fallback))
        mat.append(row)
    return mat

# ==============================
# (2) GROUP TOPSIS (crisp) + PIS/NIS table
# ==============================
def group_topsis_with_pis_nis(df: pd.DataFrame, items: List[str], tfn_maps: Dict[str,Dict[int,TFN]], levels_by_item: Dict[str,List[int]], group_col: str):
    agg={}; groups=list(df[group_col].dropna().unique())
    for g in groups:
        dfg=df[df[group_col]==g]; row={}
        for it in items:
            levels=levels_by_item[it]; mapping=tfn_maps[it]; med=int(np.median(levels)); fallback=mapping[med]
            tfns=[]
            for v in dfg[it].tolist():
                try: iv=int(v)
                except: iv=None
                if iv not in levels: iv=med
                tfns.append(mapping.get(iv,fallback))
            if tfns:
                a=np.mean([t.a for t in tfns]); b=np.mean([t.b for t in tfns]); c=np.mean([t.c for t in tfns])
            else:
                a=b=c=0.0
            row[it]=defuzz_buckley(TFN(a,b,c))  # defuzzify
        agg[g]=row
    V=pd.DataFrame.from_dict(agg, orient="index")[items]
    pis=V.max(axis=0); nis=V.min(axis=0)
    S_plus=np.sqrt(((V - pis)**2).sum(axis=1))
    S_minus=np.sqrt(((V - nis)**2).sum(axis=1))
    topsis=(S_minus/(S_plus+S_minus+1e-12)).clip(0,1)
    group_df=pd.DataFrame({"Group":V.index, "TOPSIS": topsis.values}).sort_values("TOPSIS", ascending=False)
    rows=[]
    for it in items:
        rows.append({
            "Item": it,
            "PIS_Group": V[it].idxmax(), "PIS": round(float(V[it].max()),2),
            "NIS_Group": V[it].idxmin(), "NIS": round(float(V[it].min()),2)
        })
    ideal_df=pd.DataFrame(rows)
    return group_df, ideal_df

# ==============================
# (3) Classic Apostle
# ==============================
def classic_apostle_threshold(x,y,thr_x,thr_y,labels4):
    a,b,c,d = labels4
    out=[]
    for xi,yi in zip(x,y):
        if xi>=thr_x and yi>=thr_y: out.append(a)
        elif xi<thr_x and yi>=thr_y: out.append(b)
        elif xi<thr_x and yi<thr_y: out.append(c)
        else: out.append(d)
    return out

# ==============================
# (4) Extended Apostle (FCM + α rule)
# ==============================
def fuzzy_cmeans_memberships(values: np.ndarray, c=3, m=2.0, error=1e-6, maxiter=1000, seed=42):
    data=np.array(values, dtype=float).reshape(1,-1)
    cntr,u,_,_,_,_,_=fuzz.cluster.cmeans(data,c=c,m=m,error=error,maxiter=maxiter,seed=seed)
    order=np.argsort(cntr)
    least, inter, most = order[0], order[1], order[2]
    U=np.vstack([u[most],u[least],u[inter]]).T
    return U, cntr[order]

def extended_apostle_from_memberships(Ux: np.ndarray, Uy: np.ndarray, alpha: float=0.5):
    def code(u):
        if u[1] >= alpha: return 1
        if (u<alpha).all(): return 2
        if u[2] >= alpha: return 3
        if u[0] >= alpha: return 4
        return 2
    return [f"({code(Ux[i])},{code(Uy[i])})" for i in range(Ux.shape[0])]

# ==============================
# (5) Probability ratios
# ==============================
def _prob_ratio_bootstrap(A: np.ndarray, B: np.ndarray, n_boot: int = 1000, seed: int = 42):
    rng=np.random.default_rng(seed); N=len(A); vals=[]
    for _ in range(n_boot):
        idx=rng.integers(0,N,N); As=A[idx]; Bs=B[idx]
        pA=As.mean(); pB=Bs.mean(); pAB=(As & Bs).mean()
        if pA>0 and pB>0: vals.append(pAB/(pA*pB))
    if not vals: return np.nan,np.nan,np.nan
    return float(np.mean(vals)), float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))

def conditional_probability_ratios_by_level(df: pd.DataFrame, quad_col: str, covar_cols: List[str], max_levels: int = 12, n_boot: int = 1000):
    rows=[]
    for cov in covar_cols:
        series=df[cov]; unique=series.dropna().unique()
        levels=[(str(v),v) for v in sorted(unique, key=lambda x: str(x))[:max_levels]]
        if len(unique)>max_levels: levels.append(("OTHER",None))
        for q in df[quad_col].dropna().unique():
            A=(df[quad_col]==q).astype(int).to_numpy()
            for name,val in levels:
                B=(~series.isin([v for _,v in levels[:-1]])).astype(int).to_numpy() if val is None else (series==val).astype(int).to_numpy()
                mean,lo,hi=_prob_ratio_bootstrap(A,B,n_boot=n_boot)
                rows.append({"Quadrant":str(q),"Covariate":cov,"Level":name,"Mean":round(mean,3),"CI_low":round(lo,3),"CI_high":round(hi,3)})
    return pd.DataFrame(rows)

# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="Fuzzy TOPSIS (Separated)", layout="wide")
st.title("Fuzzy-Hybrid TOPSIS (Individual) + Group TOPSIS + Apostle (Classic/Extended) + Ratios")

if st.sidebar.button("🔄 Reset analysis"):
    for k in ["analysis_done","results","idx","Ux","Uy"]:
        if k in st.session_state: del st.session_state[k]

df=None
up=st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
if up is not None:
    df=pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    st.dataframe(df.head())

if df is not None:
    st.sidebar.header("Latent variables")
    all_cols=list(df.columns)
    lname_x=st.sidebar.text_input("Latent X name", value="X")
    items_x=st.sidebar.multiselect("Items for X", all_cols, key="items_x")
    lname_y=st.sidebar.text_input("Latent Y name", value="Y")
    items_y=st.sidebar.multiselect("Items for Y", all_cols, key="items_y")

    st.sidebar.header("Scales per item")
    tfn_map_by_item={}; levels_by_item={}
    for it in items_x+items_y:
        sc_choice=st.sidebar.selectbox(f"Scale for {it}",
            ["Likert1-4","Likert1-5","Likert1-6","Likert1-7","Likert1-10","Likert1-11","Linear","Manual"], key=f"sc_{it}")
        if sc_choice=="Likert1-4": tfn_map_by_item[it]=likert_map_1_4(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-5": tfn_map_by_item[it]=likert_map_1_5(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-6": tfn_map_by_item[it]=likert_map_1_6(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-7": tfn_map_by_item[it]=likert_map_1_7(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-10": tfn_map_by_item[it]=likert_map_1_10(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-11": tfn_map_by_item[it]=likert_map_1_11(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Linear":
            lv=[int(x) for x in st.sidebar.text_input(f"Levels for {it}", value="1,2,3,4,5", key=f"lv_{it}").split(",")]
            tfn_map_by_item[it]=linear_tfn_map(lv); levels_by_item[it]=lv
        else:
            lv=[int(x) for x in st.sidebar.text_input(f"Levels for {it}", value="1,2,3,4,5", key=f"lvman_{it}").split(",")]
            levels_by_item[it]=lv; tfn_map_by_item[it]={}
            for l in lv:
                abctxt=st.sidebar.text_input(f"{it}-{l} TFN", value="0,0,25", key=f"tfn_{it}_{l}")
                a,b,c=[float(x) for x in abctxt.split(",")]
                tfn_map_by_item[it][l]=TFN(a,b,c)

    st.sidebar.header("Classic Apostle thresholds")
    thr_x=st.sidebar.slider("Threshold X", 0.0, 1.0, 0.5, 0.01)
    thr_y=st.sidebar.slider("Threshold Y", 0.0, 1.0, 0.5, 0.01)
    lab_a = st.sidebar.text_input("(X≥thr & Y≥thr)", "Quadrant A")
      lab_b = st.sidebar.text_input("(X<thr & Y≥thr)", "Quadrant B")
      lab_c = st.sidebar.text_input("(X<thr & Y<thr)", "Quadrant C")
      lab_d = st.sidebar.text_input("(X≥thr & Y<thr)", "Quadrant D")

    }
def likert_map_1_11(): return {i: TFN((i-1)*10,(i-1)*10+10,(i-1)*10+20) for i in range(1,12)}

def linear_tfn_map(levels: List[int]) -> Dict[int, TFN]:
    K = len(levels)
    centers = np.linspace(0, 100, K)
    mapping = {}
    for i, lv in enumerate(levels):
        b = float(centers[i])
        if i == 0: a, c = 0.0, (centers[i]+centers[i+1])/2
        elif i == K-1: a, c = (centers[i-1]+centers[i])/2, 100.0
        else: a, c = (centers[i-1]+centers[i])/2, (centers[i]+centers[i+1])/2
        mapping[lv] = TFN(a, b, c)
    return mapping

# ==============================
# 2) Fuzzy-Hybrid TOPSIS
# ==============================
def ensure_tfn(x):
    if isinstance(x, TFN): return x
    if isinstance(x, (int, float)): return TFN(x, x, x)
    if isinstance(x, (list, tuple)) and len(x)==3: return TFN(*x)
    raise ValueError(f"Cannot convert {x} to TFN")

def _normalize_fuzzy_matrix(matrix, is_benefit):
    m,n=len(matrix),len(matrix[0])
    matrix=[[ensure_tfn(x) for x in row] for row in matrix]
    c_max=[max(matrix[i][j].c for i in range(m)) for j in range(n)]
    a_min=[min(matrix[i][j].a for i in range(m)) for j in range(n)]
    out=[]
    for i in range(m):
        row=[]
        for j in range(n):
            x=matrix[i][j]
            if is_benefit[j]:
                denom=c_max[j] if c_max[j]!=0 else 1
                row.append(TFN(x.a/denom,x.b/denom,x.c/denom))
            else:
                amin=a_min[j] if a_min[j]!=0 else 1
                row.append(TFN(amin/x.c, amin/(x.b if x.b else 1e-9), amin/(x.a if x.a else 1e-9)))
        out.append(row)
    return out

def _apply_weights(matrix,weights):
    wsum=sum(weights); w=[wi/wsum for wi in weights]
    return [[matrix[i][j].scale(w[j]) for j in range(len(matrix[0]))] for i in range(len(matrix))]

def _fuzzy_distance(x,y): return math.sqrt((x.a-y.a)**2+(x.b-y.b)**2+(x.c-y.c)**2)

def fuzzy_topsis_cc(matrix,is_benefit,weights=None):
    m,n=len(matrix),len(matrix[0])
    if weights is None: weights=[1/n]*n
    norm=_normalize_fuzzy_matrix(matrix,is_benefit)
    vw=_apply_weights(norm,weights)
    fpis=[]; fnis=[]
    for j in range(n):
        col=[vw[i][j] for i in range(m)]
        fpis.append(TFN(max(x.a for x in col),max(x.b for x in col),max(x.c for x in col)))
        fnis.append(TFN(min(x.a for x in col),min(x.b for x in col),min(x.c for x in col)))
    d_plus=np.zeros(m); d_minus=np.zeros(m)
    for i in range(m):
        for j in range(n):
            d_plus[i]+=_fuzzy_distance(vw[i][j],fpis[j])**2
            d_minus[i]+=_fuzzy_distance(vw[i][j],fnis[j])**2
        d_plus[i]=math.sqrt(d_plus[i]); d_minus[i]=math.sqrt(d_minus[i])
    return np.clip(d_minus/(d_plus+d_minus+1e-12),0,1), fpis, fnis

def df_to_tfn_matrix(df, cols, tfn_map, levels):
    m = df.shape[0]
    mat = []
    median_level = int(np.median(levels))
    fallback = tfn_map[median_level]
    for i in range(m):
        row = []
        for c in cols:
            v = df.iloc[i][c]
            try: iv = int(v)
            except: iv = None
            if iv not in levels: iv = median_level
            row.append(tfn_map.get(iv, fallback))
        mat.append(row)
    return mat

# ==============================
# 3) Quadrants (Apostle & ECO)
# ==============================
def apostle_quadrants(x,y,x_thr,y_thr,AA,AB,BA,BB):
    out=[]
    for xi,yi in zip(x,y):
        if xi>=x_thr and yi>=y_thr: out.append(AA)
        elif xi>=x_thr and yi<y_thr: out.append(AB)
        elif xi<x_thr and yi>=y_thr: out.append(BA)
        else: out.append(BB)
    return out

def eco_fuzzy_sets_4(val):
    low=medlow=medhigh=high=0
    if val<=0.33: low=1-val/0.33
    if val>=0.66: high=(val-0.66)/0.34 if val<=1 else 1
    if 0<=val<=0.66: medlow=1-abs(val-0.33)/0.33
    if 0.33<=val<=1: medhigh=1-abs(val-0.66)/0.34
    return (max(low,0),max(medlow,0),max(medhigh,0),max(high,0))

def eco_extended_labels_4x4(x,y,x_names,y_names):
    labels=[]
    for xi,yi in zip(x,y):
        lx=np.array(eco_fuzzy_sets_4(xi)); ly=np.array(eco_fuzzy_sets_4(yi))
        labels.append(f"{x_names[lx.argmax()]}|{y_names[ly.argmax()]}")
    return labels

# ==============================
# 4) Probability Ratios (bootstrap)
# ==============================
def _prob_ratio_bootstrap(A: np.ndarray, B: np.ndarray, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    N = len(A)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        As = A[idx]; Bs = B[idx]
        pA = As.mean(); pB = Bs.mean()
        pAB = (As & Bs).mean()
        if pA > 0 and pB > 0:
            vals.append(pAB / (pA * pB))
    if not vals:
        return np.nan, np.nan, np.nan
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def conditional_probability_ratios_by_level(df: pd.DataFrame, quad_col: str, covar_cols: List[str], max_levels: int = 8, n_boot: int = 1000):
    rows = []
    for cov in covar_cols:
        series = df[cov]
        unique_vals = series.dropna().unique()
        if (series.dtype.kind in "biufc") and len(unique_vals) > max_levels:
            levels = [("__NONMISSING__", "__NONMISSING__")]
        else:
            levels = [(str(v), v) for v in sorted(unique_vals, key=lambda x: str(x))[:max_levels]]
            if len(unique_vals) > max_levels:
                levels.append(("OTHER", None))
        for q in df[quad_col].dropna().unique():
            A = (df[quad_col] == q).astype(int).to_numpy()
            for lvl_name, lvl_val in levels:
                if lvl_val == "__NONMISSING__":
                    B = (~series.isna()).astype(int).to_numpy()
                elif lvl_val is None:
                    B = (~series.isin([v for _, v in levels[:-1]])).astype(int).to_numpy()
                else:
                    B = (series == lvl_val).astype(int).to_numpy()
                mean, lo, hi = _prob_ratio_bootstrap(A, B, n_boot=n_boot)
                rows.append({"Quadrant": str(q), "Covariate": cov, "Level": lvl_name, "Mean": round(mean,3), "CI_low": round(lo,3), "CI_high": round(hi,3)})
    return pd.DataFrame(rows)

# ==============================
# 5) Streamlit App (with session_state to persist results)
# ==============================
st.set_page_config(page_title="Fuzzy TOPSIS + Apostle", layout="wide")
st.title("Fuzzy-Hybrid TOPSIS + Apostle Model")

# Reset analysis if user asks
if st.sidebar.button("🔄 Reset analysis"):
    for k in ["analysis_done", "results", "ideal_solutions"]:
        if k in st.session_state: del st.session_state[k]

df=None
up=st.file_uploader("Upload CSV or Excel",type=["csv","xlsx"])
if up is not None:
    df=pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    st.dataframe(df.head())

if df is not None:
    st.sidebar.header("Latent variables")
    all_cols=list(df.columns)
    lname_x=st.sidebar.text_input("Latent X name",value="LatX")
    items_x=st.sidebar.multiselect("Items for X",all_cols, key="items_x")
    lname_y=st.sidebar.text_input("Latent Y name",value="LatY")
    items_y=st.sidebar.multiselect("Items for Y",all_cols, key="items_y")

    st.sidebar.header("Scales per item")
    tfn_map_by_item={}; levels_by_item={}
    for it in items_x+items_y:
        sc_choice=st.sidebar.selectbox(
            f"Scale for {it}",
            ["Likert1-4","Likert1-5","Likert1-6","Likert1-7","Likert1-10","Likert1-11","Linear","Manual"],
            key=f"sc_{it}"
        )
        if sc_choice=="Likert1-4": tfn_map_by_item[it]=likert_map_1_4(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-5": tfn_map_by_item[it]=likert_map_1_5(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-6": tfn_map_by_item[it]=likert_map_1_6(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-7": tfn_map_by_item[it]=likert_map_1_7(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-10": tfn_map_by_item[it]=likert_map_1_10(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-11": tfn_map_by_item[it]=likert_map_1_11(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Linear":
            lv=[int(x) for x in st.sidebar.text_input(f"Levels for {it}",value="1,2,3,4,5",key=f"lv_{it}").split(",")]
            tfn_map_by_item[it]=linear_tfn_map(lv); levels_by_item[it]=lv
        else:
            lv=[int(x) for x in st.sidebar.text_input(f"Levels for {it}",value="1,2,3,4,5",key=f"lvman_{it}").split(",")]
            levels_by_item[it]=lv; tfn_map_by_item[it]={}
            for l in lv:
                abctxt=st.sidebar.text_input(f"{it}-{l} TFN",value="0,0,25",key=f"tfn_{it}_{l}")
                a,b,c=[float(x) for x in abctxt.split(",")]
                tfn_map_by_item[it][l]=TFN(a,b,c)

    st.sidebar.header("Quadrant naming")
    AA=st.sidebar.text_input("Classic: x≥thr & y≥thr",value="Apostles")
    AB=st.sidebar.text_input("Classic: x≥thr & y<thr",value="Mercenaries")
    BA=st.sidebar.text_input("Classic: x<thr & y≥thr",value="Hostages")
    BB=st.sidebar.text_input("Classic: x<thr & y<thr",value="Defectors")
    x_names=[st.sidebar.text_input(f"X name {i}",v) for i,v in enumerate(["LowX","MedLowX","MedHighX","HighX"])]
    y_names=[st.sidebar.text_input(f"Y name {i}",v) for i,v in enumerate(["LowY","MedLowY","MedHighY","HighY"])]

    # Run analysis button
    if st.button("Run analysis") and items_x and items_y:
        idx={}; ideal_solutions={}
        for nm,items in {lname_x:items_x,lname_y:items_y}.items():
            mat=[]
            for it in items:
                mat_item = df_to_tfn_matrix(df, [it], tfn_map_by_item[it], levels_by_item[it])
                if not mat:
                    mat = [[row[0]] for row in mat_item]
                else:
                    for r, row in enumerate(mat_item):
                        mat[r].append(row[0])
            idx[nm], fpis, fnis = fuzzy_topsis_cc(mat,[True]*len(items),[1.0]*len(items))
            ideal_solutions[nm]={"PIS":fpis,"NIS":fnis}
        x,y=idx[lname_x],idx[lname_y]
        x_thr,y_thr=np.mean(x),np.mean(y)
        res=pd.DataFrame({f"idx_{lname_x}":x,f"idx_{lname_y}":y})
        res["Classic"]=apostle_quadrants(x,y,x_thr,y_thr,AA,AB,BA,BB)
        res["Extended4x4"]=eco_extended_labels_4x4(x,y,x_names,y_names)

        st.session_state['analysis_done'] = True
        st.session_state['results'] = res
        st.session_state['ideal_solutions'] = ideal_solutions

    # Show stored results (persist across reruns)
    if st.session_state.get('analysis_done') and st.session_state.get('results') is not None:
        res = st.session_state['results']
        ideal_solutions = st.session_state['ideal_solutions']

        st.subheader("Results (individual)")
        st.dataframe(res)

        st.subheader("Ideal solutions (PIS/NIS)")
        st.json({nm:{"PIS":[(t.a,t.b,t.c) for t in v["PIS"]],"NIS":[(t.a,t.b,t.c) for t in v["NIS"]]} for nm,v in ideal_solutions.items()})

        # ==============================
        #  Group-level TOPSIS + summary
        # ==============================
        group_cols = st.sidebar.multiselect("Group by columns", list(df.columns), key="group_cols")
        if group_cols:
            summaries = []
            for gcol in group_cols:
                for gval, dfg in df.groupby(gcol):
                    for nm, items in {lname_x: items_x, lname_y: items_y}.items():
                        if not items:
                            continue
                        mat = []
                        for it in items:
                            mat_item = df_to_tfn_matrix(dfg, [it], tfn_map_by_item[it], levels_by_item[it])
                            if not mat:
                                mat = [[row[0]] for row in mat_item]
                            else:
                                for r, row in enumerate(mat_item):
                                    mat[r].append(row[0])
                        idx_g, fpis, fnis = fuzzy_topsis_cc(mat, [True]*len(items), [1.0]*len(items))
                        summaries.append({
                            "Item": nm,
                            "Group": f"{gcol}: {gval}",
                            "PIS": round(np.mean([t.b for t in fpis]), 2),
                            "NIS": round(np.mean([t.b for t in fnis]), 2)
                        })
            if summaries:
                st.subheader("Summary by group (PIS/NIS)")
                summary_df = pd.DataFrame(summaries)
                st.dataframe(summary_df)

        # ==============================
        #  Conditional Probability Ratios
        # ==============================
        covar_cols = st.sidebar.multiselect("Covariates for probability ratios", list(df.columns), key="covars")
        if covar_cols:
            st.subheader("Conditional probability ratios")
            res_full = pd.concat([df.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
            ratios = conditional_probability_ratios_by_level(res_full, "Classic", covar_cols, max_levels=8, n_boot=1000)
            st.dataframe(ratios)

# (optional) CSV download of individual results
if st.session_state.get('results') is not None:
    csv = st.session_state['results'].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download individual results CSV", data=csv, file_name="topsis_individual_results.csv")

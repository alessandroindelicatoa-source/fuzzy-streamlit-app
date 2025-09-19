\
# app_fuzzy.py (versión final con nombres fijos para Classic Apostle y Extended 4x4)
import math
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
import streamlit as st
import skfuzzy as fuzz

# --- TFN ---
@dataclass(frozen=True)
class TFN:
    a: float; b: float; c: float
    def __post_init__(self):
        if not (self.a <= self.b <= self.c):
            raise ValueError("TFN requiere a ≤ b ≤ c.")
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
        if i == 0:
            a, c = 0.0, (centers[i]+centers[i+1])/2
        elif i == K-1:
            a, c = (centers[i-1]+centers[i])/2, 100.0
        else:
            a, c = (centers[i-1]+centers[i])/2, (centers[i]+centers[i+1])/2
        mapping[lv] = TFN(a, b, c)
    return mapping

def ensure_tfn(x):
    if isinstance(x, TFN): return x
    if isinstance(x, (int, float)): return TFN(x, x, x)
    if isinstance(x, (list, tuple)) and len(x)==3: return TFN(*x)
    raise ValueError(f"No se puede convertir {x} a TFN")

def defuzz_buckley(x: TFN) -> float:
    return (x.a + 2*x.b + x.c) / 4.0

# --- TOPSIS individual ---
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

# --- Classic Apostle ---
def classic_apostle_threshold(x,y,thr_x,thr_y):
    out=[]
    for xi,yi in zip(x,y):
        if xi>=thr_x and yi>=thr_y: out.append("Apostles")
        elif xi<thr_x and yi>=thr_y: out.append("Hostages")
        elif xi<thr_x and yi<thr_y: out.append("Defectors")
        else: out.append("Mercenaries")
    return out

# --- Extended Apostle ---
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

# --- Streamlit App ---
st.set_page_config(page_title="Fuzzy TOPSIS Final", layout="wide")
st.title("Fuzzy-Hybrid TOPSIS + Group TOPSIS + Apostle + Ratios")

df=None
up=st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
if up is not None:
    df=pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    st.dataframe(df.head())

if df is not None:
    all_cols=list(df.columns)
    lname_x=st.sidebar.text_input("Latent X name", value="X")
    items_x=st.sidebar.multiselect("Items for X", all_cols, key="items_x_widget")
    lname_y=st.sidebar.text_input("Latent Y name", value="Y")
    items_y=st.sidebar.multiselect("Items for Y", all_cols, key="items_y_widget")

    thr_x=st.sidebar.slider("Threshold X", 0.0, 1.0, 0.5, 0.01)
    thr_y=st.sidebar.slider("Threshold Y", 0.0, 1.0, 0.5, 0.01)
    alpha=st.sidebar.slider("Extended FCM alpha", 0.1, 0.9, 0.5, 0.05)

    if st.button("Run analysis") and items_x and items_y:
        idx={}
        for nm,items in {lname_x:items_x, lname_y:items_y}.items():
            mat=[]
            for it in items:
                m_item=df_to_tfn_matrix(df,[it],likert_map_1_10(), list(likert_map_1_10().keys()))
                if not mat: mat=[[row[0]] for row in m_item]
                else:
                    for r,row in enumerate(m_item): mat[r].append(row[0])
            idx[nm]=fuzzy_topsis_cc(mat,[True]*len(items),[1.0]*len(items))
        x=np.array(idx[lname_x]); y=np.array(idx[lname_y])
        classic=classic_apostle_threshold(x,y,thr_x,thr_y)
        Ux,_=fuzzy_cmeans_memberships(x); Uy,_=fuzzy_cmeans_memberships(y)
        extended=extended_apostle_from_memberships(Ux,Uy,alpha=alpha)
        res=pd.DataFrame({f"{lname_x}":x, f"{lname_y}":y, "Classic":classic, "Extended4x4":extended})
        st.dataframe(res)

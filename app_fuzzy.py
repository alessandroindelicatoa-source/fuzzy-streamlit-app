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

def likert4_default_map() -> Dict[int, TFN]:
    return {1: TFN(0,0,50), 2: TFN(30,50,70), 3: TFN(50,70,90), 4: TFN(70,100,100)}

def likert5_default_map() -> Dict[int, TFN]:
    return {1: TFN(0,0,25), 2: TFN(15,30,45), 3: TFN(40,50,60), 4: TFN(55,70,85), 5: TFN(75,100,100)}

def linear_tfn_map(levels: List[int]) -> Dict[int, TFN]:
    K = len(levels)
    centers = np.linspace(0,100,K)
    mapping={}
    for i,lv in enumerate(levels):
        b = float(centers[i])
        if i==0: a=0; c=(centers[i]+centers[i+1])/2
        elif i==K-1: a=(centers[i-1]+centers[i])/2; c=100
        else: a=(centers[i-1]+centers[i])/2; c=(centers[i]+centers[i+1])/2
        mapping[lv] = TFN(a,b,c)
    return mapping

# ==============================
# Fuzzy-Hybrid TOPSIS
# ==============================
def _normalize_fuzzy_matrix(matrix, is_benefit):
    m=len(matrix); n=len(matrix[0])
    c_max=[max(matrix[i][j].c for i in range(m)) for j in range(n)]
    a_min=[min(matrix[i][j].a for i in range(m)) for j in range(n)]
    out=[]
    for i in range(m):
        row=[]
        for j in range(n):
            x=matrix[i][j]
            if is_benefit[j]:
                denom = c_max[j] if c_max[j]!=0 else 1
                row.append(TFN(x.a/denom, x.b/denom, x.c/denom))
            else:
                amin=a_min[j] if a_min[j]!=0 else 1
                row.append(TFN(amin/x.c, amin/(x.b if x.b!=0 else 1e-9), amin/(x.a if x.a!=0 else 1e-9)))
        out.append(row)
    return out

def _apply_weights(matrix,weights):
    m=len(matrix); n=len(matrix[0])
    wsum=sum(weights); w=[wi/wsum for wi in weights]
    return [[matrix[i][j].scale(w[j]) for j in range(n)] for i in range(m)]

def _fuzzy_distance(x,y):
    return math.sqrt((x.a-y.a)**2+(x.b-y.b)**2+(x.c-y.c)**2)

def fuzzy_topsis_cc(matrix,is_benefit,weights=None):
    m=len(matrix); n=len(matrix[0])
    if weights is None: weights=[1/n]*n
    norm=_normalize_fuzzy_matrix(matrix,is_benefit)
    vw=_apply_weights(norm,weights)
    fpis=[]; fnis=[]
    for j in range(n):
        col=[vw[i][j] for i in range(m)]
        fpis.append(TFN(max(x.a for x in col), max(x.b for x in col), max(x.c for x in col)))
        fnis.append(TFN(min(x.a for x in col), min(x.b for x in col), min(x.c for x in col)))
    d_plus=np.zeros(m); d_minus=np.zeros(m)
    for i in range(m):
        for j in range(n):
            d_plus[i]+= _fuzzy_distance(vw[i][j], fpis[j])**2
            d_minus[i]+= _fuzzy_distance(vw[i][j], fnis[j])**2
        d_plus[i]=math.sqrt(d_plus[i]); d_minus[i]=math.sqrt(d_minus[i])
    cc=d_minus/(d_plus+d_minus+1e-12)
    return np.clip(cc,0,1)

def df_to_tfn_matrix(df,cols,tfn_map_global,levels,tfn_map_by_col):
    m=df.shape[0]; mat=[]
    for i in range(m):
        row=[]
        for c in cols:
            col_map=tfn_map_by_col.get(c,tfn_map_global)
            v=int(df.iloc[i][c])
            row.append(col_map[v])
        mat.append(row)
    return mat

# ==============================
# ECO-Extended 4×4 fuzzy sets
# ==============================
def eco_fuzzy_sets_4(val):
    low=medlow=medhigh=high=0
    if val<=0.33: low=1-val/0.33
    if val>=0.66: high=(val-0.66)/0.34 if val<=1 else 1
    if 0<=val<=0.66: medlow=1-abs(val-0.33)/0.33
    if 0.33<=val<=1: medhigh=1-abs(val-0.66)/0.34
    return (max(low,0),max(medlow,0),max(medhigh,0),max(high,0))

def eco_extended_labels_4x4(x,y):
    labels=[]
    for xi,yi in zip(x,y):
        lx=np.array(eco_fuzzy_sets_4(xi))
        ly=np.array(eco_fuzzy_sets_4(yi))
        labels.append(f"{['LowX','MedLowX','MedHighX','HighX'][lx.argmax()]}|{['LowY','MedLowY','MedHighY','HighY'][ly.argmax()]}")
    return labels

# ==============================
# Classic Apostle (2×2)
# ==============================
def apostle_quadrants(x,y,x_thr,y_thr,q_AA,q_AB,q_BA,q_BB):
    out=[]
    for xi,yi in zip(x,y):
        if xi>=x_thr and yi>=y_thr: out.append(q_AA)
        elif xi>=x_thr and yi<y_thr: out.append(q_AB)
        elif xi<x_thr and yi>=y_thr: out.append(q_BA)
        else: out.append(q_BB)
    return out

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Fuzzy-Hybrid TOPSIS + Apostle", layout="wide")
st.title("Fuzzy-Hybrid TOPSIS with Apostle Classic (2×2) & ECO-Extended (4×4)")

df=None
up=st.file_uploader("Upload CSV or Excel",type=["csv","xlsx"])
sep=st.text_input("CSV separator",value=",")
sheet=st.text_input("Excel sheet name",value="")
if up is not None:
    try:
        if up.name.endswith(".csv"): df=pd.read_csv(up,sep=sep)
        else: df=pd.read_excel(up,sheet_name=sheet if sheet else 0)
        st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
        st.dataframe(df.head())
    except Exception as e: st.error(f"Read error: {e}")

# ---- Scale & TFNs ----
st.header("1) Global Scale & TFNs")
scale_choice=st.selectbox("Scale",["Likert 1–4","Likert 1–5","Arbitrary (linear)","Arbitrary (manual)"])
levels=[]; tfn_map_global={}; tfn_map_by_col={}
if scale_choice=="Likert 1–4": levels=[1,2,3,4]; tfn_map_global=likert4_default_map()
elif scale_choice=="Likert 1–5": levels=[1,2,3,4,5]; tfn_map_global=likert5_default_map()
else:
    levels=[int(x) for x in st.text_input("Levels","1,2,3,4,5").split(",")]
    if "linear" in scale_choice: tfn_map_global=linear_tfn_map(levels)
    else:
        for lv in levels:
            a,b,c=[float(x) for x in st.text_input(f"Level {lv} (a,b,c)","0,0,25",key=f"lv{lv}").split(",")]
            tfn_map_global[lv]=TFN(a,b,c)

if df is not None:
    st.subheader("Optional per-column TFNs")
    for col in df.columns:
        if st.checkbox(f"Custom TFNs for {col}"):
            lvls=[int(x) for x in st.text_input(f"Levels for {col}","1,2,3,4,5",key=f"lvls_{col}").split(",")]
            mapping={}
            for lv in lvls:
                a,b,c=[float(x) for x in st.text_input(f"{col}-{lv}","0,0,25",key=f"tfn_{col}_{lv}").split(",")]
                mapping[lv]=TFN(a,b,c)
            tfn_map_by_col[col]=mapping

# ---- Latents ----
if df is not None:
    st.header("2) Latent variables")
    cols=st.multiselect("Available columns",df.columns)
    latentX=st.multiselect("Items for latent X",cols)
    latentY=st.multiselect("Items for latent Y",cols)

    cat_cols=st.multiselect("Categorical columns for aggregate TOPSIS",df.columns)

    run_btn=st.button("Run analysis")
    if run_btn and latentX and latentY:
        # TOPSIS for X
        mX=df_to_tfn_matrix(df,latentX,tfn_map_global,levels,tfn_map_by_col)
        ccX=fuzzy_topsis_cc(mX,[True]*len(latentX))
        # TOPSIS for Y
        mY=df_to_tfn_matrix(df,latentY,tfn_map_global,levels,tfn_map_by_col)
        ccY=fuzzy_topsis_cc(mY,[True]*len(latentY))
        df["LatentX"]=ccX; df["LatentY"]=ccY

        # Classic quadrants
        st.subheader("Classic Apostle (2×2)")
        qAA=st.text_input("Quadrant HighX/HighY","Apostles")
        qAB=st.text_input("Quadrant HighX/LowY","Mercenaries")
        qBA=st.text_input("Quadrant LowX/HighY","Loyalists")
        qBB=st.text_input("Quadrant LowX/LowY","Defectors")
        thrX=st.slider("Threshold X",0.0,1.0,0.5)
        thrY=st.slider("Threshold Y",0.0,1.0,0.5)
        df["ClassicQuad"]=apostle_quadrants(ccX,ccY,thrX,thrY,qAA,qAB,qBA,qBB)
        st.write(df["ClassicQuad"].value_counts())
        fig,ax=plt.subplots()
        ax.scatter(ccX,ccY,c="blue")
        ax.axvline(thrX,color="red"); ax.axhline(thrY,color="red")
        st.pyplot(fig)

        # ECO Extended
        st.subheader("ECO-Extended Apostle (4×4)")
        df["ECOQuad"]=eco_extended_labels_4x4(ccX,ccY)
        st.write(df["ECOQuad"].value_counts())
        fig2,ax2=plt.subplots()
        ax2.scatter(ccX,ccY,c="green")
        st.pyplot(fig2)

        # Aggregated TOPSIS per categorical col
        if cat_cols:
            st.subheader("Aggregate TOPSIS by category")
            for cat in cat_cols:
                st.write(f"Grouping by {cat}")
                grouped=df.groupby(cat)
                for name,sub in grouped:
                    mX=df_to_tfn_matrix(sub,latentX,tfn_map_global,levels,tfn_map_by_col)
                    ccX=fuzzy_topsis_cc(mX,[True]*len(latentX))
                    mY=df_to_tfn_matrix(sub,latentY,tfn_map_global,levels,tfn_map_by_col)
                    ccY=fuzzy_topsis_cc(mY,[True]*len(latentY))
                    st.write(name,"→ mean X:",ccX.mean(),"mean Y:",ccY.mean())

        # Download
        out=BytesIO()
        df.to_csv(out,index=False)
        st.download_button("Download results CSV",out.getvalue(),"results.csv","text/csv")

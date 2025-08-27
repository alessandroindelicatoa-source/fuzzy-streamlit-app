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

# ==============================
# 0) Password gate
# ==============================
def check_password():
    passwords = st.secrets.get("passwords", {})
    users = {str(k).strip().lower(): str(v) for k, v in passwords.items()}

    if not users:
        st.warning("⚠️ No passwords configured. Running without login.")
        st.session_state["password_correct"] = True
        return True

    def _enter():
        u = st.session_state.get("username", "").strip().lower()
        p = st.session_state.get("password", "")
        ok = (u in users) and (p == users[u])
        st.session_state["password_correct"] = bool(ok)
        if ok and "password" in st.session_state:
            del st.session_state["password"]

    if "password_correct" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=_enter)
        st.stop()

    if not st.session_state["password_correct"]:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=_enter)
        st.error("❌ Incorrect username or password")
        st.stop()

check_password()

# ==============================
# 1) TFN and utilities
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

# Predefined scales
PREDEFINED_SCALES = {
    "Likert 1-4": {1: TFN(0,0,20), 2: TFN(16,33,60), 3: TFN(49,66,83), 4: TFN(80,100,100)},
    "Likert 1-5": {1: TFN(0,0,30), 2: TFN(20,30,40), 3: TFN(30,50,70), 4: TFN(60,70,80), 5: TFN(70,100,100)},
    "Likert 1-6": {1: TFN(0,0,15), 2: TFN(25,40,55), 3: TFN(45,60,75), 4: TFN(70,80,90), 5: TFN(85,100,100), 6: TFN(90,100,100)},
    "Likert 1-7": {i: TFN((i-1)*15,(i-1)*15+10,(i-1)*15+20) for i in range(1,8)},
    "Likert 1-10": {i: TFN((i-1)*10,(i-1)*10+10,(i-1)*10+20) for i in range(1,11)},
    "Likert 1-11": {i: TFN((i-1)*10,(i-1)*10+10,(i-1)*10+20) for i in range(1,12)},
}

def random_linear_map(levels: List[int]) -> Dict[int, TFN]:
    """Generate TFNs linearly spaced from 0-100"""
    mapping = {}
    step = 100/(len(levels))
    for i, lv in enumerate(levels):
        a = max(0,i*step-10); b = i*step; c = min(100,(i+1)*step+10)
        mapping[lv] = TFN(a,b,c)
    return mapping

# ==============================
# 2) Fuzzy-Hybrid TOPSIS
# ==============================
def _normalize_fuzzy_matrix(matrix: List[List[TFN]], is_benefit: List[bool]) -> List[List[TFN]]:
    m,n = len(matrix),len(matrix[0])
    c_max = [max(matrix[i][j].c for i in range(m)) for j in range(n)]
    a_min = [min(matrix[i][j].a for i in range(m)) for j in range(n)]
    out=[]
    for i in range(m):
        row=[]
        for j in range(n):
            x = matrix[i][j]
            if is_benefit[j]:
                denom = c_max[j] if c_max[j] else 1
                row.append(TFN(x.a/denom, x.b/denom, x.c/denom))
            else:
                amin = a_min[j] if a_min[j] else 1
                row.append(TFN(amin/x.c, amin/(x.b if x.b else 1e-9), amin/(x.a if x.a else 1e-9)))
        out.append(row)
    return out

def _apply_weights(matrix: List[List[TFN]], weights: List[float]) -> List[List[TFN]]:
    wsum = sum(weights); w=[wi/wsum for wi in weights]
    return [[matrix[i][j].scale(w[j]) for j in range(len(matrix[0]))] for i in range(len(matrix))]

def _fuzzy_distance(x: TFN,y:TFN)->float:
    return math.sqrt((x.a-y.a)**2+(x.b-y.b)**2+(x.c-y.c)**2)

def fuzzy_topsis_cc(matrix: List[List[TFN]], is_benefit: List[bool], weights: Optional[List[float]]=None)->np.ndarray:
    m,n = len(matrix),len(matrix[0])
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
    return d_minus/(d_plus+d_minus+1e-12)

def df_to_tfn_matrix(df: pd.DataFrame, cols: List[str], tfn_map_by_item: Dict[str,Dict[int,TFN]], levels_by_item: Dict[str,List[int]])->List[List[TFN]]:
    m=df.shape[0]; mat=[]
    for i in range(m):
        row=[]
        for c in cols:
            val=df.iloc[i][c]
            if pd.isna(val): 
                row.append(TFN(0,0,0))
            else:
                try: iv=int(val)
                except: iv=None
                if iv in levels_by_item[c]:
                    row.append(tfn_map_by_item[c][iv])
                else:
                    lv=levels_by_item[c]; mid=lv[len(lv)//2]
                    row.append(tfn_map_by_item[c][mid]) # imputación media
        mat.append(row)
    return mat

# ==============================
# 3) Quadrants
# ==============================
def apostle_quadrants(x,y,x_thr,y_thr,AA,AB,BA,BB)->List[str]:
    out=[]
    for xi,yi in zip(x,y):
        if xi>=x_thr and yi>=y_thr: out.append(AA)
        elif xi>=x_thr and yi<y_thr: out.append(AB)
        elif xi<x_thr and yi>=y_thr: out.append(BA)
        else: out.append(BB)
    return out

def eco_fuzzy_sets_4(val: float) -> Tuple[float,float,float,float]:
    low=medlow=medhigh=high=0
    if val<=0.33: low=1-val/0.33
    if val>=0.66: high=(val-0.66)/0.34 if val<=1 else 1
    if 0<=val<=0.66: medlow=1-abs(val-0.33)/0.33
    if 0.33<=val<=1: medhigh=1-abs(val-0.66)/0.34
    return (max(low,0),max(medlow,0),max(medhigh,0),max(high,0))

def eco_extended_labels_4x4(x,y,x_names,y_names)->List[str]:
    labels=[]
    for xi,yi in zip(x,y):
        lx=np.array(eco_fuzzy_sets_4(xi)); ly=np.array(eco_fuzzy_sets_4(yi))
        labels.append(f"{x_names[lx.argmax()]}|{y_names[ly.argmax()]}")
    return labels

# ==============================
# 4) Streamlit UI
# ==============================
st.set_page_config(page_title="Fuzzy-Hybrid TOPSIS + Apostle", layout="wide")
st.title("Fuzzy-Hybrid TOPSIS + Apostle Classic & ECO-Extended (Login Enabled)")

# Upload
df=None
up=st.file_uploader("Upload CSV/XLSX",type=["csv","xlsx"])
if up is not None:
    df=pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    st.dataframe(df.head())

# Latents
st.header("1) Select latent variables")
latents={}; is_benefit_by={}; weights_by={}
if df is not None:
    all_cols=list(df.columns)
    lname_x=st.text_input("Latent X name",value="LatX")
    items_x=st.multiselect("Items for X",all_cols)
    lname_y=st.text_input("Latent Y name",value="LatY")
    items_y=st.multiselect("Items for Y",all_cols)
    for nm,items in [(lname_x,items_x),(lname_y,items_y)]:
        if items:
            latents[nm]=items
            is_benefit_by[nm]=[True]*len(items)
            weights_by[nm]=[1.0]*len(items)

# TFN mapping per item
st.header("2) TFN conversion per item")
tfn_map_by_item={}; levels_by_item={}
if df is not None and latents:
    for it in set(sum(latents.values(),[])):
        sc_choice=st.selectbox(f"Scale for {it}",list(PREDEFINED_SCALES.keys())+["Random","Manual"],key=f"sc_{it}")
        if sc_choice in PREDEFINED_SCALES:
            tfn_map_by_item[it]=PREDEFINED_SCALES[sc_choice]
            levels_by_item[it]=list(PREDEFINED_SCALES[sc_choice].keys())
        elif sc_choice=="Random":
            lv=list(range(1,int(st.number_input(f"Levels for {it}",min_value=2,max_value=11,value=5))+1))
            tfn_map_by_item[it]=random_linear_map(lv); levels_by_item[it]=lv
        else: # Manual
            lvtxt=st.text_input(f"Levels for {it}",value="1,2,3,4,5",key=f"lv_{it}")
            lv=[int(x) for x in lvtxt.split(",")]
            levels_by_item[it]=lv; tfn_map_by_item[it]={}
            for l in lv:
                abctxt=st.text_input(f"TFN for {it}-{l}",value="0,0,25",key=f"tfn_{it}_{l}")
                a,b,c=[float(x) for x in abctxt.split(",")]
                tfn_map_by_item[it][l]=TFN(a,b,c)

# Run
if st.button("▶ Run analysis") and df is not None and len(latents)==2:
    idx={}
    for nm,items in latents.items():
        mat=df_to_tfn_matrix(df,items,tfn_map_by_item,levels_by_item)
        idx[nm]=fuzzy_topsis_cc(mat,is_benefit_by[nm],weights_by[nm])
    xnm,ynm=list(latents.keys()); x,y=idx[xnm],idx[ynm]

    # thresholds
    x_thr=np.mean(x); y_thr=np.mean(y)
    res=pd.DataFrame({f"idx_{xnm}":x,f"idx_{ynm}":y})

    # labels
    res["Classic"]=apostle_quadrants(x,y,x_thr,y_thr,"Apostles","Mercenaries","Hostages","Defectors")
    res["Extended4x4"]=eco_extended_labels_4x4(x,y,["LowX","MedLowX","MedHighX","HighX"],["LowY","MedLowY","MedHighY","HighY"])

    st.dataframe(res)
    st.download_button("⬇ Download CSV",res.to_csv(index=False),file_name="results.csv")

    # Plots
    fig,ax=plt.subplots(); ax.scatter(x,y)
    ax.axvline(x_thr,linestyle="--"); ax.axhline(y_thr,linestyle="--")
    st.pyplot(fig)


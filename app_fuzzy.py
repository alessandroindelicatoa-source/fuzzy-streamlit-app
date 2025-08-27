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
        st.info("No passwords configured. Running open.")
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
        st.error("Incorrect username or password")
        st.stop()
    return True

check_password()

# ==============================
# 1) TFN
# ==============================
@dataclass(frozen=True)
class TFN:
    a: float; b: float; c: float
    def __post_init__(self):
        if not (self.a <= self.b <= self.c): 
            raise ValueError("TFN requires a ≤ b ≤ c.")
    def scale(self,w:float): return TFN(self.a*w,self.b*w,self.c*w)

# Predefined scales
PREDEFINED = {
    "Likert 1-4": {1:TFN(0,0,20),2:TFN(16,33,60),3:TFN(49,66,83),4:TFN(80,100,100)},
    "Likert 1-5": {1:TFN(0,0,30),2:TFN(20,30,40),3:TFN(30,50,70),4:TFN(60,70,80),5:TFN(70,100,100)},
    "Likert 1-6": {1:TFN(0,0,15),2:TFN(25,40,55),3:TFN(45,60,75),4:TFN(70,80,90),5:TFN(85,100,100),6:TFN(90,100,100)},
    "Likert 1-11": {i:TFN((i-1)*10,(i*10)-10,(i*10)) if i<11 else TFN(90,100,100) for i in range(1,12)}
}

# ==============================
# 2) TOPSIS
# ==============================
def _normalize_fuzzy_matrix(matrix,is_benefit):
    m,n=len(matrix),len(matrix[0])
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
                row.append(TFN(amin/x.c,amin/(x.b if x.b else 1e-9),amin/(x.a if x.a else 1e-9)))
        out.append(row)
    return out

def _apply_weights(matrix,weights):
    wsum=sum(weights); w=[wi/wsum for wi in weights]
    return [[matrix[i][j].scale(w[j]) for j in range(len(matrix[0]))] for i in range(len(matrix))]

def _fuzzy_distance(x,y): 
    return math.sqrt((x.a-y.a)**2+(x.b-y.b)**2+(x.c-y.c)**2)

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
    cc=d_minus/(d_plus+d_minus+1e-12)
    return np.clip(cc,0,1)

def df_to_tfn_matrix(df,cols,tfn_map,levels):
    m=df.shape[0]; mat=[]
    for i in range(m):
        row=[]
        for c in cols:
            v=df.iloc[i][c]
            try: iv=int(v)
            except: iv=None
            if iv not in levels: iv=int(np.mean(levels)) # imputación media
            row.append(tfn_map[iv])
        mat.append(row)
    return mat

# ==============================
# 3) Quadrants
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
# 4) UI
# ==============================
st.set_page_config(page_title="Fuzzy TOPSIS + Apostle", layout="wide")
st.title("Fuzzy-Hybrid TOPSIS + Apostle Classic & ECO-Extended")

df=None
up=st.file_uploader("Upload CSV or Excel",type=["csv","xlsx"])
if up is not None:
    df=pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    st.dataframe(df.head())

if df is not None:
    st.sidebar.header("Latent variables")
    all_cols=list(df.columns)
    lname_x=st.sidebar.text_input("Latent X name",value="LatX")
    items_x=st.sidebar.multiselect("Items for X",all_cols)
    lname_y=st.sidebar.text_input("Latent Y name",value="LatY")
    items_y=st.sidebar.multiselect("Items for Y",all_cols)
    scale_choice=st.sidebar.selectbox("Scale",list(PREDEFINED.keys()))
    tfn_map,levels=PREDEFINED[scale_choice],list(PREDEFINED[scale_choice].keys())

    # Quadrant naming
    st.sidebar.header("Quadrant naming")
    AA=st.sidebar.text_input("Classic: x≥thr & y≥thr",value="Apostles")
    AB=st.sidebar.text_input("Classic: x≥thr & y<thr",value="Mercenaries")
    BA=st.sidebar.text_input("Classic: x<thr & y≥thr",value="Hostages")
    BB=st.sidebar.text_input("Classic: x<thr & y<thr",value="Defectors")
    x_names=[st.sidebar.text_input(f"X name {i}",v) for i,v in enumerate(["LowX","MedLowX","MedHighX","HighX"])]
    y_names=[st.sidebar.text_input(f"Y name {i}",v) for i,v in enumerate(["LowY","MedLowY","MedHighY","HighY"])]
    custom16={}
    for iy in range(4):
        for ix in range(4):
            default=f"{x_names[ix]}|{y_names[iy]}"
            custom16[(ix,iy)]=st.sidebar.text_input(f"ECO cell X{ix+1}-Y{iy+1}",default)

    # Grouping
    group_cols=st.sidebar.multiselect("Group by columns",all_cols)

    if st.button("Run analysis") and items_x and items_y:
        idx={}
        for nm,items in {lname_x:items_x,lname_y:items_y}.items():
            mat=df_to_tfn_matrix(df,items,tfn_map,levels)
            idx[nm]=fuzzy_topsis_cc(mat,[True]*len(items),[1.0]*len(items))
        x,y=idx[lname_x],idx[lname_y]
        x_thr,y_thr=np.mean(x),np.mean(y)
        res=pd.DataFrame({f"idx_{lname_x}":x,f"idx_{lname_y}":y})
        res["Classic"]=apostle_quadrants(x,y,x_thr,y_thr,AA,AB,BA,BB)
        res["Extended4x4"]=eco_extended_labels_4x4(x,y,x_names,y_names)
        st.subheader("Results (individual)")
        st.dataframe(res)

        # Plots
        fig,ax=plt.subplots(); ax.scatter(x,y)
        ax.axvline(x_thr,ls="--"); ax.axhline(y_thr,ls="--")
        st.pyplot(fig)

        # Group results
        if group_cols:
            for gcol in group_cols:
                st.subheader(f"Grouped by {gcol}")
                grouped=[]
                for cat,subdf in df.groupby(gcol):
                    gx_items,gy_items=items_x,items_y
                    z_x=fuzzy_topsis_cc(df_to_tfn_matrix(subdf,gx_items,tfn_map,levels),[True]*len(gx_items))
                    z_y=fuzzy_topsis_cc(df_to_tfn_matrix(subdf,gy_items,tfn_map,levels),[True]*len(gy_items))
                    gx,gy=np.mean(z_x),np.mean(z_y)
                    label_classic=(AA if gx>=x_thr and gy>=y_thr else AB if gx>=x_thr else BA if gy>=y_thr else BB)
                    lx,ly=np.array(eco_fuzzy_sets_4(gx)),np.array(eco_fuzzy_sets_4(gy))
                    eco_label=custom16[(lx.argmax(),ly.argmax())]
                    grouped.append({gcol:cat,f"idx_{lname_x}":gx,f"idx_{lname_y}":gy,"Classic":label_classic,"Extended4x4":eco_label})
                st.dataframe(pd.DataFrame(grouped))

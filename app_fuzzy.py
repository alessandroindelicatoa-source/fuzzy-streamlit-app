# app_fuzzy.py
# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
import streamlit as st

# ==============================
# 1) Triangular fuzzy numbers (TFN)
# ==============================
@dataclass(frozen=True)
class TFN:
    a: float; b: float; c: float
    def __post_init__(self):
        if not (self.a <= self.b <= self.c):
            raise ValueError("TFN requires a ≤ b ≤ c.")
    def scale(self, w: float) -> "TFN":
        return TFN(self.a*w, self.b*w, self.c*w)

# ===== Preset TFN mappings =====
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
# 4) Streamlit App
# ==============================
st.set_page_config(page_title="Fuzzy TOPSIS + Apostle", layout="wide")
st.title("Fuzzy-Hybrid TOPSIS + Apostle Model")

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

    st.sidebar.header("Scales per item")
    tfn_map_by_item={}; levels_by_item={}
    for it in items_x+items_y:
        sc_choice=st.sidebar.selectbox(f"Scale for {it}",["Likert1-4","Likert1-5","Likert1-6","Likert1-7","Likert1-10","Likert1-11","Linear","Manual"],key=f"sc_{it}")
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

    if st.button("Run analysis") and items_x and items_y:
        idx={}; ideal_solutions={}
        for nm,items in {lname_x:items_x,lname_y:items_y}.items():
            mat=[]
            for it in items:
                mat_item=df_to_tfn_matrix(df,[it],tfn_map_by_item[it],levels_by_item[it])
                if not mat: mat=[[x] for x in mat_item]
                else:
                    for r,row in enumerate(mat_item): mat[r].append(row[0])
            idx[nm], fpis, fnis = fuzzy_topsis_cc(mat,[True]*len(items),[1.0]*len(items))
            ideal_solutions[nm]={"PIS":fpis,"NIS":fnis}
        x,y=idx[lname_x],idx[lname_y]
        x_thr,y_thr=np.mean(x),np.mean(y)
        res=pd.DataFrame({f"idx_{lname_x}":x,f"idx_{lname_y}":y})
        res["Classic"]=apostle_quadrants(x,y,x_thr,y_thr,AA,AB,BA,BB)
        res["Extended4x4"]=eco_extended_labels_4x4(x,y,x_names,y_names)
        st.subheader("Results (individual)")
        st.dataframe(res)

        st.subheader("Ideal solutions (PIS/NIS)")
        st.json({nm:{"PIS":[(t.a,t.b,t.c) for t in v["PIS"]],"NIS":[(t.a,t.b,t.c) for t in v["NIS"]]} for nm,v in ideal_solutions.items()})

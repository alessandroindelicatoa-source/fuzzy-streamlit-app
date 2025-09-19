# app_fuzzy.py
# -*- coding: utf-8 -*-
"""
Fuzzy-Hybrid TOPSIS App (Final, completo)
-----------------------------------------
Incluye:
1) TOPSIS individual (tabla simple de índices TOPSIS por individuo).
2) TOPSIS por grupo (tabla simple de índices TOPSIS por grupo, sin PIS/NIS extra).
3) Tabla PIS/NIS global entre todos los grupos (una sola tabla por variable y agrupación).
4) Classic Apostle (4 cuadrantes con umbrales fijos y labels editables).
5) Extended Apostle (FCM + α → 16 categorías).
6) Probability ratios (con covariables).
7) Botones para exportar CSV de resultados individuales y de grupo.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# Fuzzy C-Means (solo para Extended Apostle)
import skfuzzy as fuzz

# ==============================
# TFN y escalas
# ==============================
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

# ==============================
# TOPSIS individual (fuzzy-hybrid)
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
# TOPSIS por grupo (índice) + tabla PIS/NIS GLOBAL
# ==============================
def group_topsis_and_ideals(
    df: pd.DataFrame,
    items: List[str],
    tfn_maps: Dict[str,Dict[int,TFN]],
    levels_by_item: Dict[str,List[int]],
    group_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve:
      - group_df: tabla simple con TOPSIS por grupo (ordenado)
      - ideals_df (GLOBAL): una sola tabla por variable y agrupación con PIS/NIS por ítem,
        mostrando qué grupo alcanza PIS y NIS. (No se repite por grupo; es global entre todos los grupos)
    """
    # Agregar TFN por grupo → defuzz (Buckley) → matriz V (groups x items)
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
            row[it]=defuzz_buckley(TFN(a,b,c))
        agg[g]=row

    V=pd.DataFrame.from_dict(agg, orient="index")[items]
    # PIS/NIS por ítem (GLOBAL sobre todos los grupos)
    pis=V.max(axis=0); nis=V.min(axis=0)

    # Distancias y TOPSIS por grupo
    S_plus=np.sqrt(((V - pis)**2).sum(axis=1))
    S_minus=np.sqrt(((V - nis)**2).sum(axis=1))
    topsis=(S_minus/(S_plus+S_minus+1e-12)).clip(0,1)
    group_df=pd.DataFrame({"Group":V.index, "TOPSIS": topsis.values}).sort_values("TOPSIS", ascending=False)

    # Tabla PIS/NIS GLOBAL (una sola) → quién alcanza PIS/NIS en cada ítem
    rows=[]
    for it in items:
        rows.append({
            "Item": it,
            "PIS": round(float(V[it].max()), 4),
            "PIS_Group": V[it].idxmax(),
            "NIS": round(float(V[it].min()), 4),
            "NIS_Group": V[it].idxmin()
        })
    ideals_df=pd.DataFrame(rows, columns=["Item","PIS","PIS_Group","NIS","NIS_Group"])

    return group_df, ideals_df

# ==============================
# Classic Apostle
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
# Extended Apostle (FCM + α)
# ==============================
def fuzzy_cmeans_memberships(values: np.ndarray, c=3, m=2.0, error=1e-6, maxiter=1000, seed=42):
    data=np.array(values, dtype=float).reshape(1,-1)
    cntr,u,_,_,_,_,_=fuzz.cluster.cmeans(data,c=c,m=m,error=error,maxiter=maxiter,seed=seed)
    order=np.argsort(cntr)   # least, inter, most
    least, inter, most = order[0], order[1], order[2]
    U=np.vstack([u[most],u[least],u[inter]]).T  # (most, least, inter)
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
# Probability ratios
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
st.set_page_config(page_title="Fuzzy TOPSIS Final", layout="wide")
st.title("Fuzzy-Hybrid TOPSIS + Group TOPSIS + Apostle (Classic/Extended) + Ratios")

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

    # Escalas por ítem
    st.sidebar.header("Escalas por ítem")
    tfn_map_by_item={}; levels_by_item={}
    for it in items_x+items_y:
        sc_choice=st.sidebar.selectbox(f"Escala para {it}",
            ["Likert1-4","Likert1-5","Likert1-6","Likert1-7","Likert1-10","Likert1-11","Linear","Manual"],
            key=f"sc_{it}")
        if sc_choice=="Likert1-4": tfn_map_by_item[it]=likert_map_1_4(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-5": tfn_map_by_item[it]=likert_map_1_5(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-6": tfn_map_by_item[it]=likert_map_1_6(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-7": tfn_map_by_item[it]=likert_map_1_7(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-10": tfn_map_by_item[it]=likert_map_1_10(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-11": tfn_map_by_item[it]=likert_map_1_11(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Linear":
            lv=[int(x) for x in st.sidebar.text_input(f"Niveles para {it}", value="1,2,3,4,5", key=f"lv_{it}").split(",")]
            tfn_map_by_item[it]=linear_tfn_map(lv); levels_by_item[it]=lv
        else:
            lv=[int(x) for x in st.sidebar.text_input(f"Niveles para {it}", value="1,2,3,4,5", key=f"lvman_{it}").split(",")]
            levels_by_item[it]=lv; tfn_map_by_item[it]={}
            for l in lv:
                abctxt=st.sidebar.text_input(f"{it}-{l} TFN", value="0,0,25", key=f"tfn_{it}_{l}")
                a,b,c=[float(x) for x in abctxt.split(",")]
                tfn_map_by_item[it][l]=TFN(a,b,c)

    # Apostle Classic (umbrales) + nombres
    st.sidebar.header("Classic Apostle (umbrales)")
    thr_x=st.sidebar.slider("Threshold X", 0.0, 1.0, 0.5, 0.01)
    thr_y=st.sidebar.slider("Threshold Y", 0.0, 1.0, 0.5, 0.01)
    lab_a = st.sidebar.text_input("(X≥thr & Y≥thr)", "Quadrant A")
    lab_b = st.sidebar.text_input("(X<thr & Y≥thr)", "Quadrant B")
    lab_c = st.sidebar.text_input("(X<thr & Y<thr)", "Quadrant C")
    lab_d = st.sidebar.text_input("(X≥thr & Y<thr)", "Quadrant D")

    # Extended Apostle
    alpha=st.sidebar.slider("Extended FCM alpha", 0.1, 0.9, 0.5, 0.05)

    if st.button("Run analysis") and items_x and items_y:
        # TOPSIS individual (fuzzy-hybrid)
        idx={}
        for nm,items in {lname_x:items_x, lname_y:items_y}.items():
            mat=[]
            for it in items:
                m_item=df_to_tfn_matrix(df,[it],tfn_map_by_item[it],levels_by_item[it])
                if not mat:
                    mat=[[row[0]] for row in m_item]
                else:
                    for r,row in enumerate(m_item):
                        mat[r].append(row[0])
            idx[nm] = fuzzy_topsis_cc(mat,[True]*len(items),[1.0]*len(items))

        x=np.array(idx[lname_x]); y=np.array(idx[lname_y])

        # Classic Apostle
        classic = classic_apostle_threshold(x,y,thr_x,thr_y,[lab_a,lab_b,lab_c,lab_d])

        # Extended Apostle
        Ux,_=fuzzy_cmeans_memberships(x); Uy,_=fuzzy_cmeans_memberships(y)
        extended = extended_apostle_from_memberships(Ux,Uy,alpha=alpha)

        res=pd.DataFrame({f"{lname_x}":x, f"{lname_y}":y, "Classic":classic, "Extended":extended})

        st.session_state['analysis_done']=True
        st.session_state['results']=res
        st.session_state['idx']={lname_x:x, lname_y:y}
        st.session_state['tfn_map_by_item']=tfn_map_by_item
        st.session_state['levels_by_item']=levels_by_item
        st.session_state['items_x']=items_x
        st.session_state['items_y']=items_y
        st.session_state['lname_x']=lname_x
        st.session_state['lname_y']=lname_y

    if st.session_state.get('analysis_done'):
        res: pd.DataFrame = st.session_state['results']
        lname_x = st.session_state['lname_x']; lname_y = st.session_state['lname_y']
        items_x = st.session_state['items_x']; items_y = st.session_state['items_y']
        tfn_map_by_item = st.session_state['tfn_map_by_item']; levels_by_item = st.session_state['levels_by_item']

        st.subheader("📊 TOPSIS individual")
        st.dataframe(res)

        # Export CSV individual
        csv_ind=res.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV (individual)", data=csv_ind, file_name="topsis_individual.csv")

        # Group TOPSIS + PIS/NIS GLOBAL + Ratios
        st.sidebar.header("Group TOPSIS & Ratios")
        group_cols=st.sidebar.multiselect("Agrupar por columnas", list(df.columns), key="group_cols")

        if group_cols:
            tabs=st.tabs(["Group TOPSIS", "PIS/NIS (global)", "Probability ratios"])
            g_cache={}

            # Group TOPSIS (solo índice)
            with tabs[0]:
                for lat_name, items in [(lname_x,items_x),(lname_y,items_y)]:
                    if not items: continue
                    st.markdown(f"### {lat_name} — Group TOPSIS (índice por grupo)")
                    for gcol in group_cols:
                        gdf, ideals = group_topsis_and_ideals(df, items, tfn_map_by_item, levels_by_item, gcol)
                        st.markdown(f"**Agrupación por:** `{gcol}`")
                        st.dataframe(gdf)
                        # Guardar para exportación y para pestaña de PIS/NIS global
                        g_cache[(lat_name,gcol)] = (gdf, ideals)

                        # Export CSV por cada tabla de grupos
                        csv_group = gdf.to_csv(index=False).encode("utf-8")
                        st.download_button(f"⬇️ Descargar CSV (Group TOPSIS) — {lat_name} / {gcol}",
                                           data=csv_group, file_name=f"group_topsis__{lat_name}__by_{gcol}.csv")

            # PIS/NIS GLOBAL (una sola tabla por variable y agrupación)
            with tabs[1]:
                for lat_name, items in [(lname_x,items_x),(lname_y,items_y)]:
                    if not items: continue
                    st.markdown(f"### {lat_name} — PIS/NIS global (entre todos los grupos)")
                    for gcol in group_cols:
                        ideals = g_cache[(lat_name,gcol)][1]
                        st.markdown(f"**Agrupación por:** `{gcol}`")
                        st.dataframe(ideals)
                        csv_ideals = ideals.to_csv(index=False).encode("utf-8")
                        st.download_button(f"⬇️ Descargar CSV (PIS-NIS GLOBAL) — {lat_name} / {gcol}",
                                           data=csv_ideals, file_name=f"pis_nis_global__{lat_name}__by_{gcol}.csv")

            # Probability ratios (usando columna 'Classic' como variable A)
            with tabs[2]:
                full=pd.concat([df.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
                ratios = conditional_probability_ratios_by_level(full, "Classic", group_cols, max_levels=12, n_boot=1000)
                st.dataframe(ratios)
                csv_rat = ratios.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Descargar CSV (Probability Ratios)", data=csv_rat, file_name="probability_ratios.csv")

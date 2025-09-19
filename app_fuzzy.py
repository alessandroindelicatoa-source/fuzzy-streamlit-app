# app_fuzzy.py
# -*- coding: utf-8 -*-
"""
Fuzzy-Hybrid TOPSIS + Apostle (Classic per fixed threshold) + Extended (Fuzzy C-Means) 
+ Group TOPSIS (with PIS/NIS) + Conditional Probability Ratios (for same group columns).

- Individual TOPSIS (two latents X,Y)
- Classic Apostle (4 quadrants) using a configurable threshold (default 0.5), NOT by means
- Extended Apostle: 16 categories via fuzzy c-means memberships (most / least / intermediate) + alpha rule
- Group TOPSIS per selected columns, with PIS/NIS table per item (paper-compliant)
- Probability ratios on same group columns
- session_state persistence
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# fuzzy c-means
import skfuzzy as fuzz

# ==============================
# Triangular fuzzy numbers (TFN)
# ==============================
@dataclass(frozen=True)
class TFN:
    a: float; b: float; c: float
    def __post_init__(self):
        if not (self.a <= self.b <= self.c):
            raise ValueError("TFN requires a ≤ b ≤ c.")
    def scale(self, w: float) -> "TFN":
        return TFN(self.a*w, self.b*w, self.c*w)

# ---- Preset TFN mappings ----
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
# Helpers
# ==============================
def ensure_tfn(x):
    if isinstance(x, TFN): return x
    if isinstance(x, (int, float)): return TFN(x, x, x)
    if isinstance(x, (list, tuple)) and len(x)==3: return TFN(*x)
    raise ValueError(f"Cannot convert {x} to TFN")

def defuzz_buckley(x: TFN) -> float:
    # ĉ = (a + 2b + c)/4
    return (x.a + 2*x.b + x.c) / 4.0

# ==============================
# Individual-level Fuzzy TOPSIS
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
    return np.clip(d_minus/(d_plus+d_minus+1e-12),0,1), fpis, fnis

def df_to_tfn_matrix(df: pd.DataFrame, cols: List[str], tfn_map: Dict[int,TFN], levels: List[int]):
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
# Classic & Extended Apostle
# ==============================
def classic_apostle_threshold(x, y, thr_x=0.5, thr_y=0.5):
    # Fixed thresholds (not by means). Names per figure.
    labels=[]
    for xi, yi in zip(x,y):
        if xi>=thr_x and yi>=thr_y: labels.append("Environmental Consistent")
        elif xi<thr_x and yi>=thr_y: labels.append("Environmental Inconsistent")
        elif xi<thr_x and yi<thr_y: labels.append("Environmental Negationist")
        else: labels.append("Environmental Conscientious")  # xi>=thr_x & yi<thr_y
    return labels

def fuzzy_cmeans_memberships(values: np.ndarray, c=3, m=2.0, error=1e-6, maxiter=1000, seed=42):
    """Return memberships ordered as (most, least, intermediate) and centers in same order."""
    # skfuzzy expects shape (features, samples)
    data = np.array(values, dtype=float).reshape(1, -1)
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data=data, c=c, m=m, error=error, maxiter=maxiter, init=None, seed=seed
    )
    order = np.argsort(cntr)  # ascending: least, intermediate, most
    least_idx, inter_idx, most_idx = order[0], order[1], order[2]
    U_most = u[most_idx, :]
    U_least = u[least_idx, :]
    U_inter = u[inter_idx, :]
    U_stacked = np.vstack([U_most, U_least, U_inter]).T  # (n,3)
    centers_ordered = np.array([cntr[most_idx], cntr[least_idx], cntr[inter_idx]])
    return U_stacked, centers_ordered

def extended_apostle_16_from_memberships(Ux: np.ndarray, Uy: np.ndarray, alpha: float=0.5):
    """Ux, Uy: (n,3) memberships in order (most, least, intermediate)"""
    n = Ux.shape[0]
    labels=[]
    def code(u):
        # u: (most, least, inter)
        if u[1] >= alpha: return 1           # least
        if (u<alpha).all(): return 2         # undetermined
        if u[2] >= alpha: return 3           # intermediate
        if u[0] >= alpha: return 4           # most
        return 2
    for i in range(n):
        cx = code(Ux[i]); cy = code(Uy[i])
        labels.append(f"({cx},{cy})")
    return labels

# ==============================
# Probability Ratios (bootstrap)
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

def conditional_probability_ratios_by_level(df: pd.DataFrame, quad_col: str, covar_cols: List[str], max_levels: int = 12, n_boot: int = 1000):
    rows = []
    for cov in covar_cols:
        series = df[cov]
        unique_vals = series.dropna().unique()
        levels = [(str(v), v) for v in sorted(unique_vals, key=lambda x: str(x))[:max_levels]]
        if len(unique_vals) > max_levels:
            levels.append(("OTHER", None))
        for q in df[quad_col].dropna().unique():
            A = (df[quad_col] == q).astype(int).to_numpy()
            for name, val in levels:
                if val is None:
                    B = (~series.isin([v for _, v in levels[:-1]])).astype(int).to_numpy()
                else:
                    B = (series == val).astype(int).to_numpy()
                mean, lo, hi = _prob_ratio_bootstrap(A, B, n_boot=n_boot)
                rows.append({"Quadrant": str(q), "Covariate": cov, "Level": name, "Mean": round(mean,3), "CI_low": round(lo,3), "CI_high": round(hi,3)})
    return pd.DataFrame(rows)

# ==============================
# Group-level TOPSIS with PIS/NIS (paper method)
# ==============================
def group_topsis_with_pis_nis(df: pd.DataFrame, items: List[str], tfn_maps: Dict[str,Dict[int,TFN]], levels_by_item: Dict[str,List[int]], group_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 1) TFN mean per group & item
    agg = {}
    groups = list(df[group_col].dropna().unique())
    for g in groups:
        dfg = df[df[group_col]==g]
        row = {}
        for it in items:
            levels = levels_by_item[it]; mapping = tfn_maps[it]
            median_level = int(np.median(levels))
            fallback = mapping[median_level]
            tfns = []
            for v in dfg[it].tolist():
                try: iv=int(v)
                except: iv=None
                if iv not in levels: iv = median_level
                tfns.append(mapping.get(iv, fallback))
            if len(tfns)==0:
                row[it]=TFN(0,0,0)
            else:
                a = np.mean([t.a for t in tfns]); b = np.mean([t.b for t in tfns]); c = np.mean([t.c for t in tfns])
                row[it]=TFN(a,b,c)
        agg[g]=row

    # 2) Defuzzify matrix
    groups_sorted = sorted(agg.keys(), key=lambda x: str(x))
    V = pd.DataFrame([[defuzz_buckley(agg[g][it]) for it in items] for g in groups_sorted], index=groups_sorted, columns=items)

    # 3) PIS/NIS per item
    pis = V.max(axis=0); nis = V.min(axis=0)

    # 4) Distances + TOPSIS per group
    S_plus = np.sqrt(((V - pis)**2).sum(axis=1))
    S_minus = np.sqrt(((V - nis)**2).sum(axis=1))
    topsis = (S_minus / (S_plus + S_minus + 1e-12)).clip(0,1)

    group_df = pd.DataFrame({"Group": groups_sorted, "TOPSIS": topsis.values})

    # 5) Ideal table
    rows=[]
    for it in items:
        g_pis = V[it].idxmax(); v_pis = V[it].max()
        g_nis = V[it].idxmin(); v_nis = V[it].min()
        rows.append({"Item": it, "PIS_Group": g_pis, "PIS": round(float(v_pis),2), "NIS_Group": g_nis, "NIS": round(float(v_nis),2)})
    ideal_df = pd.DataFrame(rows)
    return group_df, ideal_df

# ==============================
# Streamlit App (with session_state)
# ==============================
st.set_page_config(page_title="Fuzzy TOPSIS + Apostle", layout="wide")
st.title("Fuzzy-Hybrid TOPSIS + Apostle (Classic threshold + Extended FCM) + Group TOPSIS + Ratios")

# Reset
if st.sidebar.button("🔄 Reset analysis"):
    for k in ["analysis_done","results","ideal","idx","group_outputs","Ux","Uy"]:
        if k in st.session_state: del st.session_state[k]

# Upload
df=None
up=st.file_uploader("Upload CSV or Excel",type=["csv","xlsx"])
if up is not None:
    df=pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    st.write("Preview:")
    st.dataframe(df.head())

if df is not None:
    st.sidebar.header("Latent variables")
    all_cols=list(df.columns)
    lname_x=st.sidebar.text_input("Latent X name",value="EA")
    items_x=st.sidebar.multiselect("Items for X",all_cols, key="items_x")
    lname_y=st.sidebar.text_input("Latent Y name",value="STA")
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

    st.sidebar.header("Apostle settings")
    thr_x = st.sidebar.slider("Classic threshold X", 0.0, 1.0, 0.5, 0.01)
    thr_y = st.sidebar.slider("Classic threshold Y", 0.0, 1.0, 0.5, 0.01)
    alpha = st.sidebar.slider("Extended FCM alpha", 0.1, 0.9, 0.5, 0.05)

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
            ideal_solutions[nm] = {"PIS": fpis, "NIS": fnis}

        x=np.array(idx[lname_x]); y=np.array(idx[lname_y])

        # Classic (threshold)
        classic = classic_apostle_threshold(x, y, thr_x, thr_y)

        # Extended (FCM memberships)
        Ux, _ = fuzzy_cmeans_memberships(x)
        Uy, _ = fuzzy_cmeans_memberships(y)
        ext_labels = extended_apostle_16_from_memberships(Ux, Uy, alpha=alpha)

        res=pd.DataFrame({f"{lname_x}":x,f"{lname_y}":y,"Classic":classic,"Extended":ext_labels})

        st.session_state['analysis_done']=True
        st.session_state['results']=res
        st.session_state['ideal']=ideal_solutions
        st.session_state['idx']={lname_x:x, lname_y:y}
        st.session_state['Ux']=Ux; st.session_state['Uy']=Uy

    if st.session_state.get('analysis_done'):
        res: pd.DataFrame = st.session_state['results']
        st.subheader("📊 Individual results")
        st.dataframe(res)

        # ================== Group TOPSIS (paper method) ==================
        st.sidebar.header("Group TOPSIS & Ratios (same columns)")
        group_cols = st.sidebar.multiselect("Group by columns", list(df.columns), key="group_cols")
        if group_cols:
            tabs = st.tabs(["Group TOPSIS", "Ideal solutions (PIS/NIS)", "Probability ratios"])
            group_outputs = {}
            with tabs[0]:
                for lat_name, items in [(lname_x, items_x), (lname_y, items_y)]:
                    if not items: 
                        continue
                    st.markdown(f"### {lat_name} — Group TOPSIS (per {', '.join(group_cols)})")
                    for gcol in group_cols:
                        gdf, ideal_df = group_topsis_with_pis_nis(df, items, tfn_map_by_item, levels_by_item, gcol)
                        st.markdown(f"**Grouping by:** `{gcol}`")
                        st.dataframe(gdf.sort_values('TOPSIS', ascending=False))
                        group_outputs[(lat_name,gcol)] = (gdf, ideal_df)
            with tabs[1]:
                for lat_name, items in [(lname_x, items_x), (lname_y, items_y)]:
                    if not items: continue
                    for gcol in group_cols:
                        ideal_df = group_outputs[(lat_name,gcol)][1]
                        st.markdown(f"**{lat_name} — Ideal solutions by item (grouping: `{gcol}`)**")
                        st.dataframe(ideal_df)
            with tabs[2]:
                res_full = pd.concat([df.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
                ratios = conditional_probability_ratios_by_level(res_full, "Classic", group_cols, max_levels=12, n_boot=1000)
                st.dataframe(ratios)

        csv = res.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download individual results CSV", data=csv, file_name="topsis_individual_results.csv")

# app_fuzzy.py — FINAL (English)
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
import streamlit as st

# ==============================
# Password Gate (optional)
# ==============================
def gate():
    """
    Simple username/password gate using st.secrets['passwords'].
    To enable, create a .streamlit/secrets.toml like:
      [passwords]
      admin = "yourpass"
      alice = "1234"

    Remove/comment the call to gate() below to disable the login.
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True

    st.title("🔐 Access")
    st.write("Enter your username and password.")
    user = st.text_input("Username", key="auth_user")
    pwd = st.text_input("Password", type="password", key="auth_pwd")
    if st.button("Log in"):
        try:
            if "passwords" in st.secrets and user in st.secrets["passwords"]:
                if str(st.secrets["passwords"][user]) == str(pwd):
                    st.session_state.authenticated = True
                    st.success("Access granted ✅")
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            else:
                st.error("User not found in [passwords].")
        except Exception as e:
            st.error(f"Could not validate access: {e}")
    st.stop()

# ==============================
# 1) Triangular Fuzzy Numbers (TFN) + Scales
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
        1:  TFN(0, 0, 10),  2:  TFN(0, 10, 20), 3:  TFN(10, 20, 30),
        4:  TFN(20, 30, 40),5:  TFN(30, 40, 50),6:  TFN(50, 60, 70),
        7:  TFN(60, 70, 80),8:  TFN(70, 80, 90),9:  TFN(80, 90, 100),
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
# 2) Fuzzy–Hybrid TOPSIS (individual)
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
                row.append(TFN(x.a/denom, x.b/denom, x.c/denom))
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
# 3) Group TOPSIS + Global PIS/NIS
# ==============================
def group_defuzz_means(dfg: pd.DataFrame, items: List[str],
                       tfn_maps: Dict[str,Dict[int,TFN]], levels_by_item: Dict[str,List[int]]) -> Dict[str, float]:
    row = {}
    for it in items:
        levels = levels_by_item[it]; mapping = tfn_maps[it]
        med = int(np.median(levels)); fallback = mapping[med]
        tfns = []
        for v in dfg[it].tolist():
            try: iv = int(v)
            except: iv = None
            if iv not in levels: iv = med
            tfns.append(mapping.get(iv, fallback))
        if tfns:
            a = np.mean([t.a for t in tfns]); b = np.mean([t.b for t in tfns]); c = np.mean([t.c for t in tfns])
        else:
            a=b=c=0.0
        row[it] = defuzz_buckley(TFN(a,b,c))
    return row

def unified_group_topsis_table(df: pd.DataFrame,
                               items_x: List[str], items_y: List[str],
                               tfn_maps: Dict[str,Dict[int,TFN]],
                               levels_by_item: Dict[str,List[int]],
                               group_cols: List[str],
                               name_x: str, name_y: str) -> pd.DataFrame:
    """
    Returns one flat table:
    Variable | Item | Topsis-LatX | Topsis-LatY
    """
    out_rows = []
    if not group_cols:
        out_rows.append({"Variable":"ALL","Item":"ALL","Topsis-LatX":0.5,"Topsis-LatY":0.5})
        return pd.DataFrame(out_rows, columns=["Variable","Item","Topsis-LatX","Topsis-LatY"])

    for gcol in group_cols:
        groups = [g for g,_ in df.groupby(gcol, dropna=True)]
        Vx = pd.DataFrame([group_defuzz_means(df[df[gcol]==g], items_x, tfn_maps, levels_by_item) for g in groups]) if items_x else pd.DataFrame(index=range(len(groups)))
        Vy = pd.DataFrame([group_defuzz_means(df[df[gcol]==g], items_y, tfn_maps, levels_by_item) for g in groups]) if items_y else pd.DataFrame(index=range(len(groups)))

        def _topsis_from_matrix(V):
            if V.empty:
                return np.array([np.nan]*len(groups))
            pis = V.max(axis=0); nis = V.min(axis=0)
            S_plus = np.sqrt(((V - pis)**2).sum(axis=1))
            S_minus = np.sqrt(((V - nis)**2).sum(axis=1))
            return (S_minus / (S_plus + S_minus + 1e-12)).clip(0,1).to_numpy()

        tx = _topsis_from_matrix(Vx)
        ty = _topsis_from_matrix(Vy)

        for i, g in enumerate(groups):
            out_rows.append({
                "Variable": gcol,
                "Item": str(g),
                "Topsis-LatX": round(float(tx[i]) if not np.isnan(tx[i]) else np.nan, 6) if len(tx) else np.nan,
                "Topsis-LatY": round(float(ty[i]) if not np.isnan(ty[i]) else np.nan, 6) if len(ty) else np.nan
            })

    return pd.DataFrame(out_rows, columns=["Variable","Item","Topsis-LatX","Topsis-LatY"])

def global_pis_nis_across_items(df: pd.DataFrame, items_all: List[str],
                                tfn_maps: Dict[str,Dict[int,TFN]],
                                levels_by_item: Dict[str,List[int]],
                                group_cols: List[str]) -> pd.DataFrame:
    """
    ONE table across ALL items (X ∪ Y), not per latent.
    Columns -> Item | PIS | PIS_Key | NIS | NIS_Key
    """
    rows = []
    for it in items_all:
        levels = levels_by_item[it]; mapping = tfn_maps[it]
        med = int(np.median(levels)); fallback = mapping[med]
        best_val = -np.inf; best_key = None
        worst_val = np.inf; worst_key = None

        if not group_cols:
            tfns = []
            for v in df[it].tolist():
                try: iv = int(v)
                except: iv = None
                if iv not in levels: iv = med
                tfns.append(mapping.get(iv, fallback))
            if tfns:
                a = np.mean([t.a for t in tfns]); b = np.mean([t.b for t in tfns]); c = np.mean([t.c for t in tfns])
                val = defuzz_buckley(TFN(a,b,c))
            else:
                val = 0.0
            best_val = val; best_key = "ALL"
            worst_val = val; worst_key = "ALL"
        else:
            for gcol in group_cols:
                for gval, dfg in df.groupby(gcol, dropna=True):
                    tfns = []
                    for v in dfg[it].tolist():
                        try: iv = int(v)
                        except: iv = None
                        if iv not in levels: iv = med
                        tfns.append(mapping.get(iv, fallback))
                    if tfns:
                        a = np.mean([t.a for t in tfns]); b = np.mean([t.b for t in tfns]); c = np.mean([t.c for t in tfns])
                        val = defuzz_buckley(TFN(a,b,c))
                    else:
                        val = 0.0
                    key = f"{gcol}={gval}"
                    if val > best_val:
                        best_val = val; best_key = key
                    if val < worst_val:
                        worst_val = val; worst_key = key

        rows.append({
            "Item": it,
            "PIS": round(float(best_val), 4),
            "PIS_Key": str(best_key) if best_key is not None else "",
            "NIS": round(float(worst_val), 4),
            "NIS_Key": str(worst_key) if worst_key is not None else ""
        })
    return pd.DataFrame(rows, columns=["Item","PIS","PIS_Key","NIS","NIS_Key"])

# ==============================
# 4) Quadrants (Classic + Extended ECO 4×4)
# ==============================
def classic_apostle_threshold(x, y, thr_x=0.5, thr_y=0.5):
    out=[]
    for xi, yi in zip(x, y):
        if xi>=thr_x and yi>=thr_y: out.append("Apostles")
        elif xi<thr_x and yi>=thr_y: out.append("Hostages")
        elif xi<thr_x and yi<thr_y: out.append("Defectors")
        else: out.append("Mercenaries")
    return out

def eco_fuzzy_sets_4(val):
    low=medlow=medhigh=high=0.0
    if val<=0.33: low=1-val/0.33
    if val>=0.66: high=(val-0.66)/0.34 if val<=1 else 1
    if 0<=val<=0.66: medlow=1-abs(val-0.33)/0.33
    if 0.33<=val<=1: medhigh=1-abs(val-0.66)/0.34
    return (max(low,0),max(medlow,0),max(medhigh,0),max(high,0))

def eco_extended_label(xi, yi, alpha,
                       xn=["LowX","MedLowX","MedHighX","HighX"],
                       yn=["LowY","MedLowY","MedHighY","HighY"]):
    # argmax selection; alpha is available if later you want hard gating
    lx=np.array(eco_fuzzy_sets_4(xi)); ly=np.array(eco_fuzzy_sets_4(yi))
    ix = lx.argmax(); iy = ly.argmax()
    return f"{xn[ix]}|{yn[iy]}"

def eco_extended_labels_4x4(x, y, alpha,
                            xn=["LowX","MedLowX","MedHighX","HighX"],
                            yn=["LowY","MedLowY","MedHighY","HighY"]):
    return [eco_extended_label(xi, yi, alpha, xn, yn) for xi, yi in zip(x, y)]

# ==============================
# 5) Probability Ratios (with bootstrap CIs)
# ==============================
def _prob_ratio_bootstrap(A: np.ndarray, B: np.ndarray, n_boot: int = 1000, seed: int = 42):
    rng=np.random.default_rng(seed); N=len(A); vals=[]
    for _ in range(n_boot):
        idx=rng.integers(0,N,N); As=A[idx]; Bs=B[idx]
        pA=As.mean(); pB=Bs.mean(); pAB=(As & Bs).mean()
        if pA>0 and pB>0: vals.append(pAB/(pA*pB))
    if not vals: return np.nan,np.nan,np.nan
    return float(np.mean(vals)), float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))

def probability_ratios_for_target(df: pd.DataFrame, target_col: str, covar_cols: List[str], max_levels: int = 12, n_boot: int = 1000):
    rows=[]
    for cov in covar_cols:
        series=df[cov]; unique=series.dropna().unique()
        levels=[(str(v),v) for v in sorted(unique, key=lambda x: str(x))[:max_levels]]
        if len(unique)>max_levels: levels.append(("OTHER",None))
        targets = df[target_col].dropna().unique()
        for q in targets:
            A=(df[target_col]==q).astype(int).to_numpy()
            for name,val in levels:
                if val is None:
                    B=(~series.isin([v for _,v in levels[:-1]])).astype(int).to_numpy()
                else:
                    B=(series==val).astype(int).to_numpy()
                mean,lo,hi=_prob_ratio_bootstrap(A,B,n_boot=n_boot)
                rows.append({"Covariate":cov,"Group":name,"Target":str(q),
                             "Ratio":round(mean,3),"CI_low":round(lo,3),"CI_high":round(hi,3)})
    return pd.DataFrame(rows)

# ==============================
# 6) Streamlit App
# ==============================
st.set_page_config(page_title="Fuzzy-Hybrid TOPSIS Suite", layout="wide")
gate()  # comment out this line if you want open access

st.title("Fuzzy-Hybrid TOPSIS + Unified Group TOPSIS + Apostle (Classic/Extended) + Ratios")

# Upload
df=None
up=st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
if up is not None:
    df=pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    st.dataframe(df.head())

if df is not None:
    all_cols=list(df.columns)

    # Latents and items
    st.sidebar.header("Latent variables")
    lname_x=st.sidebar.text_input("Latent X name", value="LatX")
    items_x=st.sidebar.multiselect("Items for X", all_cols, key="items_x_widget")
    lname_y=st.sidebar.text_input("Latent Y name", value="LatY")
    items_y=st.sidebar.multiselect("Items for Y", all_cols, key="items_y_widget")

    # Scales per item (TFN)
    st.sidebar.header("TFN scale per item")
    tfn_map_by_item={}; levels_by_item={}
    for it in items_x+items_y:
        sc_choice=st.sidebar.selectbox(f"Scale for {it}",
            ["Likert1-4","Likert1-5","Likert1-6","Likert1-7","Likert1-10","Likert1-11","Linear","Manual"],
            key=f"sc_{it}")
        if sc_choice=="Likert1-4":
            tfn_map_by_item[it]=likert_map_1_4(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-5":
            tfn_map_by_item[it]=likert_map_1_5(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-6":
            tfn_map_by_item[it]=likert_map_1_6(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-7":
            tfn_map_by_item[it]=likert_map_1_7(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-10":
            tfn_map_by_item[it]=likert_map_1_10(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
        elif sc_choice=="Likert1-11":
            tfn_map_by_item[it]=likert_map_1_11(); levels_by_item[it]=list(tfn_map_by_item[it].keys())
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

    # Apostle parameters (Classic fixed at 0.5; Extended alpha editable)
    st.sidebar.header("Extended Apostle (alpha)")
    alpha=st.sidebar.slider("Extended FCM alpha", 0.1, 0.9, 0.5, 0.05)

    # Editable labels for Simplified Quadrants (4 targets)
    st.sidebar.header("Quadrant-4 labels (for ratios)")
    q4_hh = st.sidebar.text_input("HighX|HighY label", value="HighX|HighY")
    q4_hl = st.sidebar.text_input("HighX|LowY label",  value="HighX|LowY")
    q4_lh = st.sidebar.text_input("LowX|HighY label",  value="LowX|HighY")
    q4_ll = st.sidebar.text_input("LowX|LowY label",   value="LowX|LowY")

    # Variables for grouping and ratios
    st.sidebar.header("Analysis variables")
    group_cols=st.sidebar.multiselect("Group TOPSIS by:", all_cols, key="group_cols_widget")
    ratio_cols=st.sidebar.multiselect("Covariates for Probability Ratios:", all_cols, key="ratio_cols_widget")

    # Run
    if st.button("Run analysis") and items_x and items_y:
        idx={}
        for nm,items in {lname_x:items_x, lname_y:items_y}.items():
            mat=[]
            for it in items:
                m_item=df_to_tfn_matrix(df,[it],tfn_map_by_item[it],levels_by_item[it])
                if not mat: mat=[[row[0]] for row in m_item]
                else:
                    for r,row in enumerate(m_item): mat[r].append(row[0])
            idx[nm]=fuzzy_topsis_cc(mat,[True]*len(items),[1.0]*len(items))

        x=np.array(idx[lname_x]); y=np.array(idx[lname_y])
        classic = classic_apostle_threshold(x,y,0.5,0.5)
        extended = eco_extended_labels_4x4(x,y,alpha)

        res=pd.DataFrame({f"{lname_x}":x, f"{lname_y}":y, "Classic":classic, "Extended4x4":extended})
        st.session_state['results']=res
        st.session_state['lname_x']=lname_x; st.session_state['lname_y']=lname_y
        st.session_state['items_x']=items_x; st.session_state['items_y']=items_y
        st.session_state['tfn_map_by_item']=tfn_map_by_item; st.session_state['levels_by_item']=levels_by_item
        st.session_state['group_cols']=group_cols; st.session_state['ratio_cols']=ratio_cols
        st.session_state['q4_labels']=(q4_hh,q4_hl,q4_lh,q4_ll)
        st.success("Analysis completed.")

    # Display
    if 'results' in st.session_state:
        res=st.session_state['results']
        lname_x=st.session_state['lname_x']; lname_y=st.session_state['lname_y']
        items_x=st.session_state['items_x']; items_y=st.session_state['items_y']
        tfn_map_by_item=st.session_state['tfn_map_by_item']; levels_by_item=st.session_state['levels_by_item']
        group_cols=st.session_state['group_cols']; ratio_cols=st.session_state['ratio_cols']
        q4_hh,q4_hl,q4_lh,q4_ll = st.session_state['q4_labels']

        st.subheader("📊 Individual TOPSIS indices")
        st.dataframe(res)
        st.download_button("⬇️ Download CSV (individual)", data=res.to_csv(index=False).encode("utf-8"),
                           file_name="topsis_individual.csv")

        # ---- Global PIS/NIS (single table across ALL items)
        items_all = items_x + items_y
        st.subheader("🌐 Global PIS/NIS (single table across ALL items)")
        pis_nis_global = global_pis_nis_across_items(df, items_all, tfn_map_by_item, levels_by_item, group_cols)
        st.dataframe(pis_nis_global)
        st.download_button("⬇️ Download CSV — Global PIS/NIS",
                           data=pis_nis_global.to_csv(index=False).encode("utf-8"),
                           file_name="global_pis_nis_all_items.csv")

        # ---- Unified Group TOPSIS table (single)
        st.subheader("🧮 Group TOPSIS (single unified table)")
        st.caption("Columns: Variable (grouping variable), Item (group value), Topsis-LatX, Topsis-LatY")
        unified_groups = unified_group_topsis_table(df, items_x, items_y, tfn_map_by_item, levels_by_item, group_cols, lname_x, lname_y)
        st.dataframe(unified_groups)
        st.download_button("⬇️ Download CSV — Unified Group TOPSIS",
                           data=unified_groups.to_csv(index=False).encode("utf-8"),
                           file_name="group_topsis_unified.csv")

        # ---- Probability Ratios (Extended 16 categories)
        st.subheader("📊 Probability Ratios — Extended Apostle (16 categories)")
        full_ext = pd.concat([df.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
        if ratio_cols:
            ratios_extended = probability_ratios_for_target(full_ext, "Extended4x4", ratio_cols, max_levels=12, n_boot=1000)
            st.dataframe(ratios_extended)
            st.download_button("⬇️ Download Extended Ratios",
                               data=ratios_extended.to_csv(index=False).encode("utf-8"),
                               file_name="ratios_extended.csv")
        else:
            st.info("Select covariates to compute Extended Apostle ratios.")

        # ---- Probability Ratios (Simplified Quadrants 4, editable labels)
        st.subheader("📊 Probability Ratios — Simplified Quadrants (4 categories)")
        # Create quadrant-4 target
        thr_x=0.5; thr_y=0.5
        def q4_label(xv,yv):
            if xv>=thr_x and yv>=thr_y: return q4_hh
            if xv>=thr_x and yv<thr_y:  return q4_hl
            if xv<thr_x  and yv>=thr_y: return q4_lh
            return q4_ll

        q4 = [q4_label(xv,yv) for xv,yv in zip(res[lname_x], res[lname_y])]
        full_q4 = pd.concat([df.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
        full_q4["Quadrant4"] = q4

        if ratio_cols:
            ratios_quadrants4 = probability_ratios_for_target(full_q4, "Quadrant4", ratio_cols, max_levels=12, n_boot=1000)
            st.dataframe(ratios_quadrants4)
            st.download_button("⬇️ Download Quadrant4 Ratios",
                               data=ratios_quadrants4.to_csv(index=False).encode("utf-8"),
                               file_name="ratios_quadrants4.csv")
        else:
            st.info("Select covariates to compute Quadrant-4 ratios.")

# app_fuzzy.py
# -*- coding: utf-8 -*-
import io, math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ========== TFN y utilidades ==========
@dataclass(frozen=True)
class TFN:
    a: float; b: float; c: float
    def __post_init__(self):
        if not (self.a <= self.b <= self.c):
            raise ValueError("TFN requiere a ≤ b ≤ c.")
    def scale(self, w: float) -> "TFN":
        return TFN(self.a*w, self.b*w, self.c*w)

def likert4_default_map() -> Dict[int, TFN]:
    return {1: TFN(0,0,50), 2: TFN(30,50,70), 3: TFN(50,70,90), 4: TFN(70,100,100)}

def likert5_default_map() -> Dict[int, TFN]:
    return {1: TFN(0,0,25), 2: TFN(15,30,45), 3: TFN(40,50,60), 4: TFN(55,70,85), 5: TFN(75,100,100)}

def linear_tfn_map(levels: List[int]) -> Dict[int, TFN]:
    K = len(levels)
    if K < 2: raise ValueError("Se requieren ≥2 niveles.")
    centers = np.linspace(0,100,K)
    mapping: Dict[int, TFN] = {}
    for i, lv in enumerate(levels):
        b = float(centers[i])
        if i == 0:
            a = 0.0; c = float((centers[i] + centers[i+1]) / 2.0)
        elif i == K-1:
            a = float((centers[i-1] + centers[i]) / 2.0); c = 100.0
        else:
            a = float((centers[i-1] + centers[i]) / 2.0)
            c = float((centers[i] + centers[i+1]) / 2.0)
        a = max(0.0, min(a, b)); c = min(100.0, max(c, b))
        mapping[lv] = TFN(a,b,c)
    return mapping

# ========== Fuzzy-Hybrid TOPSIS ==========
def _normalize_fuzzy_matrix(matrix: List[List[TFN]], is_benefit: List[bool]) -> List[List[TFN]]:
    m, n = len(matrix), len(matrix[0])
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
                row.append(TFN(amin/x.c, amin/x.b if x.b != 0 else amin/1e-9, amin/x.a if x.a != 0 else amin/1e-9))
        out.append(row)
    return out

def _apply_weights(matrix: List[List[TFN]], weights: List[float]) -> List[List[TFN]]:
    m, n = len(matrix), len(matrix[0])
    s = sum(weights); 
    if s <= 0: raise ValueError("La suma de pesos debe ser > 0.")
    w = [wi/s for wi in weights]
    return [[matrix[i][j].scale(w[j]) for j in range(n)] for i in range(m)]

def _fdist(x: TFN, y: TFN) -> float:
    return math.sqrt((x.a-y.a)**2 + (x.b-y.b)**2 + (x.c-y.c)**2)

def fuzzy_topsis_cc(matrix: List[List[TFN]], is_benefit: List[bool], weights: Optional[List[float]]=None) -> np.ndarray:
    m, n = len(matrix), len(matrix[0])
    if weights is None: weights = [1.0/n]*n
    norm = _normalize_fuzzy_matrix(matrix, is_benefit)
    vw = _apply_weights(norm, weights)
    fpis: List[TFN] = []; fnis: List[TFN] = []
    for j in range(n):
        col = [vw[i][j] for i in range(m)]
        fpis.append(TFN(max(x.a for x in col), max(x.b for x in col), max(x.c for x in col)))
        fnis.append(TFN(min(x.a for x in col), min(x.b for x in col), min(x.c for x in col)))
    d_plus = np.zeros(m); d_minus = np.zeros(m)
    for i in range(m):
        for j in range(n):
            d_plus[i]  += _fdist(vw[i][j], fpis[j])**2
            d_minus[i] += _fdist(vw[i][j], fnis[j])**2
        d_plus[i]  = math.sqrt(d_plus[i])
        d_minus[i] = math.sqrt(d_minus[i])
    cc = d_minus / (d_plus + d_minus + 1e-12)
    return np.clip(cc, 0, 1)

# ========== Fuzzy C-Means simple ==========
def fuzzy_c_means(X: np.ndarray, c: int=3, m: float=2.0, max_iter: int=200, tol: float=1e-5, seed: int=42):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    U = rng.uniform(0,1,(c,n)); U /= U.sum(axis=0, keepdims=True)
    for _ in range(max_iter):
        um = U**m
        centers = (um @ X) / (um.sum(axis=1, keepdims=True)+1e-12)
        dist = np.zeros((c,n))
        for k in range(c):
            d = X - centers[k]; dist[k] = np.sqrt((d**2).sum(axis=1))
        dist = np.fmax(dist,1e-12)
        U_new = 1.0/(dist**2); U_new /= U_new.sum(axis=0, keepdims=True)
        if np.linalg.norm(U_new-U) < tol: U = U_new; break
        U = U_new
    return U, centers

# ========== Etiquetas Apostle ==========
def apostle_quadrants(x: np.ndarray, y: np.ndarray, x_thr: float, y_thr: float,
                      AA: str, AB: str, BA: str, BB: str) -> List[str]:
    out = []
    for xi, yi in zip(x,y):
        if   xi >= x_thr and yi >= y_thr: out.append(AA)
        elif xi >= x_thr and yi <  y_thr: out.append(AB)
        elif xi <  x_thr and yi >= y_thr: out.append(BA)
        else:                              out.append(BB)
    return out

def extended_categories(u_x: np.ndarray, u_y: np.ndarray,
                        names_x=("LowX","MidX","HighX"),
                        names_y=("LowY","MidY","HighY")) -> List[str]:
    ix = u_x.argmax(axis=0); iy = u_y.argmax(axis=0)
    mapx = {0:names_x[0],1:names_x[1],2:names_x[2]}
    mapy = {0:names_y[0],1:names_y[1],2:names_y[2]}
    return [f"{mapx[i]}|{mapy[j]}" for i,j in zip(ix,iy)]

# ========== Helpers de pipeline ==========
def df_to_tfn_matrix(df: pd.DataFrame, cols: List[str], tfn_map: Dict[int, TFN], levels: List[int]) -> List[List[TFN]]:
    lvlset = set(levels)
    for c in cols:
        vals = pd.unique(df[c].dropna())
        if not set(map(int,vals)).issubset(lvlset):
            raise ValueError(f"Columna '{c}' con valores fuera de escala {sorted(lvlset)}.")
    m = df.shape[0]; mat: List[List[TFN]] = []
    for i in range(m):
        row = []
        for c in cols:
            v = df.iloc[i][c]
            if pd.isna(v): raise ValueError(f"NaN en fila {i}, col {c}. Imputa/elimina NA.")
            row.append(tfn_map[int(v)])
        mat.append(row)
    return mat

def run_latent_index(df: pd.DataFrame, items: List[str], tfn_map: Dict[int, TFN], levels: List[int],
                     is_benefit: Optional[List[bool]]=None, weights: Optional[List[float]]=None) -> np.ndarray:
    if is_benefit is None: is_benefit = [True]*len(items)
    matrix = df_to_tfn_matrix(df, items, tfn_map, levels)
    return fuzzy_topsis_cc(matrix, is_benefit=is_benefit, weights=weights)

def run_pipeline(df: pd.DataFrame,
                 latents: Dict[str, List[str]],
                 tfn_map: Dict[int, TFN], levels: List[int],
                 is_benefit_by_latent: Optional[Dict[str, List[bool]]] = None,
                 weights_by_latent: Optional[Dict[str, List[float]]] = None,
                 fcm_c: int=3, fcm_m: float=2.0,
                 thresholds: Tuple[float,float]=(0.5,0.5),
                 quad_labels: Tuple[str,str,str,str]=("Apostles","Mercenaries","Hostages","Defectors"),
                 seed: int=42) -> Dict[str, Any]:

    names = list(latents.keys())
    idx: Dict[str, np.ndarray] = {}
    U_by: Dict[str, np.ndarray] = {}
    C_by: Dict[str, np.ndarray] = {}

    for nm in names:
        items = latents[nm]
        bene = None if (is_benefit_by_latent is None or nm not in is_benefit_by_latent) else is_benefit_by_latent[nm]
        w    = None if (weights_by_latent     is None or nm not in weights_by_latent)     else weights_by_latent[nm]
        z = run_latent_index(df, items, tfn_map, levels, bene, w)
        z = np.clip(z,0,1)
        idx[nm] = z
        U, C = fuzzy_c_means(z.reshape(-1,1), c=fcm_c, m=fcm_m, seed=seed)
        U_by[nm] = U; C_by[nm] = C

    classic = None; ext = None
    if len(names) >= 2:
        xnm, ynm = names[0], names[1]
        AA,AB,BA,BB = quad_labels
        classic = apostle_quadrants(idx[xnm], idx[ynm], thresholds[0], thresholds[1], AA,AB,BA,BB)
        if U_by[xnm].shape[0]==3 and U_by[ynm].shape[0]==3:
            def order(u,c): 
                order = np.argsort(c[:,0]); return u[order]
            ux = order(U_by[xnm], C_by[xnm]); uy = order(U_by[ynm], C_by[ynm])
            ext = extended_categories(ux, uy)
    return {"idx": idx, "U": U_by, "C": C_by, "classic": classic, "extended": ext}

# ========== UI ==========
st.set_page_config(page_title="Fuzzy-Hybrid TOPSIS + ECO-Apostle", layout="wide")
st.title("Variables latentes con Fuzzy-Hybrid TOPSIS + ECO-Apostle")

with st.sidebar:
    st.header("1) Datos")
    up = st.file_uploader("CSV o Excel", type=["csv","xlsx"])
    sep = st.text_input("Separador CSV (si CSV)", value=",")
    sheet = st.text_input("Hoja Excel (si aplica)", value="")
    df = None
    if up:
        try:
            df = pd.read_csv(up, sep=sep) if up.name.lower().endswith(".csv") else pd.read_excel(up, sheet_name=sheet if sheet else 0)
            st.success(f"Cargado: {df.shape[0]} filas × {df.shape[1]} columnas.")
        except Exception as e:
            st.error(f"Error leyendo datos: {e}")

    st.header("2) Escala y TFN")
    sc_choice = st.selectbox("Tipo de escala", ["Likert 1–4 (preset)","Likert 1–5 (preset)","Escala arbitraria (TFN lineales)","Escala arbitraria (TFN manuales)"])
    levels: List[int] = []; tfn_map: Dict[int, TFN] = {}

    if sc_choice.startswith("Likert 1–4"):
        levels = [1,2,3,4]; tfn_map = likert4_default_map()
    elif sc_choice.startswith("Likert 1–5"):
        levels = [1,2,3,4,5]; tfn_map = likert5_default_map()
    else:
        levels_str = st.text_input("Niveles (ej. 1,2,3,...,10)", value="1,2,3,4,5")
        try:
            levels = sorted(list({int(x.strip()) for x in levels_str.split(",") if x.strip()!=''}))
        except:
            levels = []; st.warning("Revisa el formato de niveles.")
        if levels:
            if "lineales" in sc_choice:
                try: tfn_map = linear_tfn_map(levels)
                except Exception as e: st.error(f"No se pudieron generar TFN: {e}")
            else:
                st.caption("Define TFN manuales (a,b,c) para cada nivel:")
                tfn_map = {}
                for lv in levels:
                    abctxt = st.text_input(f"Nivel {lv} (a,b,c)", value="0,0,25" if lv==levels[0] else "", key=f"tfn_{lv}")
                    try:
                        a,b,c = [float(x.strip()) for x in abctxt.split(",")]
                        tfn_map[lv] = TFN(a,b,c)
                    except: pass

    if tfn_map:
        with st.expander("Ver TFN"):
            st.dataframe(pd.DataFrame([{"nivel":lv,"a":t.a,"b":t.b,"c":t.c} for lv,t in sorted(tfn_map.items())]), use_container_width=True)

    st.header("3) Variables latentes (elige ítems del archivo)")
    latents: Dict[str, List[str]] = {}
    bene_by: Dict[str, List[bool]] = {}
    w_by: Dict[str, List[float]] = {}

    if df is not None and len(df.columns)>0 and levels and tfn_map:
        cols = list(df.columns)
        nlat = st.number_input("¿Cuántas variables latentes?", min_value=1, max_value=12, value=2, step=1)
        for i in range(int(nlat)):
            st.subheader(f"Latente #{i+1}")
            lname = st.text_input(f"Nombre latente #{i+1}", value=f"Lat{i+1}", key=f"lname_{i}")
            sel = st.multiselect(f"Ítems/columnas para {lname}", cols, key=f"lcols_{i}")
            latents[lname] = sel
            bc, ww = [], []
            for c in sel:
                c1,c2 = st.columns(2)
                with c1:
                    bcopt = st.selectbox(f"{c}: Beneficio/Coste", ["Beneficio","Coste"], key=f"bc_{i}_{c}")
                    bc.append(bcopt=="Beneficio")
                with c2:
                    w = st.number_input(f"Peso {c}", value=1.0, step=0.1, key=f"w_{i}_{c}")
                    ww.append(float(w))
            bene_by[lname] = bc if sel else []
            w_by[lname] = ww if sel else []

    st.header("4) Clustering y umbrales")
    fcm_c = st.number_input("Clusters FCM por latente (c)", min_value=2, max_value=6, value=3, step=1)
    fcm_m = st.number_input("Parámetro de difusidad m", min_value=1.1, max_value=5.0, value=2.0, step=0.1)
    x_thr = st.number_input("Umbral X (1ª latente)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    y_thr = st.number_input("Umbral Y (2ª latente)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    st.header("5) Nombres manuales de cuadrantes (Apostle)")
    AA = st.text_input("x≥UmbralX & y≥UmbralY", value="Apostles")
    AB = st.text_input("x≥UmbralX & y<UmbralY",  value="Mercenaries")
    BA = st.text_input("x<UmbralX & y≥UmbralY",  value="Hostages")
    BB = st.text_input("x<UmbralX & y<UmbralY",  value="Defectors")

run_btn = st.button("▶ Ejecutar análisis")

# ========== Ejecución ==========
if run_btn:
    try:
        if df is None: st.error("Carga primero un archivo."); st.stop()
        if not levels or not tfn_map

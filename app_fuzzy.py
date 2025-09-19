# app_fuzzy.py (versión corregida)
import math
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
import streamlit as st
import skfuzzy as fuzz

# ==============================
# Definición de TFN y escalas
# ==============================
@dataclass(frozen=True)
class TFN:
    a: float; b: float; c: float
    def __post_init__(self):
        if not (self.a <= self.b <= self.c):
            raise ValueError("TFN requiere a ≤ b ≤ c.")
    def scale(self, w: float) -> "TFN":
        return TFN(self.a*w, self.b*w, self.c*w)

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

# ==============================
# Normalización TOPSIS individual
# ==============================
def ensure_tfn(x):
    if isinstance(x, TFN): return x
    if isinstance(x, (int, float)): return TFN(x, x, x)
    if isinstance(x, (list, tuple)) and len(x)==3: return TFN(*x)
    raise ValueError(f"No se puede convertir {x} a TFN")

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

# ==============================
# Classic Apostle thresholds
# ==============================
st.set_page_config(page_title="Fuzzy TOPSIS", layout="wide")
st.title("Demo Classic Apostle con thresholds")

thr_x=st.sidebar.slider("Threshold X", 0.0, 1.0, 0.5, 0.01)
thr_y=st.sidebar.slider("Threshold Y", 0.0, 1.0, 0.5, 0.01)

lab_a = st.sidebar.text_input("(X≥thr & Y≥thr)", "Quadrant A")
lab_b = st.sidebar.text_input("(X<thr & Y≥thr)", "Quadrant B")
lab_c = st.sidebar.text_input("(X<thr & Y<thr)", "Quadrant C")
lab_d = st.sidebar.text_input("(X≥thr & Y<thr)", "Quadrant D")

st.write("Configuración cargada correctamente ✅")

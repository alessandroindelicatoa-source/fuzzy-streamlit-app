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
        st.error("No passwords configured. Please add them in .streamlit/secrets.toml")
        st.stop()

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
        st.error("😕 Incorrect username or password")
        st.stop()

check_password()

# ==============================
# 1) TFN
# ==============================
@dataclass(frozen=True)
class TFN:
    a: float; b: float; c: float
    def __post_init__(self): assert self.a <= self.b <= self.c
    def scale(self, w: float): return TFN(self.a*w, self.b*w, self.c*w)

# Predefined TFN scales
def tfn_map_likert(scale:int):
    if scale==4: return {1:TFN(0,0,20),2:TFN(16,33,60),3:TFN(49,66,83),4:TFN(80,100,100)}
    if scale==5: return {1:TFN(0,0,30),2:TFN(20,30,40),3:TFN(30,50,70),4:TFN(60,70,80),5:TFN(70,100,100)}
    if scale==6: return {1:TFN(0,0,15),2:TFN(25,40,55),3:TFN(45,60,75),4:TFN(70,80,90),5:TFN(85,100,100),6:TFN(90,100,100)}
    if scale==11: return {i:TFN(max(0,(i-1)*10),i*10,(i+1)*10 if i<11 else 100) for i in range(1,12)}
    return {}

# ==============================
# Resto del código simplificado para demo
# ==============================
st.title("Fuzzy-Hybrid TOPSIS + Apostle Classic & ECO-Extended (Login Enabled)")
st.info("App running with password protection.")


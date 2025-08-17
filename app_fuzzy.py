import streamlit as st
import pandas as pd
import numpy as np

st.title("Latent Variables via Fuzzy-Hybrid TOPSIS + ECO-Extended Apostle")

# Sidebar uploader
st.sidebar.header("1) Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
sep = st.sidebar.text_input("CSV separator (default=,)", value=",")

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file, sep=sep)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Central uploader (for mobile convenience)
central_up = st.file_uploader("Upload CSV or Excel here (optional)", type=["csv","xlsx"], key="central_up")
if central_up is not None:
    try:
        if central_up.name.lower().endswith(".csv"):
            df = pd.read_csv(central_up)
        else:
            df = pd.read_excel(central_up, sheet_name=0)
        st.success(f"Loaded (central): {df.shape[0]} rows × {df.shape[1]} columns.")
    except Exception as e:
        st.error(f"Read error (central): {e}")

if df is not None:
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    st.sidebar.header("2) Latent Variables")
    items = list(df.columns)
    num_latent = st.sidebar.number_input("Number of latent variables", 1, 5, 2)
    latent_defs = {}
    for i in range(int(num_latent)):
        st.sidebar.subheader(f"Latent {i+1}")
        sel = st.sidebar.multiselect(f"Select items for Latent {i+1}", items, key=f"sel_{i}")
        if sel:
            weights = [st.sidebar.number_input(f"Weight for {c}", 0.0, 10.0, 1.0, key=f"w_{i}_{c}") for c in sel]
            latent_defs[f"Latent_{i+1}"] = (sel, weights)

    if st.button("Run analysis"):
        latent_scores = pd.DataFrame()
        for k,(sel,weights) in latent_defs.items():
            sub = df[sel].astype(float)
            w = np.array(weights)/np.sum(weights)
            latent_scores[k] = np.dot(sub, w)
        st.write("Latent variable scores:")
        st.dataframe(latent_scores.head())

        # Simple Apostle quadrant split
        st.header("ECO-Extended Apostle Classification")
        x_latent = st.selectbox("Choose latent for X axis", latent_scores.columns)
        y_latent = st.selectbox("Choose latent for Y axis", latent_scores.columns)
        thr_x = st.slider("Threshold X", float(latent_scores[x_latent].min()), float(latent_scores[x_latent].max()), float(latent_scores[x_latent].mean()))
        thr_y = st.slider("Threshold Y", float(latent_scores[y_latent].min()), float(latent_scores[y_latent].max()), float(latent_scores[y_latent].mean()))

        q_names = {}
        for q in ["Q1 (high-high)","Q2 (low-high)","Q3 (low-low)","Q4 (high-low)"]:
            q_names[q] = st.text_input(f"Name for {q}", value=q)

        def classify(row):
            if row[x_latent]>=thr_x and row[y_latent]>=thr_y: return q_names["Q1 (high-high)"]
            if row[x_latent]<thr_x and row[y_latent]>=thr_y: return q_names["Q2 (low-high)"]
            if row[x_latent]<thr_x and row[y_latent]<thr_y: return q_names["Q3 (low-low)"]
            return q_names["Q4 (high-low)"]
        latent_scores["Quadrant"] = latent_scores.apply(classify, axis=1)
        st.write("Classification results:")
        st.dataframe(latent_scores)
else:
    st.info("Please upload a data file first.")

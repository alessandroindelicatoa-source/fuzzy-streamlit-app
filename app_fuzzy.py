
import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class TFN:
    a: float; b: float; c: float
    def scale(self,w:float): return TFN(self.a*w,self.b*w,self.c*w)

def defuzz_buckley(x: TFN) -> float:
    return (x.a + 2*x.b + x.c)/4.0

# ==============================
# Streamlit App
# ==============================
st.title("Scaffold")
if 'results' in st.session_state:
    st.subheader("📊 Resultados individuales (TOPSIS)")


# --- NEW: Global PIS/NIS across ALL items (X ∪ Y) and ALL selected grouping columns ---
def global_pis_nis_across_items(df: pd.DataFrame, items_all: List[str],
                                tfn_maps: Dict[str,Dict[int,TFN]],
                                levels_by_item: Dict[str,List[int]],
                                group_cols: List[str]) -> pd.DataFrame:
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


# --- NEW: Unified Group TOPSIS table (one table, NOT split) ---
def unified_group_topsis_table(df: pd.DataFrame,
                               items_x: List[str], items_y: List[str],
                               tfn_maps: Dict[str,Dict[int,TFN]],
                               levels_by_item: Dict[str,List[int]],
                               group_cols: List[str],
                               name_x: str, name_y: str) -> pd.DataFrame:
    def _defuzz_means(dfg, items):
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

    out_rows = []
    if not group_cols:
        tx = 0.5; ty = 0.5
        out_rows.append({"Variable":"ALL","Item":"ALL","Topsis-LatX":tx,"Topsis-LatY":ty})
        return pd.DataFrame(out_rows, columns=["Variable","Item","Topsis-LatX","Topsis-LatY"])

    for gcol in group_cols:
        groups = [g for g,_ in df.groupby(gcol, dropna=True)]
        Vx = pd.DataFrame([_defuzz_means(df[df[gcol]==g], items_x) for g in groups]) if items_x else pd.DataFrame(index=range(len(groups)))
        Vy = pd.DataFrame([_defuzz_means(df[df[gcol]==g], items_y) for g in groups]) if items_y else pd.DataFrame(index=range(len(groups)))

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


# --- UI hooks example (place inside your 'if results' block) ---
# items_all = items_x + items_y
# pis_nis_global = global_pis_nis_across_items(df, items_all, tfn_map_by_item, levels_by_item, group_cols)
# st.subheader("🌐 Global PIS/NIS (single table across ALL items)")
# st.dataframe(pis_nis_global)
# st.download_button("⬇️ Download CSV — Global PIS/NIS", data=pis_nis_global.to_csv(index=False).encode("utf-8"),
#                    file_name="global_pis_nis_all_items.csv")
#
# unified_groups = unified_group_topsis_table(df, items_x, items_y, tfn_map_by_item, levels_by_item, group_cols, lname_x, lname_y)
# st.subheader("🧮 Group TOPSIS (single table)")
# st.caption("Columns: Variable (grouping variable), Item (group value), Topsis-LatX, Topsis-LatY")
# st.dataframe(unified_groups)
# st.download_button("⬇️ Download CSV — Unified Group TOPSIS", data=unified_groups.to_csv(index=False).encode("utf-8"),
#                    file_name="group_topsis_unified.csv")

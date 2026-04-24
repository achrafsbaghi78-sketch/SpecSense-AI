import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="SpecSense AI", 
    layout="wide", 
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# ===== CONFIG =====
SHEET_ID = "1Xy4tgkGs1OXOTh-OMAsR7YsfkUPxttF7qalhDdhHa90"
URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

st.title("🎯 SpecSense AI - Quality 4.0 Suite")
st.caption("MSA Gage R&R + SPC Live + Cpk + FMEA + Pareto | IATF 16949:2016 Compliant")

# ===== LOAD DATA =====
@st.cache_data(ttl=30)
def load_data():
    try:
        df = pd.read_csv(URL)
        df['Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce')
        df['Measurement'] = pd.to_numeric(df['Measurement'], errors='coerce')
        df['Trial'] = pd.to_numeric(df['Trial'], errors='coerce')
        df['USL'] = pd.to_numeric(df['USL'], errors='coerce')
        df['LSL'] = pd.to_numeric(df['LSL'], errors='coerce')
        df['Severity'] = pd.to_numeric(df['Severity'], errors='coerce')
        df['Occurrence'] = pd.to_numeric(df['Occurrence'], errors='coerce')
        df['Detection'] = pd.to_numeric(df['Detection'], errors='coerce')
        df = df.dropna(subset=['Measurement'])
        return df
    except Exception as e:
        st.error(f"❌ Error loading Google Sheet: {e}")
        st.info("Vérifier: 1. Sheet = 'Anyone with link - Viewer'  2. Colonnes correctes")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# ===== SEPARATE MSA vs SPC =====
df_msa = df[df['Trial'].notna()].copy()
df_spc = df[df['Trial'].isna()].copy()

# ===== SIDEBAR =====
st.sidebar.metric("📊 Total Mesures", len(df))
st.sidebar.metric("📏 MSA Points", len(df_msa))
st.sidebar.metric("🏭 SPC Points", len(df_spc))
st.sidebar.divider()
st.sidebar.success("SpecSense AI v1.0 Live")
st.sidebar.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ===== TABS =====
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📏 MSA Gage R&R", 
    "📊 SPC X̄-R", 
    "📈 Capability Cpk", 
    "📋 Pareto Defects", 
    "🎯 FMEA RPN"
])

# ===== TAB 1: MSA GAGE R&R =====
with tab1:
    st.header("MSA Gage R&R ANOVA - IATF §7.1.5.1")
    
    n_parts = df_msa['Part_ID'].nunique()
    n_op = df_msa['Operator'].nunique()
    n_trials = df_msa['Trial'].nunique()
    
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Pièces", n_parts)
    col_info2.metric("Opérateurs", n_op)
    col_info3.metric("Essais", n_trials)
    
    if len(df_msa) < 30 or n_parts < 5 or n_op < 2 or n_trials < 2:
        st.warning(f"⚠️ Data MSA na9sa: {len(df_msa)} lignes. Standard: 10 Pièces × 3 Opérateurs × 3 Essais = 90")
        st.info("Kml data MSA bach ykhdem %GRR")
    else:
        # Constants AIAG MSA 4th
        K1_dict = {2: 0.8862, 3: 0.5908}
        K2_dict = {2: 0.7071, 3: 0.5231}
        K3_dict = {5: 0.4030, 10: 0.3146}
        
        K1 = K1_dict.get(n_trials, 0.5908)
        K2 = K2_dict.get(n_op, 0.5231)
        K3 = K3_dict.get(n_parts, 0.3146)
        
        # EV Répétabilité
        ranges = df_msa.groupby(['Part_ID', 'Operator'])['Measurement'].agg(['max', 'min'])
        ranges['R'] = ranges['max'] - ranges['min']
        R_bar = ranges['R'].mean()
        EV = K1 * R_bar
        
        # AV Reproductibilité
        X_bar_op = df_msa.groupby('Operator')['Measurement'].mean()
        X_DIFF = X_bar_op.max() - X_bar_op.min()
        AV = np.sqrt(max(0, (K2 * X_DIFF)**2 - EV**2/(n_parts*n_trials)))
        
        # PV Variation Pièce
        X_bar_part = df_msa.groupby('Part_ID')['Measurement'].mean()
        Rp = X_bar_part.max() - X_bar_part.min()
        PV = Rp * K3
        
        # GRR & %GRR
        GRR = np.sqrt(EV**2 + AV**2)
        TV = np.sqrt(GRR**2 + PV**2)
        per_GRR = (GRR / TV) * 100 if TV > 0 else 0
        per_EV = (EV / TV) * 100

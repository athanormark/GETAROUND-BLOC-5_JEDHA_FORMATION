import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="GetAround - Delay Analysis", page_icon="🚗", layout="wide")

st.title("🚗 GetAround - Analyse des retards")
st.markdown("Dashboard d'aide a la decision : quel seuil minimum entre deux locations ?")

# --- Chargement ---
DATA_URL = "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx"

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_URL)
    return df

df = load_data()
total = len(df)
ended = df[df["state"] == "ended"]
canceled = df[df["state"] == "canceled"]

# --- Sidebar ---
st.sidebar.header("Parametres de simulation")
threshold = st.sidebar.slider("Seuil minimum (minutes)", 0, 720, 120, step=15)
scope = st.sidebar.radio("Scope", ["Toutes les voitures", "Connect uniquement"])

# --- KPIs ---
st.header("Vue d'ensemble")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total locations", f"{total:,}")
col2.metric("Terminees", f"{len(ended):,}", f"{len(ended)/total*100:.1f}%")
col3.metric("Annulees", f"{len(canceled):,}", f"{len(canceled)/total*100:.1f}%")

delays = ended["delay_at_checkout_in_minutes"].dropna()
late = delays[delays > 0]
col4.metric("En retard", f"{len(late):,}", f"{len(late)/len(delays)*100:.1f}%")

# --- Distributions ---
st.header("Distribution des retards")

col_left, col_right = st.columns(2)

with col_left:
    fig = px.pie(df, names="checkin_type", title="Type de checkin")
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    fig = px.pie(df, names="state", title="Statut des locations")
    st.plotly_chart(fig, use_container_width=True)

fig = px.histogram(
    delays.clip(-200, 500), nbins=100,
    title="Distribution des retards au checkout (clippe -200 a 500 min)",
    labels={"value": "Retard (minutes)", "count": "Nombre"}
)
fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Heure prevue")
st.plotly_chart(fig, use_container_width=True)

fig = px.box(
    ended, x="checkin_type", y="delay_at_checkout_in_minutes",
    title="Retards par type de checkin", range_y=[-200, 500]
)
st.plotly_chart(fig, use_container_width=True)

# --- Analyse impact ---
st.header("Impact sur les locations suivantes")

prev_delays_map = df.set_index("rental_id")["delay_at_checkout_in_minutes"]
with_prev = df[df["previous_ended_rental_id"].notna()].copy()
with_prev["prev_delay"] = with_prev["previous_ended_rental_id"].map(prev_delays_map)
problematic = with_prev[with_prev["prev_delay"] > with_prev["time_delta_with_previous_rental_in_minutes"]]

col1, col2, col3 = st.columns(3)
col1.metric("Locations consecutives", f"{len(with_prev):,}")
col2.metric("Cas problematiques", f"{len(problematic):,}", f"{len(problematic)/len(with_prev)*100:.1f}%")
prob_canceled = problematic[problematic["state"] == "canceled"]
col3.metric("Annulations liees", f"{len(prob_canceled):,}")

fig = px.histogram(
    with_prev["time_delta_with_previous_rental_in_minutes"], nbins=50,
    title="Delta entre locations consecutives",
    labels={"value": "Delta (minutes)", "count": "Nombre"}
)
st.plotly_chart(fig, use_container_width=True)

# --- Simulation ---
st.header(f"Simulation : seuil = {threshold} min | scope = {scope.lower()}")

if scope == "Connect uniquement":
    subset = with_prev[with_prev["checkin_type"] == "connect"]
    prob_sub = problematic[problematic["checkin_type"] == "connect"]
else:
    subset = with_prev
    prob_sub = problematic

if threshold == 0:
    affected = 0
    solved = 0
else:
    affected = (subset["time_delta_with_previous_rental_in_minutes"] < threshold).sum()
    solved = (prob_sub["time_delta_with_previous_rental_in_minutes"] + threshold >= prob_sub["prev_delay"]).sum()

col1, col2, col3 = st.columns(3)
col1.metric("Locations bloquees", f"{affected}", f"{affected/len(subset)*100:.1f}%" if len(subset) > 0 else "0%")
col2.metric("Cas resolus", f"{solved}", f"{solved/len(prob_sub)*100:.1f}%" if len(prob_sub) > 0 else "0%")
pct_revenue = affected / total * 100
col3.metric("Impact revenu (locations)", f"{pct_revenue:.1f}%")

# --- Courbe tradeoff ---
st.header("Tradeoff selon le seuil")

results = []
for t in range(0, 721, 15):
    for s in ["all", "connect"]:
        sub = with_prev if s == "all" else with_prev[with_prev["checkin_type"] == "connect"]
        ps = problematic if s == "all" else problematic[problematic["checkin_type"] == "connect"]
        if t == 0:
            aff, sol = 0, 0
        else:
            aff = (sub["time_delta_with_previous_rental_in_minutes"] < t).sum()
            sol = (ps["time_delta_with_previous_rental_in_minutes"] + t >= ps["prev_delay"]).sum()
        results.append({
            "threshold": t, "scope": s,
            "pct_bloquees": round(aff / len(sub) * 100, 1) if len(sub) > 0 else 0,
            "pct_resolus": round(sol / len(ps) * 100, 1) if len(ps) > 0 else 0
        })

df_res = pd.DataFrame(results)

fig = make_subplots(specs=[[{"secondary_y": True}]])
for s, color in [("all", "blue"), ("connect", "green")]:
    sub = df_res[df_res["scope"] == s]
    fig.add_trace(go.Scatter(x=sub["threshold"], y=sub["pct_resolus"],
                             name=f"Cas resolus ({s})", mode="lines",
                             line=dict(color=color)), secondary_y=False)
    fig.add_trace(go.Scatter(x=sub["threshold"], y=sub["pct_bloquees"],
                             name=f"Bloquees ({s})", mode="lines",
                             line=dict(color=color, dash="dash")), secondary_y=True)

fig.add_vline(x=threshold, line_dash="dot", line_color="red", annotation_text=f"Seuil: {threshold}min")
fig.update_layout(title="Impact du seuil sur les cas resolus et les locations bloquees")
fig.update_xaxes(title_text="Seuil (minutes)")
fig.update_yaxes(title_text="% cas resolus", secondary_y=False)
fig.update_yaxes(title_text="% locations bloquees", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)

# --- Tableau recapitulatif ---
st.header("Tableau recapitulatif")
summary = df_res[df_res["threshold"].isin([30, 60, 120, 180, 240, 360, 720])]
st.dataframe(summary.style.format({"pct_bloquees": "{:.1f}%", "pct_resolus": "{:.1f}%"}), use_container_width=True)

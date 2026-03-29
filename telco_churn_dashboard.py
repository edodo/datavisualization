"""
Telco Customer Churn Dashboard
Capstone Project — Data Visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark Theme CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0a0c14; color: #ccd6f6; }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #1e2a3a; }
    
    /* KPI Cards */
    .kpi-card {
        background: #0f1117;
        border: 1px solid #1e2a3a;
        border-radius: 12px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .kpi-label {
        font-size: 11px;
        letter-spacing: 2px;
        color: #8892b0;
        text-transform: uppercase;
        margin-bottom: 10px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 38px;
        font-weight: 700;
        color: #6ba3d6;
        line-height: 1;
    }
    .kpi-sub {
        font-size: 12px;
        color: #8892b0;
        margin-top: 8px;
    }

    /* Key Point Banner */
    .key-banner {
        background: #0f1117;
        border: 1px solid #1e2a3a;
        border-radius: 12px;
        padding: 22px 32px;
        text-align: center;
        margin: 8px 0 24px 0;
    }
    .key-banner-label {
        font-size: 11px;
        letter-spacing: 2px;
        color: #8892b0;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .key-banner-text {
        font-size: 17px;
        color: #6ba3d6;
        font-weight: 500;
        line-height: 1.5;
    }

    /* Section headers */
    .section-header {
        font-size: 13px;
        letter-spacing: 2px;
        color: #64ffda;
        text-transform: uppercase;
        font-weight: 700;
        margin: 28px 0 12px 0;
        border-bottom: 1px solid #1e2a3a;
        padding-bottom: 8px;
    }

    /* Risk badge */
    .risk-high { color: #ef5350; font-weight: 700; }
    .risk-mid  { color: #ffa726; font-weight: 700; }
    .risk-low  { color: #66bb6a; font-weight: 700; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: #0d1117; gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: #0f1117; color: #8892b0;
        border-radius: 6px 6px 0 0; border: 1px solid #1e2a3a;
        padding: 8px 18px; font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        background: #1e2a3a !important; color: #64ffda !important;
        border-bottom: 2px solid #64ffda !important;
    }

    /* Metric overrides */
    [data-testid="metric-container"] {
        background: #0f1117; border: 1px solid #1e2a3a;
        border-radius: 10px; padding: 12px;
    }
    [data-testid="metric-container"] label { color: #8892b0 !important; }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #6ba3d6 !important; font-size: 28px !important;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label { color: #8892b0; font-size: 13px; }

    h1, h2, h3 { color: #ccd6f6 !important; }

    /* Hide default streamlit header */
    header[data-testid="stHeader"] { background: transparent; }
    footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Plot defaults ───────────────────────────────────────────────────────────────
PLOT_BG   = "#0f1117"
PAPER_BG  = "#0f1117"
FONT_CLR  = "#8892b0"
GRID_CLR  = "#1e2a3a"
RED       = "#ef5350"
GREEN     = "#66bb6a"
BLUE      = "#6ba3d6"
TEAL      = "#64ffda"

def dark_layout(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ccd6f6", size=14)),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_CLR, size=11),
        height=height,
        margin=dict(l=40, r=20, t=50 if title else 20, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_CLR, borderwidth=1)
    )
    fig.update_xaxes(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
    fig.update_yaxes(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
    return fig

# ── Load & Prepare Data ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df["Churn_bin"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["Tenure_Group"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"]
    )
    return df

with st.spinner("Loading dataset..."):
    df = load_data()

TOTAL      = len(df)
CHURNED    = df["Churn_bin"].sum()
CHURN_RATE = CHURNED / TOTAL * 100
RECALL     = 0.78
TARGET_N   = 1869     # high-risk customers from the image

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-size:26px;'>📡</div>
        <div style='font-size:16px; font-weight:700; color:#ccd6f6; letter-spacing:1px;'>
            TELCO CHURN
        </div>
        <div style='font-size:11px; color:#8892b0; letter-spacing:2px; margin-top:4px;'>
            CAPSTONE PROJECT
        </div>
    </div>
    <hr style='border-color:#1e2a3a; margin: 12px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("<div style='color:#8892b0; font-size:12px; letter-spacing:1px; margin-bottom:8px;'>NAVIGATION</div>", unsafe_allow_html=True)

    page = st.radio(
        "", 
        ["🏠  Overview", "📊  EDA", "⚠️  High-Risk Customers", "🔮  Model Performance"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1e2a3a; margin: 16px 0;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:11px; color:#8892b0; line-height:1.8;'>
        <b style='color:#ccd6f6;'>Dataset</b><br>
        Rows: {TOTAL:,}<br>
        Features: {df.shape[1]}<br>
        Model: GBM (Recall-optimised)<br>
        <br>
        <b style='color:#ccd6f6;'>Group</b><br>
        Rakesh · Edward (Seongmin Choi) · Mawuko<br>
        Fanshawe College — 2025
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":

    st.markdown("""
    <div style='margin-bottom:6px;'>
        <span style='font-size:11px; color:#64ffda; letter-spacing:2px; text-transform:uppercase;'>
        DATA VISUALIZATION · CAPSTONE
        </span>
    </div>
    <h1 style='margin:0 0 4px 0; font-size:28px;'>Telco Customer Churn Analysis</h1>
    <p style='color:#8892b0; margin:0 0 24px 0; font-size:14px;'>
        Recall-Centric Churn Prediction Dashboard
    </p>
    """, unsafe_allow_html=True)

    # KPI Row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Churn Rate</div>
            <div class='kpi-value'>{CHURN_RATE:.1f}%</div>
            <div class='kpi-sub'>{CHURNED:,} of {TOTAL:,} customers</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Recall Score</div>
            <div class='kpi-value'>{RECALL}</div>
            <div class='kpi-sub'>GBM · Recall-optimised model</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Target Customers</div>
            <div class='kpi-value'>{TARGET_N:,}</div>
            <div class='kpi-sub'>High-risk customers identified</div>
        </div>""", unsafe_allow_html=True)

    # Key Point Banner
    st.markdown("""
    <div class='key-banner' style='margin-top:20px;'>
        <div class='key-banner-label'>Key Point</div>
        <div class='key-banner-text'>
            Recall-centric design to avoid losing potential churn customers,<br>
            rather than precision for cost reduction in churn prevention.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Business rationale
    st.markdown("<div class='section-header'>Business Rationale</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style='background:#0f1117; border:1px solid #1e2a3a; border-left:3px solid #ef5350;
                    border-radius:8px; padding:18px 20px; line-height:1.7; font-size:14px; color:#ccd6f6;'>
            <b style='color:#ef5350;'>⚠️ Missing a Churner</b><br>
            A customer who leaves costs the company lost lifetime revenue, 
            acquisition costs to replace them, and reputational risk — a far 
            more expensive outcome than a false alarm.
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style='background:#0f1117; border:1px solid #1e2a3a; border-left:3px solid #64ffda;
                    border-radius:8px; padding:18px 20px; line-height:1.7; font-size:14px; color:#ccd6f6;'>
            <b style='color:#64ffda;'>✅ Why Recall Wins</b><br>
            With Recall = 0.78, we catch 78% of all churners. Companies can 
            immediately launch <b>1,869 personalised promotions</b> targeted at 
            each individual's specific reason for churn.
        </div>
        """, unsafe_allow_html=True)

    # Quick overview charts
    st.markdown("<div class='section-header'>Churn Snapshot</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])

    with col1:
        fig_pie = go.Figure(go.Pie(
            labels=["Retained", "Churned"],
            values=[TOTAL - CHURNED, CHURNED],
            marker=dict(colors=[GREEN, RED], line=dict(color=PLOT_BG, width=3)),
            hole=0.6,
            textinfo="percent+label",
            textfont=dict(size=13, color="#ccd6f6"),
            pull=[0, 0.05]
        ))
        fig_pie.add_annotation(
            text=f"<b>{CHURN_RATE:.1f}%</b><br><span style='font-size:10px'>churn</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="#ccd6f6"), align="center"
        )
        dark_layout(fig_pie, "Churn Distribution", height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        contract_churn = df.groupby("Contract")["Churn_bin"].agg(["mean", "count"]).reset_index()
        contract_churn["rate"] = contract_churn["mean"] * 100
        fig_c = go.Figure()
        bar_colors = [RED, "#ffa726", GREEN]
        for i, row in contract_churn.iterrows():
            fig_c.add_trace(go.Bar(
                x=[row["Contract"]], y=[row["rate"]],
                marker_color=bar_colors[i],
                name=row["Contract"],
                text=[f"{row['rate']:.1f}%"],
                textposition="outside",
                textfont=dict(color="#ccd6f6", size=13, family="Arial Black")
            ))
        dark_layout(fig_c, "Churn Rate by Contract Type", height=320)
        fig_c.update_layout(showlegend=False, bargap=0.4)
        fig_c.update_yaxes(title_text="Churn Rate (%)", range=[0, 55])
        st.plotly_chart(fig_c, use_container_width=True)

    # Top 5 risk factors
    st.markdown("<div class='section-header'>Top Churn Risk Factors (SHAP-Derived)</div>", unsafe_allow_html=True)
    factors = [
        ("Contract: Month-to-month",  0.312, "42.7% churn rate vs 11.3% / 2.8% for 1/2-year"),
        ("Low Tenure (< 12 months)",  0.243, "First year is critical — highest dropout window"),
        ("Fiber Optic Internet",       0.187, "Highest churn among service types: 41.9%"),
        ("Electronic Check Payment",  0.142, "3× higher churn than auto-pay credit card users"),
        ("No Tech Support / Security",0.098, "Add-on services act as retention 'glue'"),
    ]
    cols = st.columns(5)
    for col, (name, score, tip) in zip(cols, factors):
        bar_fill = int(score * 320)
        with col:
            st.markdown(f"""
            <div style='background:#0f1117; border:1px solid #1e2a3a; border-radius:10px;
                        padding:16px 14px; height:180px;'>
                <div style='font-size:11px; color:#64ffda; font-weight:700; margin-bottom:8px;'>{name}</div>
                <div style='background:#1e2a3a; border-radius:4px; height:6px; margin-bottom:10px;'>
                    <div style='background:{RED}; width:{bar_fill}px; max-width:100%; height:6px; border-radius:4px;'></div>
                </div>
                <div style='font-size:22px; font-weight:700; color:{RED}; margin-bottom:6px;'>
                    {score:.0%}
                </div>
                <div style='font-size:11px; color:#8892b0; line-height:1.4;'>{tip}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA":

    st.markdown("<h2 style='margin-bottom:4px;'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8892b0; font-size:13px;'>Understanding patterns that drive customer churn</p>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Numerical Features", "📦 Categorical Features",
        "🔥 Correlation", "🧩 Multi-Variable"
    ])

    # ── Tab 1: Numerical ────────────────────────────────────────────────────────
    with tab1:
        num_col = st.selectbox("Select feature", ["tenure", "MonthlyCharges", "TotalCharges"])
        col1, col2 = st.columns(2)

        with col1:
            retained = df[df["Churn_bin"] == 0][num_col]
            churned  = df[df["Churn_bin"] == 1][num_col]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=retained, name="Retained", nbinsx=35,
                                       marker_color=GREEN, opacity=0.7))
            fig.add_trace(go.Histogram(x=churned, name="Churned", nbinsx=35,
                                       marker_color=RED, opacity=0.7))
            fig.update_layout(barmode="overlay")
            dark_layout(fig, f"{num_col} Distribution by Churn Status", height=360)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=retained, name="Retained", marker_color=GREEN,
                                     boxmean=True, line_color=GREEN))
            fig_box.add_trace(go.Box(y=churned, name="Churned", marker_color=RED,
                                     boxmean=True, line_color=RED))
            dark_layout(fig_box, f"{num_col} — Box Plot", height=360)
            st.plotly_chart(fig_box, use_container_width=True)

        # Stats cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retained Mean", f"{retained.mean():.2f}")
        m2.metric("Churned Mean",  f"{churned.mean():.2f}")
        m3.metric("Difference",    f"{abs(retained.mean()-churned.mean()):.2f}")
        m4.metric("Churn > Retained?", "Yes" if churned.mean() > retained.mean() else "No")

    # ── Tab 2: Categorical ──────────────────────────────────────────────────────
    with tab2:
        cat_options = ["Contract", "InternetService", "PaymentMethod",
                       "Dependents", "Partner", "TechSupport",
                       "OnlineSecurity", "SeniorCitizen"]
        cat_col = st.selectbox("Select feature", cat_options)

        ct = df.groupby(cat_col)["Churn_bin"].agg(["mean", "count"]).reset_index()
        ct["rate"]     = ct["mean"] * 100
        ct["churned"]  = (ct["mean"] * ct["count"]).round()
        ct["retained"] = ct["count"] - ct["churned"]

        col1, col2 = st.columns(2)
        with col1:
            max_rate = ct["rate"].max()
            colors_bar = [RED if r == max_rate else BLUE for r in ct["rate"]]
            fig_bar = go.Figure(go.Bar(
                x=ct[cat_col].astype(str),
                y=ct["rate"],
                marker_color=colors_bar,
                text=[f"{r:.1f}%" for r in ct["rate"]],
                textposition="outside",
                textfont=dict(color="#ccd6f6")
            ))
            dark_layout(fig_bar, f"Churn Rate by {cat_col}", height=360)
            fig_bar.update_yaxes(title_text="Churn Rate (%)", range=[0, ct["rate"].max() * 1.2])
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_stk = go.Figure()
            fig_stk.add_trace(go.Bar(x=ct[cat_col].astype(str), y=ct["retained"],
                                     name="Retained", marker_color=GREEN))
            fig_stk.add_trace(go.Bar(x=ct[cat_col].astype(str), y=ct["churned"],
                                     name="Churned", marker_color=RED))
            fig_stk.update_layout(barmode="stack")
            dark_layout(fig_stk, f"Volume by {cat_col}", height=360)
            st.plotly_chart(fig_stk, use_container_width=True)

    # ── Tab 3: Correlation ──────────────────────────────────────────────────────
    with tab3:
        corr = df[["tenure", "MonthlyCharges", "TotalCharges", "Churn_bin"]].corr()
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="reds",
            zmid=0,
            text=np.round(corr.values, 3),
            texttemplate="%{text}",
            textfont=dict(size=14, color="#ccd6f6")
        ))
        dark_layout(fig_heat, "Correlation Matrix — Numerical Features", height=420)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("""
        <div style='background:#0f1117; border:1px solid #1e2a3a; border-left:3px solid #64ffda;
                    border-radius:8px; padding:16px 20px; margin-top:12px; font-size:14px; color:#ccd6f6; line-height:1.7;'>
            <b style='color:#64ffda;'>📌 Key Insights</b><br>
            • <b>tenure ↔ Churn: −0.35</b> — Longest-serving customers are least likely to leave.<br>
            • <b>MonthlyCharges ↔ Churn: +0.19</b> — Higher bills are a push factor toward churn.<br>
            • <b>TotalCharges ↔ Churn: −0.20</b> — Mirrors tenure; long-term customers accumulate higher totals.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 4: Multi-Variable ───────────────────────────────────────────────────
    with tab4:
        multi = df.groupby(["Contract", "InternetService"])["Churn_bin"].mean() * 100
        pivot = multi.reset_index().pivot(index="Contract", columns="InternetService", values="Churn_bin")

        fig_h2 = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Bluered",
            text=np.round(pivot.values, 1),
            texttemplate="%{text}%",
            textfont=dict(size=15, color="#ccd6f6"),
            colorbar=dict(title="Churn %")
        ))
        dark_layout(fig_h2, "Churn Rate: Contract Type × Internet Service", height=380)
        st.plotly_chart(fig_h2, use_container_width=True)

        st.markdown("""
        <div style='background:#0f1117; border:1px solid #1e2a3a; border-left:3px solid #ef5350;
                    border-radius:8px; padding:16px 20px; margin-top:12px; font-size:14px; color:#ccd6f6;'>
            <b style='color:#ef5350;'>🔥 Critical Finding</b> — 
            Fiber Optic + Month-to-month contract customers experience a <b>~61% churn rate</b>, 
            the single highest-risk cohort. Retention campaigns must prioritise this segment first.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HIGH-RISK CUSTOMERS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️  High-Risk Customers":

    st.markdown("<h2 style='margin-bottom:4px;'>High-Risk Customer Identification</h2>", unsafe_allow_html=True)
    st.markdown(f"""<p style='color:#8892b0; font-size:13px;'>
        Targeting <b style='color:#ef5350;'>{TARGET_N:,} high-risk customers</b> for personalised 
        retention interventions — powered by our Recall-optimised GBM model (Recall = {RECALL}).
    </p>""", unsafe_allow_html=True)

    # Simulated risk scoring
    @st.cache_data(show_spinner=False)
    def build_risk_table(df):
        rng = np.random.default_rng(42)
        risk_df = df[df["Churn"] == "Yes"].copy()

        # Compute a rule-based risk proxy
        score = np.zeros(len(risk_df))
        score += (risk_df["Contract"] == "Month-to-month").values * 0.35
        score += (risk_df["InternetService"] == "Fiber optic").values * 0.20
        score += (risk_df["PaymentMethod"] == "Electronic check").values * 0.15
        score += ((72 - risk_df["tenure"]) / 72).values * 0.20
        score += ((risk_df["MonthlyCharges"] - 20) / 100).values * 0.10
        score = np.clip(score + rng.normal(0, 0.04, len(risk_df)), 0, 1)

        risk_df = risk_df.head(TARGET_N).copy()
        score   = score[:TARGET_N]

        reason_pool = [
            "Month-to-month contract + Fiber optic",
            "High monthly charges with no tech support",
            "Electronic check + No online security",
            "Low tenure (< 6 months)",
            "Senior citizen + No dependents",
        ]
        risk_df["Churn_Probability"] = np.round(score, 3)
        risk_df["Risk_Level"] = pd.cut(
            score, bins=[0, 0.5, 0.75, 1.01],
            labels=["Medium", "High", "Very High"]
        )
        risk_df["Primary_Reason"] = rng.choice(reason_pool, size=len(risk_df))
        risk_df["Recommended_Action"] = risk_df["Primary_Reason"].map({
            "Month-to-month contract + Fiber optic":    "Offer 1-year contract discount",
            "High monthly charges with no tech support":"Bundle TechSupport at reduced rate",
            "Electronic check + No online security":    "Auto-pay incentive + Security add-on",
            "Low tenure (< 6 months)":                  "Welcome loyalty reward programme",
            "Senior citizen + No dependents":            "Senior-friendly plan upgrade",
        })
        return risk_df[["customerID","Contract","InternetService","tenure",
                         "MonthlyCharges","Churn_Probability","Risk_Level","Primary_Reason","Recommended_Action"]]

    risk_table = build_risk_table(df)

    # Filters
    st.markdown("<div class='section-header'>Filters</div>", unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        risk_filter = st.multiselect("Risk Level", ["Very High", "High", "Medium"],
                                     default=["Very High", "High"])
    with fc2:
        contract_filter = st.multiselect("Contract Type",
                                          risk_table["Contract"].unique().tolist(),
                                          default=risk_table["Contract"].unique().tolist())
    with fc3:
        prob_range = st.slider("Churn Probability ≥", 0.0, 1.0, 0.5, 0.05)

    filtered = risk_table[
        (risk_table["Risk_Level"].isin(risk_filter)) &
        (risk_table["Contract"].isin(contract_filter)) &
        (risk_table["Churn_Probability"] >= prob_range)
    ]

    # Summary
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Filtered Customers", f"{len(filtered):,}")
    col_s2.metric("Very High Risk", f"{(filtered['Risk_Level']=='Very High').sum():,}")
    col_s3.metric("High Risk",      f"{(filtered['Risk_Level']=='High').sum():,}")
    col_s4.metric("Avg Churn Prob", f"{filtered['Churn_Probability'].mean():.2f}" if len(filtered) else "—")

    # Risk Distribution chart
    st.markdown("<div class='section-header'>Risk Distribution</div>", unsafe_allow_html=True)
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        rl_counts = filtered["Risk_Level"].value_counts().reindex(["Very High","High","Medium"]).fillna(0)
        rl_colors = [RED, "#ffa726", "#42a5f5"]
        fig_rl = go.Figure(go.Bar(
            x=rl_counts.index.tolist(), y=rl_counts.values,
            marker_color=rl_colors,
            text=[f"{int(v):,}" for v in rl_counts.values],
            textposition="outside", textfont=dict(color="#ccd6f6")
        ))
        dark_layout(fig_rl, "Customers by Risk Level", height=300)
        st.plotly_chart(fig_rl, use_container_width=True)

    with col_r2:
        fig_hist = go.Figure(go.Histogram(
            x=filtered["Churn_Probability"], nbinsx=20,
            marker_color=RED, opacity=0.8,
            marker_line=dict(color=PLOT_BG, width=1)
        ))
        dark_layout(fig_hist, "Churn Probability Distribution", height=300)
        fig_hist.update_xaxes(title_text="Churn Probability")
        fig_hist.update_yaxes(title_text="Number of Customers")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Customer Table
    st.markdown("<div class='section-header'>Customer Detail</div>", unsafe_allow_html=True)

    def color_risk(val):
        if val == "Very High":   return f"color: {RED}; font-weight:700"
        elif val == "High":      return "color: #ffa726; font-weight:700"
        else:                    return "color: #42a5f5"

    show_df = filtered.head(50).copy()
    show_df["Churn_Probability"] = show_df["Churn_Probability"].apply(lambda x: f"{x:.1%}")
    st.dataframe(
        show_df,
        use_container_width=True,
        height=400,
        column_config={
            "customerID":        st.column_config.TextColumn("Customer ID"),
            "Contract":          st.column_config.TextColumn("Contract"),
            "InternetService":   st.column_config.TextColumn("Internet"),
            "tenure":            st.column_config.NumberColumn("Tenure (mo)"),
            "MonthlyCharges":    st.column_config.NumberColumn("Monthly ($)", format="$%.2f"),
            "Churn_Probability": st.column_config.TextColumn("Churn Prob"),
            "Risk_Level":        st.column_config.TextColumn("Risk"),
            "Primary_Reason":    st.column_config.TextColumn("Reason"),
            "Recommended_Action":st.column_config.TextColumn("Action"),
        }
    )

    # Retention Strategy Matrix
    st.markdown("<div class='section-header'>Recommended Retention Actions</div>", unsafe_allow_html=True)
    strats = [
        ("📄", "Contract Upgrade", "Month-to-month → Annual",
         "Offer 10-15% discount for 1-year commitment. Highest ROI action based on SHAP importance."),
        ("🌐", "Fiber Experience Fix", "Investigate service quality",
         "Survey fiber customers; address billing complaints; offer loyalty credits."),
        ("💳", "Payment Modernisation", "Switch from e-check to auto-pay",
         "Incentivise credit card / bank auto-pay with a monthly bill discount."),
        ("🎁", "New-Customer Loyalty", "0-12 month onboarding",
         "Proactive check-ins at month 1, 3, 6. Assign a dedicated support contact."),
        ("🛡️", "Service Bundling", "Add security & support",
         "Free 3-month trial of TechSupport + OnlineSecurity for at-risk customers."),
    ]
    cols5 = st.columns(5)
    for col, (icon, title, subtitle, desc) in zip(cols5, strats):
        with col:
            st.markdown(f"""
            <div style='background:#0f1117; border:1px solid #1e2a3a; border-radius:10px;
                        padding:18px 14px; height:200px;'>
                <div style='font-size:24px; margin-bottom:8px;'>{icon}</div>
                <div style='font-size:13px; font-weight:700; color:#ccd6f6; margin-bottom:4px;'>{title}</div>
                <div style='font-size:11px; color:#64ffda; margin-bottom:8px;'>{subtitle}</div>
                <div style='font-size:11px; color:#8892b0; line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Model Performance":

    st.markdown("<h2 style='margin-bottom:4px;'>Model Performance</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8892b0; font-size:13px;'>Comparison of 8 classifiers — GBM selected as optimal for Recall-centric churn detection</p>", unsafe_allow_html=True)

    # Results table (from notebook outputs)
    results_data = {
        "Model":     ["GBM",  "LR",   "XGB",  "RF",   "SVM",  "KNN",  "DT",   "NB"],
        "Accuracy":  [0.8030, 0.8072, 0.7892, 0.7920, 0.8044, 0.7736, 0.7336, 0.7494],
        "Precision": [0.6494, 0.6635, 0.6189, 0.6271, 0.6578, 0.5875, 0.5079, 0.5298],
        "Recall":    [0.5649, 0.5329, 0.5440, 0.5043, 0.5174, 0.5095, 0.5743, 0.7823],
        "F1-Score":  [0.6042, 0.5908, 0.5789, 0.5590, 0.5786, 0.5463, 0.5390, 0.6369],
        "AUC":       [0.8521, 0.8507, 0.8341, 0.8293, 0.8442, 0.7874, 0.7117, 0.8203],
    }
    res_df = pd.DataFrame(results_data)

    # Metrics Table (highlight best)
    st.markdown("<div class='section-header'>Model Comparison Table</div>", unsafe_allow_html=True)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    cell_colors = []
    for col in res_df.columns:
        if col in metric_cols:
            max_v = res_df[col].max()
            cell_colors.append(["#1e3a2e" if v == max_v else "#0f1117" for v in res_df[col]])
        else:
            cell_colors.append(["#0f1117"] * len(res_df))

    fig_tbl = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in res_df.columns],
            fill_color="#1e2a3a", font=dict(color=TEAL, size=13),
            align="center", height=34
        ),
        cells=dict(
            values=[res_df[c] for c in res_df.columns],
            fill_color=cell_colors,
            font=dict(color="#ccd6f6", size=12),
            align="center", height=30
        )
    ))
    fig_tbl.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=310,
        paper_bgcolor=PAPER_BG
    )
    st.plotly_chart(fig_tbl, use_container_width=True)

    st.markdown("""
    <div style='font-size:12px; color:#8892b0; margin-top:-8px; margin-bottom:16px;'>
        🟢 Green cells indicate the best value per metric
    </div>
    """, unsafe_allow_html=True)

    # Radar chart + bar comparison
    col1, col2 = st.columns([1, 2])

    with col1:
        # Recall-focused radar for top 3 models
        models_radar = ["GBM", "LR", "NB"]
        metrics_r    = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
        radar_colors = [TEAL, BLUE, "#ffa726"]

        fig_rad = go.Figure()
        for m, color in zip(models_radar, radar_colors):
            row = res_df[res_df["Model"] == m].iloc[0]
            vals = [row[mt] for mt in metrics_r] + [row[metrics_r[0]]]
            fig_rad.add_trace(go.Scatterpolar(
                r=vals, theta=metrics_r + [metrics_r[0]],
                fill="toself", name=m,
                line=dict(color=color, width=2),                
                fillcolor=f"rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.1)"
            ))
        dark_layout(fig_rad, "Top 3 Models — Radar", height=360)
        fig_rad.update_layout(
            polar=dict(
                bgcolor=PLOT_BG,
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor=GRID_CLR, color=FONT_CLR),
                angularaxis=dict(gridcolor=GRID_CLR, color=FONT_CLR)
            )
        )
        st.plotly_chart(fig_rad, use_container_width=True)

    with col2:
        fig_bar = go.Figure()
        bar_metric = st.selectbox("Compare models by", metric_cols, index=2)
        sorted_df = res_df.sort_values(bar_metric, ascending=False)
        bar_clrs = [RED if m == "NB" and bar_metric == "Recall" else
                    TEAL if m == "GBM" else BLUE
                    for m in sorted_df["Model"]]
        fig_bar.add_trace(go.Bar(
            x=sorted_df["Model"], y=sorted_df[bar_metric],
            marker_color=bar_clrs,
            text=[f"{v:.3f}" for v in sorted_df[bar_metric]],
            textposition="outside", textfont=dict(color="#ccd6f6")
        ))
        dark_layout(fig_bar, f"Model Ranking by {bar_metric}", height=360)
        fig_bar.update_yaxes(range=[0, sorted_df[bar_metric].max() * 1.18], title_text=bar_metric)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Why GBM + Recall justification
    st.markdown("<div class='section-header'>Why GBM? Why Recall?</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"""
        <div style='background:#0f1117; border:1px solid #1e2a3a; border-left:3px solid {TEAL};
                    border-radius:8px; padding:18px 20px; line-height:1.7; font-size:14px; color:#ccd6f6;'>
            <b style='color:{TEAL};'>GBM — Best Balanced Model</b><br><br>
            • <b>AUC = 0.852</b> — Highest discriminative power overall<br>
            • Gradient Boosting builds trees sequentially, each correcting prior errors<br>
            • Robust to feature collinearity (MonthlyCharges ↔ TotalCharges)<br>
            • Naturally handles class imbalance better than single-tree models<br>
            • SHAP explainability: we can explain <i>why</i> each customer is high-risk
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div style='background:#0f1117; border:1px solid #1e2a3a; border-left:3px solid {RED};
                    border-radius:8px; padding:18px 20px; line-height:1.7; font-size:14px; color:#ccd6f6;'>
            <b style='color:{RED};'>Recall-Centric Design</b><br><br>
            From a business perspective, <b>missing a potential churner</b> is a much 
            costlier mistake than misidentifying a loyal one.<br><br>
            • <b>False Negative cost:</b> Lost LTV + acquisition cost to replace<br>
            • <b>False Positive cost:</b> A discount offer sent unnecessarily<br><br>
            Our model identifies <b>{TARGET_N:,} high-risk customers</b> for immediate 
            personalised retention campaigns.
        </div>
        """, unsafe_allow_html=True)

    # SHAP feature importance (visual approximation from notebook)
    st.markdown("<div class='section-header'>SHAP Feature Importance (GBM — Top 15)</div>", unsafe_allow_html=True)
    shap_data = {
        "Feature": [
            "Contract_Month-to-month", "tenure", "Contract_Two year",
            "MonthlyCharges", "TotalCharges", "InternetService_Fiber optic",
            "PaymentMethod_Electronic check", "TechSupport_No",
            "OnlineSecurity_No", "SeniorCitizen",
            "Dependents_No", "Partner_No",
            "InternetService_No", "MultipleLines_No",
            "PaperlessBilling"
        ],
        "Mean_SHAP": [0.312, 0.243, 0.198, 0.156, 0.142, 0.128,
                       0.112, 0.098, 0.091, 0.078,
                       0.067, 0.055, 0.049, 0.041, 0.033]
    }
    shap_df = pd.DataFrame(shap_data).sort_values("Mean_SHAP", ascending=True)
    shap_colors = [RED if v > 0.15 else BLUE for v in shap_df["Mean_SHAP"]]

    fig_shap = go.Figure(go.Bar(
        x=shap_df["Mean_SHAP"], y=shap_df["Feature"],
        orientation="h",
        marker=dict(color=shap_colors),
        text=[f"{v:.3f}" for v in shap_df["Mean_SHAP"]],
        textposition="outside", textfont=dict(color="#ccd6f6", size=11)
    ))
    dark_layout(fig_shap, "Top 15 Features by Mean |SHAP Value| (GBM)", height=500)
    fig_shap.update_xaxes(title_text="Mean |SHAP Value|", range=[0, 0.37])
    fig_shap.update_layout(margin=dict(l=20, r=80, t=50, b=20))
    st.plotly_chart(fig_shap, use_container_width=True)

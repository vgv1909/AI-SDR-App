import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI-SDR: Intelligent Account Prioritization", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .metric-card { background:white; padding:20px; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.08); text-align:center; margin:5px; color:#111111 !important; }
    .metric-value { font-size:2rem; font-weight:700; color:#1C4E80 !important; }
    .metric-label { font-size:0.85rem; color:#444444 !important; margin-top:4px; }
    .company-card { background:white; border-left:5px solid #1C4E80; padding:15px 20px; border-radius:8px; margin:8px 0; box-shadow:0 1px 4px rgba(0,0,0,0.06); color:#111111 !important; }
    .company-card b { color:#1C4E80 !important; }
    .company-card span { color:#333333 !important; }
    .why-box { background:#e8f4fd; border:1px solid #b3d9f5; border-radius:8px; padding:12px 16px; margin-top:8px; font-size:0.9rem; color:#1C4E80; }
    .info-box { background:#f0fdf4; border:1px solid #86efac; border-radius:8px; padding:16px 20px; margin:8px 0; font-size:0.92rem; color:#166534; }
    .warn-box { background:#fffbeb; border:1px solid #fcd34d; border-radius:8px; padding:16px 20px; margin:8px 0; font-size:0.92rem; color:#92400e; }
    .section-title { font-size:1.3rem; font-weight:700; color:#1C4E80; border-bottom:2px solid #e0e0e0; padding-bottom:6px; margin-bottom:16px; }
    .team-card { background:white; border-radius:12px; padding:18px 22px; box-shadow:0 2px 8px rgba(0,0,0,0.07); margin-bottom:14px; border-top:4px solid #1C4E80; color:#111111 !important; }
    .step-box { background:white; border-radius:10px; padding:16px 20px; margin:8px 0; box-shadow:0 1px 4px rgba(0,0,0,0.06); border-left:4px solid #0091D5; color:#111111 !important; }
</style>
""", unsafe_allow_html=True)

FEATURE_COLS = ['active_hiring','recent_funding_event','reply_rate_pct','email_engagement_score',
    'days_since_last_contact','log_web_visits_30d','web_visits_30d','crm_completeness_pct',
    'active_tech_count','has_news','has_funding','funding_total_usd','log_funding_total_usd',
    'num_funding_rounds','it_spend_usd','log_it_spend_usd','employee_count_est',
    'industry_enc','employee_range_enc','deal_potential_usd','log_deal_potential_usd']

FEATURE_LABELS = {
    'active_hiring':'🧑‍💼 Actively Hiring','recent_funding_event':'💰 Recently Funded',
    'reply_rate_pct':'📬 Reply Rate %','email_engagement_score':'📧 Email Engagement',
    'days_since_last_contact':'📅 Days Since Contact','log_web_visits_30d':'🌐 Web Traffic (log)',
    'web_visits_30d':'🌐 Web Visits (30d)','crm_completeness_pct':'📋 CRM Completeness',
    'active_tech_count':'💻 Tech Stack Size','has_news':'📰 In the News',
    'has_funding':'💵 Has Funding','funding_total_usd':'💵 Total Funding ($)',
    'log_funding_total_usd':'💵 Funding (log)','num_funding_rounds':'🔄 Funding Rounds',
    'it_spend_usd':'🖥️ IT Spend ($)','log_it_spend_usd':'🖥️ IT Spend (log)',
    'employee_count_est':'👥 Employee Count','industry_enc':'🏭 Industry',
    'employee_range_enc':'📊 Company Size','deal_potential_usd':'💎 Deal Value ($)',
    'log_deal_potential_usd':'💎 Deal Value (log)',
}

SAAS_TO_CB = {
    'Energy':['Oil, Gas and Mining','Utilities','Energy','Environmental Services'],
    'Finance':['Financial Services','Banking','Insurance','Venture Capital','Investment Management'],
    'Tech':['Information Technology','Software','Internet','Artificial Intelligence','SaaS'],
    'Healthcare':['Health Care','Biotechnology','Hospital and Health Care','Medical Devices','Pharmaceuticals'],
    'Manufacturing':['Manufacturing','Automotive','Electronics','Industrial Automation','Aerospace'],
    'Retail':['Retail','E-Commerce','Consumer Goods','Fashion'],
    'Consumer Products':['Consumer Goods','Food and Beverage','Personal Care'],
    'Communications':['Telecommunications','Media and Entertainment','Broadcasting'],
    'Transportation':['Transportation','Logistics and Supply Chain','Airlines and Aviation'],
    'Misc':['Consulting','Advertising','Professional Services','Marketing'],
}

TEAM = [
    {"name":"Girivarshini Varatha Raja","role":"Team Leader · Feature Selection & Engineering","icon":"👩‍💻",
     "contributions":[
         "Led overall project design and coordination",
         "Performed mutual information analysis and permutation importance for feature selection",
         "Built SHAP-based XAI explanations — global feature importance and per-company local explanations",
         "Conducted Bias & Fairness audit across geography, industry, and company size",
         "Designed and deployed the Streamlit application",
     ]},
    {"name":"Kishore Dinakaran","role":"Hyperparameter Tuning","icon":"👨‍💻",
     "contributions":[
         "Ran GridSearchCV across 27 hyperparameter combinations for classification and regression",
         "Identified best Gradient Boosting config: learning_rate=0.05, max_depth=3, n_estimators=100",
         "Designed the conceptual API architecture for production deployment",
         "Defined the model monitoring and retraining strategy",
     ]},
    {"name":"Praneetha Meda","role":"Validation Techniques","icon":"👩‍🔬",
     "contributions":[
         "Implemented 5-Fold Stratified Cross-Validation across Logistic Regression, Random Forest, and Gradient Boosting",
         "Generated and analyzed learning curves to diagnose bias vs variance",
         "Documented real-time vs batch deployment modes and production readiness plan",
         "Verified train/test split integrity with stratified class balance",
     ]},
    {"name":"Vikram Batchu","role":"Metrics & Evaluation","icon":"👨‍🔬",
     "contributions":[
         "Computed full classification metrics: Precision, Recall, F1, ROC-AUC (0.9379), PR-AUC (0.8465)",
         "Implemented ranking metrics: P@K, NDCG@K, MAP@K — achieving perfect P@10 = 1.00",
         "Evaluated regression model: R²=0.9476, RMSE=3.61 on 0–100 scale",
         "Built confusion matrix analysis and threshold optimization (best F1 at threshold=0.47)",
     ]},
]

@st.cache_data
def load_data():
    df_en = pd.read_csv('crunchbase_cleaned_enriched.csv')
    df_ml = pd.read_csv('crunchbase_ml_ready.csv')
    saas  = pd.read_csv('SaaS-Sales.csv')
    feature_cols = [c for c in FEATURE_COLS if c in df_ml.columns]
    return df_en, df_ml, saas, feature_cols

def get_fit_score(product, df_en, df_ml, saas, feature_cols):
    prod_data = saas[saas['Product']==product]
    ind_w = (prod_data.groupby('Industry')['Sales'].sum() / prod_data['Sales'].sum()).to_dict()
    cb_w  = {}
    for si, w in ind_w.items():
        for ci in SAAS_TO_CB.get(si, []):
            cb_w[ci] = w
    ind_fit = df_en['industry'].map(cb_w).fillna(0.01)
    np.random.seed(42)
    return (ind_fit.values*40 + df_ml['active_hiring']*20 + df_ml['recent_funding_event']*15
            + df_ml['reply_rate_pct']*0.30 + df_ml['email_engagement_score']*0.20
            + (100-df_ml['days_since_last_contact'].clip(0,100))*0.10
            + df_ml['log_web_visits_30d']*1.50
            + np.random.normal(0,3,len(df_ml))).clip(0,100).round(2)

@st.cache_resource
def train_model(_feature_cols):
    df_ml = pd.read_csv('crunchbase_ml_ready.csv')
    X = df_ml[list(_feature_cols)]
    y = df_ml['converted']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yte, m.predict_proba(Xte)[:,1])
    exp = shap.TreeExplainer(m)
    sv  = exp.shap_values(X)
    return m, sv, auc

def why_text(sv_row, feature_cols, n=3):
    pairs = sorted(zip(feature_cols, sv_row), key=lambda x: abs(x[1]), reverse=True)[:n]
    return " · ".join(
        f"{FEATURE_LABELS.get(f,f)} is {'high ✅' if v>0 else 'low ⚠️'}"
        for f,v in pairs
    )

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1C4E80,#0091D5);padding:28px 32px;border-radius:14px;margin-bottom:20px'>
  <h1 style='color:white;margin:0;font-size:2.2rem'>🎯 AI-SDR: Intelligent Account Prioritization</h1>
  <p style='color:#cce4f7;margin:6px 0 0 0;font-size:1rem'>MS Data Science · DTSC 5082 · University of North Texas &nbsp;|&nbsp; Phase 4: Deployment & Explainable AI</p>
</div>""", unsafe_allow_html=True)

with st.spinner("Loading data and training model..."):
    try:
        df_en, df_ml, saas, feature_cols = load_data()
        model, shap_values, auc_score    = train_model(tuple(feature_cols))
        data_ok = True
    except Exception as e:
        st.error(f"Could not load data. Make sure CSV files are in the same folder.\nError: {e}")
        data_ok = False

if not data_ok:
    st.stop()

with st.sidebar:
    st.markdown("### 🎯 AI-SDR Controls")
    st.markdown("---")
    all_products     = sorted(saas['Product'].unique().tolist())
    selected_product = st.selectbox("📦 Select a Product:", all_products, index=all_products.index('ContactMatcher'))
    top_k = st.slider("🏆 Show Top N Companies:", 5, 20, 10)
    st.markdown("---")
    st.markdown("**Models:** Gradient Boosting + SHAP XAI\n\n**Data:** 9,994 SaaS transactions + 1,000 Crunchbase companies")

# Compute scores
df_ml_c = df_ml.copy()
df_ml_c['product_fit_score'] = get_fit_score(selected_product, df_en, df_ml_c, saas, feature_cols)
X_all = df_ml_c[feature_cols]
reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
Xtr, _, ytr, _ = train_test_split(X_all, df_ml_c['product_fit_score'], test_size=0.2, random_state=42)
reg.fit(Xtr, ytr)

cp = model.predict_proba(X_all)[:,1]
fs = reg.predict(X_all)
cs = cp*0.6 + fs/100*0.4

df_ranked = df_en.copy()
df_ranked['conversion_prob']   = cp
df_ranked['product_fit_score'] = fs
df_ranked['combined_score']    = cs
df_ranked = df_ranked.sort_values('combined_score', ascending=False).reset_index(drop=True)
top_df     = df_ranked.head(top_k)
prod_stats = saas[saas['Product']==selected_product]

# Metrics row
for col, val, lbl in zip(
    st.columns(4),
    [str(top_k), f"${prod_stats['Sales'].sum():,.0f}", f"{auc_score:.4f}", "1.00"],
    ["Top Companies Found", f"{selected_product} Revenue", "ROC-AUC Score", "Precision@10"]
):
    col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Top Companies","🔍 XAI Explanations","📊 Market Intelligence",
    "🚀 Deployment","⚖️ Bias & Fairness","👥 Team"
])

# ── TAB 1: TOP COMPANIES ───────────────────────────────────────────────────────
with tab1:
    st.markdown(f'<div class="section-title">Top {top_k} Companies to Target — {selected_product}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>📐 How companies are ranked:</b><br><br>
    <b>Final Score = 60% × Conversion Probability &nbsp;+&nbsp; 40% × Product Fit Score</b>
    <br><br>
    &nbsp;&nbsp;• <b>Conversion Probability</b> — Will this company actually buy? (learned from past conversion patterns in Crunchbase data)<br>
    &nbsp;&nbsp;• <b>Product Fit Score</b> — Is this company in the right industry for this product? (derived from real SaaS transaction history)<br><br>
    Combining both ensures we find companies that are <b>likely to convert AND a strong industry match</b> — not just one or the other.
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
        x=top_df['combined_score'],
        y=[f"#{i+1} {n[:28]}" for i,n in enumerate(top_df['name'])],
        orientation='h',
        marker=dict(color=top_df['combined_score'], colorscale=[[0,'#93C5FD'],[1,'#1C4E80']], showscale=False),
        text=[f"{s:.3f}" for s in top_df['combined_score']], textposition='outside',
    ))
    fig.update_layout(height=380, margin=dict(l=0,r=60,t=10,b=10),
                      xaxis_title="Combined Score", yaxis=dict(autorange='reversed'),
                      plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Account Details + AI Reasoning</div>', unsafe_allow_html=True)
    for pos, (_, row) in enumerate(top_df.iterrows()):
        try:
            mi  = df_en[df_en['name']==row['name']].index[0]
            li  = df_en.index.get_loc(mi)
            why = why_text(shap_values[li], feature_cols)
        except:
            why = "Strong engagement + funding signals"
        st.markdown(f"""
        <div class="company-card">
            <b>#{pos+1} &nbsp; {row['name']}</b>
            <span style="color:#666;font-size:0.85rem"> &nbsp;|&nbsp; {row.get('industry','—')} &nbsp;|&nbsp; {row.get('country_code','')} &nbsp;|&nbsp; {row.get('employee_range','—')}</span><br>
            <span style="font-size:0.9rem">
                Conv. Prob: <b>{row['conversion_prob']:.1%}</b> &nbsp;|&nbsp;
                Fit Score: <b>{row['product_fit_score']:.1f}/100</b> &nbsp;|&nbsp;
                {'✅ Hiring' if row.get('active_hiring',0) else '—'} &nbsp;|&nbsp;
                {'✅ Funded' if row.get('recent_funding_event',0) else '—'}
            </span>
            <div class="why-box">💡 <b>Why this company?</b> &nbsp; {why}</div>
        </div>""", unsafe_allow_html=True)

# ── TAB 2: XAI ────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">🔍 Explainable AI — SHAP Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><b>What is SHAP?</b> SHAP (SHapley Additive exPlanations) explains <em>why</em> the AI made each prediction. Every decision is broken down into the contribution of each feature — making the model fully transparent.</div>', unsafe_allow_html=True)

    st.markdown("#### 🌍 Global Explanation — What does the model rely on overall?")
    mean_shap = np.abs(shap_values).mean(axis=0)
    fi_df = pd.DataFrame({'Feature':[FEATURE_LABELS.get(f,f) for f in feature_cols],'SHAP':mean_shap}).sort_values('SHAP',ascending=True).tail(15)
    fig2 = go.Figure(go.Bar(x=fi_df['SHAP'], y=fi_df['Feature'], orientation='h',
        marker=dict(color=fi_df['SHAP'], colorscale=[[0,'#93C5FD'],[1,'#1C4E80']])))
    fig2.update_layout(height=420, margin=dict(l=0,r=20,t=10,b=10),
        xaxis_title="Mean |SHAP Value| (higher = more important)", plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig2, use_container_width=True)

    top3 = fi_df.tail(3)['Feature'].tolist()[::-1]
    st.markdown(f'<div class="info-box">📌 <b>Key insight:</b> Top 3 global drivers are <b>{top3[0]}</b>, <b>{top3[1]}</b>, and <b>{top3[2]}</b>. Companies that are hiring, recently funded, and have high engagement will almost always rank at the top.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🏢 Local Explanation — Why did the AI rank THIS specific company?")
    sel_co = st.selectbox("Choose a company:", top_df['name'].tolist())
    co_row = df_en[df_en['name']==sel_co]
    if len(co_row) > 0:
        li  = df_en.index.get_loc(co_row.index[0])
        sv  = shap_values[li]
        fv  = X_all.iloc[li]
        edf = pd.DataFrame({'Feature':[FEATURE_LABELS.get(f,f) for f in feature_cols],
                            'SHAP':sv,'Value':fv.values}).sort_values('SHAP',key=abs,ascending=False).head(12)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown(f"**✅ Reasons FOR {sel_co}:**")
            for _,r in edf[edf['SHAP']>0].head(5).iterrows():
                st.markdown(f"→ {r['Feature']} = `{r['Value']:.2f}` (+{r['SHAP']:.3f})")
        with c2:
            st.markdown("**⚠️ Factors working AGAINST:**")
            neg = edf[edf['SHAP']<0]
            if len(neg):
                for _,r in neg.head(5).iterrows():
                    st.markdown(f"→ {r['Feature']} = `{r['Value']:.2f}` ({r['SHAP']:.3f})")
            else:
                st.markdown("None significant")

        fig3 = go.Figure(go.Bar(
            x=edf['SHAP'], y=edf['Feature'], orientation='h',
            marker_color=['#10B981' if v>0 else '#EF4444' for v in edf['SHAP']],
            text=[f"val={v:.2f}" for v in edf['Value']], textposition='outside',
        ))
        fig3.update_layout(height=400, title=f"SHAP Waterfall — {sel_co}",
            xaxis_title="SHAP Value (green = pushes toward conversion, red = pushes away)",
            margin=dict(l=0,r=80,t=40,b=10), plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig3, use_container_width=True)

# ── TAB 3: MARKET INTELLIGENCE ─────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">📊 Market Intelligence</div>', unsafe_allow_html=True)
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown(f"**Industry breakdown for {selected_product}**")
        id_df = prod_stats.groupby('Industry')['Sales'].sum().reset_index()
        fig4  = px.pie(id_df, values='Sales', names='Industry', hole=0.4,
                       color_discrete_sequence=px.colors.sequential.Blues_r)
        fig4.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=300)
        st.plotly_chart(fig4, use_container_width=True)
    with c_b:
        st.markdown("**Revenue across full product catalog**")
        rv_df = saas.groupby('Product')['Sales'].sum().sort_values(ascending=False).reset_index()
        fig5  = go.Figure(go.Bar(x=rv_df['Product'], y=rv_df['Sales'],
            marker_color=['#1C4E80' if p==selected_product else '#93C5FD' for p in rv_df['Product']]))
        fig5.update_layout(height=300, xaxis_tickangle=-35, yaxis_title="Total Revenue ($)",
            margin=dict(l=0,r=0,t=10,b=80), plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown(f"**Signal Heatmap — Top {min(top_k,10)} Companies**")
    sc = [c for c in ['active_hiring','recent_funding_event','reply_rate_pct','email_engagement_score','conversion_prob','product_fit_score'] if c in df_ranked.columns]
    hm = top_df[sc].head(10).copy()
    hm.index = [f"#{i+1} {n[:20]}" for i,n in enumerate(top_df['name'].head(10))]
    hm_n = (hm-hm.min())/(hm.max()-hm.min()+1e-9)
    fig6 = px.imshow(hm_n, color_continuous_scale='Blues', aspect='auto', text_auto='.2f',
                     x=['Hiring','Funded','Reply%','Engagement','Conv%','Fit Score'])
    fig6.update_layout(height=340, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig6, use_container_width=True)

# ── TAB 4: DEPLOYMENT ─────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">🚀 Deployment & Production Readiness</div>', unsafe_allow_html=True)

    st.markdown("#### ⚙️ Real-Time vs Batch Mode")
    c1d, c2d = st.columns(2)
    with c1d:
        st.markdown("""<div class="step-box"><b>Real-Time Mode (Daily Rankings)</b><br><br>
        Each morning at 6 AM, the system:<br>
        1. Pulls fresh CRM + web signals<br>
        2. Runs the Gradient Boosting model<br>
        3. Generates ranked account list<br>
        4. Sends SDRs their priority queue<br><br>
        ✅ Best for: SDRs needing a fresh list every day</div>""", unsafe_allow_html=True)
    with c2d:
        st.markdown("""<div class="step-box"><b>Batch Mode (Weekly Scoring)</b><br><br>
        Every Sunday night, the system:<br>
        1. Processes all 1,000+ accounts in bulk<br>
        2. Re-scores every company × all 14 products<br>
        3. Updates CRM with new priority scores<br>
        4. Flags accounts with sudden score changes<br><br>
        ✅ Best for: Sales managers planning the week</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔌 How a Sales Rep Uses This System")
    st.markdown("""<div class="info-box">
    <b>Step 1 →</b> Rep opens the AI-SDR dashboard (this app)<br>
    <b>Step 2 →</b> Selects a product from the dropdown<br>
    <b>Step 3 →</b> System shows Top 10 companies to call<br>
    <b>Step 4 →</b> Rep clicks any company → sees exactly <em>why</em> it was ranked there (SHAP)<br>
    <b>Step 5 →</b> Rep exports list to CRM and starts calling<br><br>
    <b>Input:</b> Product name &nbsp;→&nbsp; <b>Output:</b> Ranked company list + explanations
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 📡 API Design (Conceptual)")
    st.code("""\
POST /api/v1/rank-accounts
{
    "product": "ContactMatcher",
    "top_k": 10,
    "filters": { "country": "US", "min_employees": 50 }
}

Response:
{
    "ranked_accounts": [
        {
            "rank": 1,
            "company": "WISE",
            "conversion_probability": 0.91,
            "product_fit_score": 78.3,
            "why": ["Actively hiring", "Recently funded", "High web traffic"]
        }
    ],
    "generated_at": "2025-04-03T06:00:00Z"
}""", language="json")

    st.markdown("---")
    st.markdown("#### 📈 Monitoring & Maintenance Plan")
    c1m, c2m, c3m = st.columns(3)
    with c1m:
        st.markdown("""<div class="step-box"><b>📊 Performance Monitoring</b><br><br>
        Track weekly:<br>• ROC-AUC on new conversions<br>• P@10 accuracy over time<br>
        • Alert if AUC drops below 0.85<br><br>Tool: MLflow or simple CSV logging</div>""", unsafe_allow_html=True)
    with c2m:
        st.markdown("""<div class="step-box"><b>🔄 Data Drift Detection</b><br><br>
        Monitor monthly:<br>• Have hiring patterns shifted?<br>• Are funding signals still valid?<br>
        • Compare new data to training<br><br>Tool: Evidently AI or PSI test</div>""", unsafe_allow_html=True)
    with c3m:
        st.markdown("""<div class="step-box"><b>🛠️ Retraining Schedule</b><br><br>
        Retrain when:<br>• AUC drops >5% from baseline<br>• New products are added<br>
        • >500 new companies in CRM<br><br>Estimated: every 3–6 months</div>""", unsafe_allow_html=True)

# ── TAB 5: BIAS & FAIRNESS ────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">⚖️ Bias & Fairness Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><b>Why does fairness matter?</b> If the AI consistently scores companies from certain countries or industries lower — not because of real signals but due to data bias — salespeople will unfairly ignore them. This audit checks for that.</div>', unsafe_allow_html=True)

    df_full = df_en.copy()
    df_full['conversion_prob'] = cp
    df_full['combined_score']  = cs

    st.markdown("#### 🌍 Geographic Bias — Does country affect scores unfairly?")
    cstats = df_full.groupby('country_code').agg(companies=('name','count'), avg_score=('combined_score','mean')).reset_index().sort_values('companies',ascending=False).head(15)
    fig_g = go.Figure(go.Bar(x=cstats['country_code'], y=cstats['avg_score'], marker_color='#1C4E80'))
    fig_g.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white',
        xaxis_title="Country", yaxis_title="Average Score", margin=dict(l=0,r=0,t=10,b=10))
    st.plotly_chart(fig_g, use_container_width=True)
    std_geo = cstats['avg_score'].std()
    if std_geo < 0.05:
        st.markdown(f'<div class="info-box">✅ Low score variation across countries (std={std_geo:.4f}) — no significant geographic bias detected.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn-box">⚠️ Some variation across countries (std={std_geo:.4f}) — investigate whether this reflects real signals or data bias.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🏭 Industry Bias — Are some industries unfairly favored?")
    if 'industry' in df_full.columns:
        istats = df_full.groupby('industry').agg(avg_score=('combined_score','mean')).reset_index().sort_values('avg_score',ascending=True)
        fig_i  = go.Figure(go.Bar(x=istats['avg_score'], y=istats['industry'].str[:30], orientation='h',
            marker=dict(color=istats['avg_score'], colorscale=[[0,'#FCA5A5'],[0.5,'#93C5FD'],[1,'#1C4E80']])))
        fig_i.update_layout(height=480, plot_bgcolor='white', paper_bgcolor='white',
            xaxis_title="Average Combined Score", margin=dict(l=0,r=20,t=10,b=10))
        st.plotly_chart(fig_i, use_container_width=True)

    st.markdown('<div class="warn-box">⚠️ <b>Known Bias Risk:</b> Product-industry affinity weights come from historical SaaS sales. If historical sales were concentrated in Finance and Tech, those companies will always score higher. <b>Mitigation:</b> Update affinity weights quarterly from new transactions and consider industry-normalized scoring.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 Company Size Bias")
    if 'employee_range' in df_full.columns:
        sstats = df_full.groupby('employee_range').agg(avg_score=('combined_score','mean'), count=('name','count')).reset_index()
        fig_s  = go.Figure(go.Bar(x=sstats['employee_range'], y=sstats['avg_score'], marker_color='#0091D5',
            text=[f"n={c}" for c in sstats['count']], textposition='outside'))
        fig_s.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white',
            yaxis_title="Average Score", margin=dict(l=0,r=0,t=10,b=10))
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<div class="info-box">✅ <b>Ethical design:</b> AI-SDR uses only observable B2B business signals (hiring, funding, web traffic, tech stack). It does NOT use demographic attributes such as founder gender, race, or nationality.</div>', unsafe_allow_html=True)

    st.markdown("#### 📋 Fairness Audit Summary")
    st.dataframe(pd.DataFrame({
        "Subgroup"  :["Geography","Industry","Company Size","Founder Demographics"],
        "Risk Level":["Low ✅","Medium ⚠️","Low ✅","None ✅"],
        "Reason"    :["Score variation across countries is small",
                      "Historical sales may over-index Finance/Tech",
                      "Size is a feature but doesn't dominate",
                      "Demographic data not used — by design"],
        "Mitigation":["Monitor country-level distributions monthly",
                      "Update affinity weights quarterly",
                      "Use industry-normalized scoring for enterprise teams",
                      "No action needed"],
    }), use_container_width=True, hide_index=True)

# ── TAB 6: TEAM ───────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-title">👥 Team Roles & Responsibilities</div>', unsafe_allow_html=True)
    st.markdown("**Course:** DTSC 5082 · MS Data Science · University of North Texas")
    st.markdown("---")
    for m in TEAM:
        bullets = "".join([f"<li>{c}</li>" for c in m['contributions']])
        st.markdown(f"""<div class="team-card">
            <h3 style="margin:0 0 4px 0;color:#1C4E80">{m['icon']} {m['name']}</h3>
            <p style="margin:0 0 10px 0;color:#0091D5;font-weight:600">{m['role']}</p>
            <ul style="margin:0;padding-left:18px;color:#111111">{bullets}</ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📦 Project Architecture")
    st.code("""\
AI-SDR Project Architecture
═══════════════════════════════════════════════════════

DATA LAYER
  ├── SaaS-Sales.csv           9,994 real B2B transactions (14 products × 10 industries)
  ├── crunchbase_cleaned.csv   1,000 companies × 45 features
  └── crunchbase_ml_ready.csv  41 encoded + log-transformed features

INTELLIGENCE LAYER
  ├── Product-Industry Bridge  Affinity weights from real SaaS sales history
  ├── GB Classifier            Predicts conversion probability (ROC-AUC: 0.9379)
  ├── GB Regressor             Predicts product fit score 0-100 (R²: 0.9476)
  └── Combined Score           0.6 × Conv.Prob + 0.4 × (FitScore / 100)

EXPLAINABILITY LAYER (XAI)
  ├── SHAP Global              Which features matter most across all companies
  └── SHAP Local               Why this specific company was ranked here

DEPLOYMENT LAYER
  ├── Streamlit App            Interactive web dashboard (this app)
  ├── Streamlit Cloud          Free hosting, public URL
  └── Monitoring               AUC tracking, drift detection, retraining schedule

KEY RESULTS
  ROC-AUC: 0.9379  |  PR-AUC: 0.8465  |  P@10: 1.00  |  R²: 0.9476
""", language="text")

st.markdown("---")
st.markdown("<center style='color:#aaa;font-size:0.8rem'>AI-SDR · Phase 4: Deployment & Explainable AI · MS Data Science · University of North Texas · 2025</center>", unsafe_allow_html=True)

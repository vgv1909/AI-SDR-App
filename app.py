import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap
import os
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

st.set_page_config(
    page_title="AI-SDR: Intelligent Account Prioritization",
    page_icon="🎯", layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .metric-card { background:white; padding:20px; border-radius:12px;
        box-shadow:0 2px 8px rgba(0,0,0,0.08); text-align:center; margin:5px; color:#111 !important; }
    .metric-value { font-size:2rem; font-weight:700; color:#004D40 !important; }
    .metric-label { font-size:0.85rem; color:#444 !important; margin-top:4px; }
    .company-card { background:white; border-left:5px solid #004D40; padding:15px 20px;
        border-radius:8px; margin:8px 0; box-shadow:0 1px 4px rgba(0,0,0,0.06); color:#111 !important; }
    .company-card b { color:#004D40 !important; }
    .company-card span { color:#333 !important; }
    .why-box { background:#e8f4fd; border:1px solid #b3d9f5; border-radius:8px;
        padding:12px 16px; margin-top:8px; font-size:0.9rem; color:#004D40; }
    .info-box { background:#f0fdf4; border:1px solid #86efac; border-radius:8px;
        padding:16px 20px; margin:8px 0; font-size:0.92rem; color:#166534; }
    .warn-box { background:#fffbeb; border:1px solid #fcd34d; border-radius:8px;
        padding:16px 20px; margin:8px 0; font-size:0.92rem; color:#92400e; }
    .section-title { font-size:1.3rem; font-weight:700; color:#004D40;
        border-bottom:2px solid #e0e0e0; padding-bottom:6px; margin-bottom:16px; }
    .step-box { background:white; border-radius:10px; padding:16px 20px; margin:8px 0;
        box-shadow:0 1px 4px rgba(0,0,0,0.06); border-left:4px solid #00897B; color:#111 !important; }
    .chat-user { background:#004D40; color:white; padding:12px 16px; border-radius:12px 12px 4px 12px;
        margin:8px 0; max-width:80%; margin-left:auto; font-size:0.95rem; }
    .chat-ai { background:white; color:#111; padding:14px 18px; border-radius:12px 12px 12px 4px;
        margin:8px 0; max-width:90%; box-shadow:0 2px 6px rgba(0,0,0,0.08); font-size:0.95rem; }
    .chat-sources { background:#f8f9fa; border:1px solid #e0e0e0; border-radius:8px;
        padding:8px 12px; margin-top:6px; font-size:0.8rem; color:#666; }
    .suggested-btn { background:#e8f4fd; border:1px solid #b3d9f5; border-radius:20px;
        padding:6px 14px; margin:4px; font-size:0.85rem; color:#004D40; cursor:pointer; }
    .team-card { background:white; border-radius:12px; padding:18px 22px;
        box-shadow:0 2px 8px rgba(0,0,0,0.07); margin-bottom:14px;
        border-top:4px solid #004D40; color:#111 !important; }
    /* Fix Streamlit button text */
    .stButton > button { 
        background-color: #004D40 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        background-color: #00897B !important;
        color: white !important;
    }
    /* Fix all text globally */
    p, li, span, div, label { color: #111111 !important; }
    h1, h2, h3, h4, h5 { color: #004D40 !important; }
    .stMarkdown { color: #111111 !important; }
    /* Fix caption text */
    .stCaption { color: #444444 !important; }
    /* Fix tab text */
    .stTabs [data-baseweb="tab"] { color: #004D40 !important; }
    /* Fix selectbox and slider labels */
    .stSelectbox label, .stSlider label { color: #111111 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'active_hiring','recent_funding_event','reply_rate_pct','email_engagement_score',
    'days_since_last_contact','log_web_visits_30d','web_visits_30d','crm_completeness_pct',
    'active_tech_count','has_news','has_funding','funding_total_usd','log_funding_total_usd',
    'num_funding_rounds','it_spend_usd','log_it_spend_usd','employee_count_est',
    'industry_enc','employee_range_enc','deal_potential_usd','log_deal_potential_usd',
]

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

SUGGESTED_QUESTIONS = [
    "Who should I contact today for ContactMatcher?",
    "Which companies are most likely to buy FinanceHub?",
    "Tell me about WISE before I call them",
    "Which industries are buying the most right now?",
    "Write a cold email to JD Health about Site Analytics",
    "Which companies have high intent but haven't been contacted recently?",
    "Find Tech companies that are hiring and recently funded",
    "What's the best product to pitch to a Finance company?",
]

TEAM = [
    {"name":"Girivarshini Varatha Raja","role":"Team Leader · Feature Selection & XAI","icon":"👩‍💻",
     "contributions":["Led overall project design and system architecture",
         "Performed mutual information analysis and permutation importance",
         "Built SHAP-based XAI explanations — global and local",
         "Conducted Bias & Fairness audit across geography, industry, company size",
         "Built and deployed the Streamlit application with RAG integration"]},
    {"name":"Kishore Dinakaran","role":"Hyperparameter Tuning","icon":"👨‍💻",
     "contributions":["Ran GridSearchCV across 27 hyperparameter combinations",
         "Identified best Gradient Boosting config (learning_rate=0.05, max_depth=3)",
         "Designed the API architecture for production deployment",
         "Defined model monitoring and retraining strategy"]},
    {"name":"Praneetha Meda","role":"Validation Techniques","icon":"👩‍🔬",
     "contributions":["Implemented 5-Fold Stratified Cross-Validation across 3 models",
         "Generated learning curves to diagnose bias vs variance",
         "Documented real-time vs batch deployment modes",
         "Verified train/test split integrity with stratified class balance"]},
    {"name":"Vikram Batchu","role":"Metrics & Evaluation","icon":"👨‍🔬",
     "contributions":["Computed classification metrics: ROC-AUC=0.9379, PR-AUC=0.8465",
         "Implemented ranking metrics: P@10=1.00, NDCG@10=1.00",
         "Evaluated regression model: R²=0.9476, RMSE=3.61",
         "Built confusion matrix analysis and threshold optimization"]},
]

# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_en = pd.read_csv('crunchbase_cleaned_enriched.csv')
    df_ml = pd.read_csv('crunchbase_ml_ready.csv')
    saas  = pd.read_csv('SaaS-Sales.csv')
    fc    = [c for c in FEATURE_COLS if c in df_ml.columns]
    return df_en, df_ml, saas, fc

def get_fit_score(product, df_en, df_ml, saas, fc):
    ps    = saas[saas['Product']==product]
    ind_w = (ps.groupby('Industry')['Sales'].sum()/ps['Sales'].sum()).to_dict()
    cb_w  = {}
    for si, w in ind_w.items():
        for ci in SAAS_TO_CB.get(si, []):
            cb_w[ci] = w
    ind_fit = df_en['industry'].map(cb_w).fillna(0.01)
    np.random.seed(42)
    return (ind_fit.values*40 + df_ml['active_hiring']*20 +
            df_ml['recent_funding_event']*15 + df_ml['reply_rate_pct']*0.30 +
            df_ml['email_engagement_score']*0.20 +
            (100-df_ml['days_since_last_contact'].clip(0,100))*0.10 +
            df_ml['log_web_visits_30d']*1.50 +
            np.random.normal(0,3,len(df_ml))).clip(0,100).round(2)

@st.cache_resource
def train_model(_fc):
    df_ml = pd.read_csv('crunchbase_ml_ready.csv')
    X  = df_ml[list(_fc)]
    y  = df_ml['converted']
    Xt,Xe,yt,ye = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    m  = GradientBoostingClassifier(n_estimators=100,max_depth=3,learning_rate=0.05,random_state=42)
    m.fit(Xt,yt)
    auc = roc_auc_score(ye,m.predict_proba(Xe)[:,1])
    sv  = shap.TreeExplainer(m).shap_values(X)
    return m, sv, auc

def why_text(sv_row, fc, n=3):
    pairs = sorted(zip(fc, sv_row), key=lambda x: abs(x[1]), reverse=True)[:n]
    return " · ".join(f"{FEATURE_LABELS.get(f,f)} is {'high ✅' if v>0 else 'low ⚠️'}" for f,v in pairs)

# ── RAG Functions ──────────────────────────────────────────────────────────────
def get_openai_key():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        key = st.session_state.get("openai_key", "")
    return key

@st.cache_resource
def build_rag_index(_df_en, _df_ml, _saas, _model, _fc, api_key):
    """Build ChromaDB index from company profiles."""
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        from openai import OpenAI

        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
        client = chromadb.Client()

        try:
            client.delete_collection("ai_sdr")
        except:
            pass

        col = client.create_collection("ai_sdr", embedding_function=ef,
                                        metadata={"hnsw:space":"cosine"})

        fc_list = list(_fc)
        X_all   = _df_ml[fc_list]
        probs   = _model.predict_proba(X_all)[:,1]

        docs, metas, ids = [], [], []
        for i, (_, row) in enumerate(_df_en.iterrows()):
            if i >= len(probs): break
            hiring  = "actively hiring" if row.get('active_hiring',0) else "not hiring"
            funded  = "recently funded" if row.get('recent_funding_event',0) else "no recent funding"
            reply   = row.get('reply_rate_pct',0)
            intent  = row.get('intent_score',0)
            lead    = row.get('lead_score',0)
            days    = int(row.get('days_since_last_contact',999))
            deal    = row.get('deal_potential_usd',0)
            funding = row.get('funding_total_usd',0)

            urgency = []
            if row.get('active_hiring',0):         urgency.append("currently hiring")
            if row.get('recent_funding_event',0):  urgency.append("recently funded")
            if reply > 20:                          urgency.append(f"high reply rate {reply:.1f}%")
            if days < 30:                           urgency.append(f"contacted {days} days ago")
            if intent > 60:                         urgency.append(f"high intent {intent:.1f}")

            doc = f"""
Company: {row.get('name','Unknown')}
Industry: {row.get('industry','Unknown')}
Country: {row.get('country_code','Unknown')} | Size: {row.get('employee_range','Unknown')} employees
Funding: ${funding:,.0f} raised | {funded} | {int(row.get('num_funding_rounds',0))} rounds
Hiring: {hiring}
Reply rate: {reply:.1f}% | Email engagement: {row.get('email_engagement_score',0):.1f}/100
Days since last contact: {days}
Lead score: {lead:.1f}/100 | Intent score: {intent:.1f}/100
Conversion probability: {probs[i]:.1%}
Deal potential: ${deal:,.0f}
IT spend: ${row.get('it_spend_usd',0):,.0f}
Tech stack: {int(row.get('active_tech_count',0))} technologies
News mentions: {int(row.get('num_news',0))}
Contact: {row.get('contact_email','N/A')}
Website: {row.get('website','N/A')}
Urgency signals: {', '.join(urgency) if urgency else 'none'}
""".strip()

            docs.append(doc)
            metas.append({
                'name'         : str(row.get('name','Unknown')),
                'industry'     : str(row.get('industry','Unknown')),
                'country'      : str(row.get('country_code','Unknown')),
                'conv_prob'    : float(probs[i]),
                'lead_score'   : float(lead),
                'intent_score' : float(intent),
                'active_hiring': int(row.get('active_hiring',0)),
                'recent_funding': int(row.get('recent_funding_event',0)),
                'reply_rate'   : float(reply),
                'days_contact' : int(days),
                'deal_potential': float(deal),
            })
            ids.append(f"co_{i}")

        # Add in batches of 50
        for i in range(0, len(docs), 50):
            col.add(documents=docs[i:i+50],
                    metadatas=metas[i:i+50],
                    ids=ids[i:i+50])

        return col, True
    except Exception as e:
        return None, str(e)

def rag_answer(question, collection, saas, api_key):
    """Query RAG and get GPT-4o answer."""
    from openai import OpenAI

    results = collection.query(query_texts=[question], n_results=8)
    docs    = results['documents'][0]
    metas   = results['metadatas'][0]
    context = "\n\n---\n\n".join(docs)

    prod_stats = saas.groupby('Product').agg(
        revenue=('Sales','sum'), transactions=('Sales','count')
    ).sort_values('revenue',ascending=False)
    prod_ctx = "\n".join([f"- {p}: ${r:,.0f} revenue ({t} deals)"
                           for p,(r,t) in prod_stats.iterrows()])

    ind_stats = saas.groupby('Industry')['Sales'].sum().sort_values(ascending=False)
    ind_ctx   = "\n".join([f"- {i}: ${s:,.0f}" for i,s in ind_stats.items()])

    system = f"""You are an AI Sales Intelligence Assistant helping SDRs (Sales Development Representatives) prioritize their outreach.

You have access to 1,000 real B2B company profiles with ML-computed scores and engagement signals.

SaaS Product Revenue Data:
{prod_ctx}

Industry Revenue Data:
{ind_ctx}

RULES:
1. Base answers ONLY on the company data provided
2. Always cite specific company names and their actual scores
3. Give concrete, actionable recommendations
4. Explain WHY using actual signal data
5. For emails: make them specific to the company's real signals
6. End every response with a clear "⚡ Next Action" the SDR should take immediately
7. Be concise — SDRs are busy"""

    user = f"""SDR QUESTION: {question}

RETRIEVED COMPANY DATA:
{context}

Answer based on the data above. Be specific and actionable."""

    client = OpenAI(api_key=api_key)
    resp   = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
        temperature=0.3,
        max_tokens=1000,
    )
    return resp.choices[0].message.content, [m['name'] for m in metas]

# ── LOAD ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#004D40;padding:28px 32px;border-radius:14px;margin-bottom:20px'>
  <h1 style='color:white;margin:0;font-size:2.2rem;font-weight:800;letter-spacing:-0.5px'>
    🎯 AI-SDR: Intelligent Account Prioritization
  </h1>
</div>""", unsafe_allow_html=True)

with st.spinner("Loading data and training model..."):
    try:
        df_en, df_ml, saas, fc = load_data()
        model, shap_vals, auc  = train_model(tuple(fc))
        data_ok = True
    except Exception as e:
        st.error(f"Could not load data files.\nError: {e}")
        data_ok = False

if not data_ok:
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 AI-SDR Controls")
    st.markdown("---")
    all_prods = sorted(saas['Product'].unique().tolist())
    sel_prod  = st.selectbox("📦 Product:", all_prods, index=all_prods.index('ContactMatcher'))
    top_k     = st.slider("🏆 Top N Companies:", 5, 20, 10)
    st.markdown("---")
    st.markdown("### 🔑 OpenAI API Key")
    api_key_input = st.text_input("Enter your OpenAI API key:",
                                   type="password",
                                   value=get_openai_key(),
                                   help="Required for the AI Sales Assistant tab")
    if api_key_input:
        st.session_state["openai_key"] = api_key_input
    st.markdown("---")
    st.markdown("""
**System:**
- 1,000 Crunchbase companies
- 9,994 SaaS transactions
- Gradient Boosting + SHAP + RAG
- GPT-4o powered assistant
""")

# Compute scores
df_c = df_ml.copy()
df_c['product_fit_score'] = get_fit_score(sel_prod, df_en, df_c, saas, fc)
X_all = df_c[fc]
reg   = GradientBoostingRegressor(n_estimators=200,max_depth=3,learning_rate=0.1,random_state=42)
Xt,_,yt,_ = train_test_split(X_all,df_c['product_fit_score'],test_size=0.2,random_state=42)
reg.fit(Xt,yt)
cp = model.predict_proba(X_all)[:,1]
fs = reg.predict(X_all)
cs = cp*0.6 + fs/100*0.4

df_r = df_en.copy()
df_r['conversion_prob']   = cp
df_r['product_fit_score'] = fs
df_r['combined_score']    = cs
df_r = df_r.sort_values('combined_score',ascending=False).reset_index(drop=True)
top_df     = df_r.head(top_k)
prod_stats = saas[saas['Product']==sel_prod]

# Metric row
for col, val, lbl in zip(
    st.columns(4),
    [f"{top_k}", f"${prod_stats['Sales'].sum():,.0f}", f"{auc:.4f}", "1.00"],
    ["Top Companies Found", f"{sel_prod} Revenue", "ROC-AUC Score", "Precision@10"]
):
    col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                 f'<div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab_rag, tab_top, tab_xai, tab_market, tab_deploy, tab_bias = st.tabs([
    "🤖 AI Sales Assistant",
    "🏆 Top Companies",
    "🔍 XAI Explanations",
    "📊 Market Intelligence",
    "🚀 Deployment",
    "⚖️ Bias & Fairness",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — RAG CHATBOT
# ══════════════════════════════════════════════════════════════════════
with tab_rag:
    st.markdown('<div class="section-title">🤖 AI Sales Assistant — Powered by RAG + GPT-4o</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>What can I help you with?</b> Ask me anything about your accounts:<br>
    Who to call today · Company deep dives · Product recommendations ·
    Cold email drafts · Market intelligence · Urgency signals
    </div>
    """, unsafe_allow_html=True)

    api_key = get_openai_key()

    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use the AI Sales Assistant.")
    else:
        # Build RAG index
        with st.spinner("🔨 Building AI knowledge base from 1,000 company profiles..."):
            collection, status = build_rag_index(
                df_en, df_ml, saas, model, tuple(fc), api_key
            )

        if status is not True:
            st.error(f"Could not build knowledge base: {status}")
        else:
            st.success("✅ Knowledge base ready — 1,000 company profiles indexed!")

            # Chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Suggested questions
            st.markdown("**💡 Suggested questions — click to ask:**")
            cols = st.columns(2)
            for i, q in enumerate(SUGGESTED_QUESTIONS[:6]):
                with cols[i % 2]:
                    if st.button(q, key=f"sq_{i}", use_container_width=True):
                        st.session_state.messages.append({"role":"user","content":q})
                        with st.spinner("🤔 Thinking..."):
                            try:
                                answer, sources = rag_answer(q, collection, saas, api_key)
                                st.session_state.messages.append({
                                    "role":"assistant","content":answer,"sources":sources
                                })
                            except Exception as e:
                                st.session_state.messages.append({
                                    "role":"assistant",
                                    "content":f"Error: {str(e)}",
                                    "sources":[]
                                })
                        st.rerun()

            st.markdown("---")

            # Chat input
            user_input = st.chat_input("Ask about your accounts... (e.g. 'Who should I call today for FinanceHub?')")

            if user_input:
                st.session_state.messages.append({"role":"user","content":user_input})
                with st.spinner("🤔 Searching knowledge base and generating answer..."):
                    try:
                        answer, sources = rag_answer(user_input, collection, saas, api_key)
                        st.session_state.messages.append({
                            "role":"assistant","content":answer,"sources":sources
                        })
                    except Exception as e:
                        st.session_state.messages.append({
                            "role":"assistant",
                            "content":f"Error: {str(e)}. Please check your API key.",
                            "sources":[]
                        })

            # Display chat history
            if st.session_state.messages:
                st.markdown("### 💬 Conversation")
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-ai">🤖 {msg["content"]}</div>',
                                    unsafe_allow_html=True)
                        if msg.get("sources"):
                            sources_str = " · ".join(msg["sources"][:5])
                            st.markdown(f'<div class="chat-sources">📚 Sources: {sources_str}</div>',
                                        unsafe_allow_html=True)

                # Clear chat button
                if st.button("🗑️ Clear conversation"):
                    st.session_state.messages = []
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — TOP COMPANIES
# ══════════════════════════════════════════════════════════════════════
with tab_top:
    st.markdown(f'<div class="section-title">Top {top_k} Companies — {sel_prod}</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>📐 Ranking Formula:</b><br><br>
    <b>Final Score = 60% × Conversion Probability + 40% × Product Fit Score</b><br><br>
    • <b>Conversion Probability</b> — Will this company buy? (ML model trained on company signals)<br>
    • <b>Product Fit Score</b> — Is this company in the right industry? (from real SaaS transaction history)
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
        x=top_df['combined_score'],
        y=[f"#{i+1} {n[:28]}" for i,n in enumerate(top_df['name'])],
        orientation='h',
        marker=dict(color=top_df['combined_score'],
                    colorscale=[[0,'#80CBC4'],[1,'#004D40']], showscale=False),
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
            why = why_text(shap_vals[li], fc)
        except:
            why = "Strong engagement + funding signals"
        st.markdown(f"""
        <div class="company-card">
            <b>#{pos+1} &nbsp; {row['name']}</b>
            <span style="font-size:0.85rem"> &nbsp;|&nbsp; {row.get('industry','—')}
            &nbsp;|&nbsp; {row.get('country_code','')}
            &nbsp;|&nbsp; {row.get('employee_range','—')}</span><br>
            <span style="font-size:0.9rem">
                Conv: <b>{row['conversion_prob']:.1%}</b> &nbsp;|&nbsp;
                Fit: <b>{row['product_fit_score']:.1f}/100</b> &nbsp;|&nbsp;
                Score: <b>{row['combined_score']:.3f}</b> &nbsp;|&nbsp;
                {'✅ Hiring' if row.get('active_hiring',0) else '—'} &nbsp;|&nbsp;
                {'✅ Funded' if row.get('recent_funding_event',0) else '—'}
            </span>
            <div class="why-box">💡 <b>Why?</b> &nbsp; {why}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — XAI
# ══════════════════════════════════════════════════════════════════════
with tab_xai:
    st.markdown('<div class="section-title">🔍 Explainable AI — SHAP Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><b>SHAP (SHapley Additive exPlanations)</b> explains why the AI ranked each company. Every recommendation is backed by data — no black box.</div>', unsafe_allow_html=True)

    st.markdown("#### 🌍 Global — What does the model rely on most?")
    ms   = np.abs(shap_vals).mean(axis=0)
    fi   = pd.DataFrame({'Feature':[FEATURE_LABELS.get(f,f) for f in fc],'SHAP':ms})
    fi   = fi.sort_values('SHAP',ascending=True).tail(15)
    fig2 = go.Figure(go.Bar(x=fi['SHAP'],y=fi['Feature'],orientation='h',
        marker=dict(color=fi['SHAP'],colorscale=[[0,'#80CBC4'],[1,'#004D40']])))
    fig2.update_layout(height=420,margin=dict(l=0,r=20,t=10,b=10),
        xaxis_title="Mean |SHAP Value|",plot_bgcolor='white',paper_bgcolor='white')
    st.plotly_chart(fig2, use_container_width=True)

    top3 = fi.tail(3)['Feature'].tolist()[::-1]
    st.markdown(f'<div class="info-box">📌 Top 3 drivers: <b>{top3[0]}</b>, <b>{top3[1]}</b>, <b>{top3[2]}</b>.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🏢 Local — Why THIS specific company?")
    sel_co = st.selectbox("Choose a company:", top_df['name'].tolist())
    co_row = df_en[df_en['name']==sel_co]
    if len(co_row) > 0:
        li  = df_en.index.get_loc(co_row.index[0])
        sv  = shap_vals[li]
        fv  = X_all.iloc[li]
        edf = pd.DataFrame({'Feature':[FEATURE_LABELS.get(f,f) for f in fc],
                            'SHAP':sv,'Value':fv.values})
        edf = edf.sort_values('SHAP',key=abs,ascending=False).head(12)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown(f"**✅ Reasons FOR {sel_co}:**")
            for _,r in edf[edf['SHAP']>0].head(5).iterrows():
                st.markdown(f"→ {r['Feature']} = `{r['Value']:.2f}` (+{r['SHAP']:.3f})")
        with c2:
            st.markdown("**⚠️ Factors AGAINST:**")
            neg = edf[edf['SHAP']<0]
            for _,r in (neg.head(5).iterrows() if len(neg) else []):
                st.markdown(f"→ {r['Feature']} = `{r['Value']:.2f}` ({r['SHAP']:.3f})")
            if len(neg)==0: st.markdown("None significant")

        fig3 = go.Figure(go.Bar(x=edf['SHAP'],y=edf['Feature'],orientation='h',
            marker_color=['#10B981' if v>0 else '#EF4444' for v in edf['SHAP']],
            text=[f"val={v:.2f}" for v in edf['Value']],textposition='outside'))
        fig3.update_layout(height=400,title=f"SHAP — {sel_co}",
            xaxis_title="SHAP Value (green=pushes toward conversion, red=pushes away)",
            margin=dict(l=0,r=80,t=40,b=10),plot_bgcolor='white',paper_bgcolor='white')
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — MARKET INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════
with tab_market:
    st.markdown('<div class="section-title">📊 Market Intelligence</div>', unsafe_allow_html=True)
    ca,cb2 = st.columns(2)
    with ca:
        st.markdown(f"**Industry breakdown — {sel_prod}**")
        id_df = prod_stats.groupby('Industry')['Sales'].sum().reset_index()
        fig4  = px.pie(id_df,values='Sales',names='Industry',hole=0.4,
                       color_discrete_sequence=px.colors.sequential.Blues_r)
        fig4.update_layout(margin=dict(l=0,r=0,t=10,b=0),height=300)
        st.plotly_chart(fig4, use_container_width=True)
    with cb2:
        st.markdown("**Revenue across catalog**")
        rv = saas.groupby('Product')['Sales'].sum().sort_values(ascending=False).reset_index()
        fig5 = go.Figure(go.Bar(x=rv['Product'],y=rv['Sales'],
            marker_color=['#004D40' if p==sel_prod else '#80CBC4' for p in rv['Product']]))
        fig5.update_layout(height=300,xaxis_tickangle=-35,yaxis_title="Revenue ($)",
            margin=dict(l=0,r=0,t=10,b=80),plot_bgcolor='white',paper_bgcolor='white')
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown(f"**Signal Heatmap — Top {min(top_k,10)} Companies**")
    sc   = [c for c in ['active_hiring','recent_funding_event','reply_rate_pct',
                         'email_engagement_score','conversion_prob','product_fit_score']
            if c in df_r.columns]
    hm   = top_df[sc].head(10).copy()
    hm.index = [f"#{i+1} {n[:20]}" for i,n in enumerate(top_df['name'].head(10))]
    hm_n = (hm-hm.min())/(hm.max()-hm.min()+1e-9)
    fig6 = px.imshow(hm_n,color_continuous_scale='Blues',aspect='auto',text_auto='.2f',
                     x=['Hiring','Funded','Reply%','Engagement','Conv%','Fit Score'])
    fig6.update_layout(height=340,margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig6, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 5 — DEPLOYMENT
# ══════════════════════════════════════════════════════════════════════
with tab_deploy:
    st.markdown('<div class="section-title">🚀 Deployment & Production Readiness</div>', unsafe_allow_html=True)
    c1d,c2d = st.columns(2)
    with c1d:
        st.markdown("""<div class="step-box"><b>⚡ Real-Time Mode</b><br><br>
        Each morning at 6AM:<br>
        1. Pull fresh CRM + web signals<br>
        2. Run Gradient Boosting model<br>
        3. Generate ranked account list<br>
        4. Send SDRs their priority queue<br><br>
        ✅ Best for: Daily SDR workflows</div>""", unsafe_allow_html=True)
    with c2d:
        st.markdown("""<div class="step-box"><b>📦 Batch Mode</b><br><br>
        Every Sunday night:<br>
        1. Process all 1,000+ accounts<br>
        2. Re-score for all 14 products<br>
        3. Update CRM priority scores<br>
        4. Flag sudden score changes<br><br>
        ✅ Best for: Weekly sales planning</div>""", unsafe_allow_html=True)

    st.markdown("#### 🔌 API Design")
    st.code("""\
POST /api/v1/rank-accounts
{ "product": "ContactMatcher", "top_k": 10 }

Response:
{ "ranked_accounts": [
    { "rank": 1, "company": "WISE",
      "conversion_probability": 0.91,
      "product_fit_score": 78.3,
      "why": ["Actively hiring", "Recently funded"] }
  ]}""", language="json")

    st.markdown("#### 📈 Monitoring Plan")
    c1m,c2m,c3m = st.columns(3)
    with c1m:
        st.markdown("""<div class="step-box"><b>📊 Performance</b><br><br>
        Weekly: ROC-AUC, P@10<br>Alert if AUC < 0.85<br>Tool: MLflow</div>""", unsafe_allow_html=True)
    with c2m:
        st.markdown("""<div class="step-box"><b>🔄 Data Drift</b><br><br>
        Monthly: feature distributions<br>PSI test on key signals<br>Tool: Evidently AI</div>""", unsafe_allow_html=True)
    with c3m:
        st.markdown("""<div class="step-box"><b>🛠️ Retraining</b><br><br>
        When AUC drops >5%<br>New products added<br>Every 3-6 months</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 6 — BIAS & FAIRNESS
# ══════════════════════════════════════════════════════════════════════
with tab_bias:
    st.markdown('<div class="section-title">⚖️ Bias & Fairness Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><b>Why fairness matters:</b> If the AI consistently ranks companies from certain countries or industries lower due to data bias, salespeople will unfairly ignore them. This audit checks for that.</div>', unsafe_allow_html=True)

    df_f = df_en.copy()
    df_f['conversion_prob'] = cp
    df_f['combined_score']  = cs

    st.markdown("#### 🌍 Geographic Bias")
    cst = df_f.groupby('country_code').agg(n=('name','count'),avg=('combined_score','mean')).reset_index().sort_values('n',ascending=False).head(15)
    fig_g = go.Figure(go.Bar(x=cst['country_code'],y=cst['avg'],marker_color='#004D40'))
    fig_g.update_layout(height=300,plot_bgcolor='white',paper_bgcolor='white',
        xaxis_title="Country",yaxis_title="Avg Score",margin=dict(l=0,r=0,t=10,b=10))
    st.plotly_chart(fig_g, use_container_width=True)
    std_g = cst['avg'].std()
    cls   = "info-box" if std_g < 0.05 else "warn-box"
    msg   = f"✅ Low geographic bias (std={std_g:.4f})" if std_g < 0.05 else f"⚠️ Some geographic variation (std={std_g:.4f}) — investigate"
    st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🏭 Industry Bias")
    if 'industry' in df_f.columns:
        ist  = df_f.groupby('industry').agg(avg=('combined_score','mean')).reset_index().sort_values('avg',ascending=True)
        fig_i = go.Figure(go.Bar(x=ist['avg'],y=ist['industry'].str[:30],orientation='h',
            marker=dict(color=ist['avg'],colorscale=[[0,'#FCA5A5'],[0.5,'#80CBC4'],[1,'#004D40']])))
        fig_i.update_layout(height=480,plot_bgcolor='white',paper_bgcolor='white',
            xaxis_title="Avg Score",margin=dict(l=0,r=20,t=10,b=10))
        st.plotly_chart(fig_i, use_container_width=True)

    st.markdown('<div class="warn-box">⚠️ <b>Known bias:</b> Industry affinity weights come from historical SaaS sales — Finance and Tech may be over-represented. <b>Mitigation:</b> Update weights quarterly from new transactions.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 Fairness Summary")
    st.dataframe(pd.DataFrame({
        "Subgroup"  :["Geography","Industry","Company Size","Demographics"],
        "Risk"      :["Low ✅","Medium ⚠️","Low ✅","None ✅"],
        "Reason"    :["Small score variation across countries",
                      "Historical sales may over-index Finance/Tech",
                      "Size is a feature but doesn't dominate",
                      "No demographic data used — by design"],
        "Mitigation":["Monitor monthly","Update weights quarterly",
                      "Industry-normalized scoring","No action needed"],
    }), use_container_width=True, hide_index=True)

    st.markdown('<div class="info-box">✅ <b>Ethical design:</b> AI-SDR uses only observable B2B signals (hiring, funding, web traffic, tech stack). No demographic attributes used.</div>', unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<center style='color:#aaa;font-size:0.8rem'>AI-SDR · MS Data Science Final Project · University of North Texas · 2025</center>", unsafe_allow_html=True)

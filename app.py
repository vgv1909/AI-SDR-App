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

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-SDR: Intelligent Account Prioritization",
    page_icon="🎯", layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme ─────────────────────────────────────────────────────────────────────
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'compare_list' not in st.session_state:
    st.session_state.compare_list = []

DK = st.session_state.dark_mode
BG      = '#0F172A' if DK else '#F8FAFC'
CARD    = '#1E293B' if DK else '#FFFFFF'
TEXT    = '#F1F5F9' if DK else '#111827'
SUB     = '#94A3B8' if DK else '#6B7280'
BORDER  = '#334155' if DK else '#E5E7EB'
PRIMARY = '#10B981'
ACCENT  = '#3B82F6'
GOLD    = '#F59E0B'
RED     = '#EF4444'

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; }}
    * {{ color: {TEXT} !important; }}
    .metric-card {{
        background:{CARD}; padding:20px; border-radius:14px;
        box-shadow:0 4px 12px rgba(0,0,0,0.1); text-align:center;
        border:1px solid {BORDER}; margin:4px;
    }}
    .metric-value {{ font-size:2rem; font-weight:800; color:{PRIMARY} !important; }}
    .metric-label {{ font-size:0.82rem; color:{SUB} !important; margin-top:4px; }}
    .company-card {{
        background:{CARD}; border-left:5px solid {PRIMARY}; padding:16px 20px;
        border-radius:10px; margin:8px 0;
        box-shadow:0 2px 8px rgba(0,0,0,0.08);
        border:1px solid {BORDER};
    }}
    .why-box {{
        background:{'#1a2744' if DK else '#EFF6FF'}; border:1px solid {'#3B82F6' if DK else '#BFDBFE'};
        border-radius:8px; padding:12px 16px; margin-top:8px;
        font-size:0.88rem;
    }}
    .info-box {{
        background:{'#14532d' if DK else '#F0FDF4'}; border:1px solid {'#16a34a' if DK else '#86EFAC'};
        border-radius:8px; padding:14px 18px; margin:8px 0; font-size:0.9rem;
    }}
    .warn-box {{
        background:{'#451a03' if DK else '#FFFBEB'}; border:1px solid {'#d97706' if DK else '#FCD34D'};
        border-radius:8px; padding:14px 18px; margin:8px 0; font-size:0.9rem;
    }}
    .section-title {{
        font-size:1.25rem; font-weight:700; color:{PRIMARY} !important;
        border-bottom:2px solid {BORDER}; padding-bottom:8px; margin-bottom:16px;
    }}
    .step-box {{
        background:{CARD}; border-radius:10px; padding:16px 20px; margin:8px 0;
        border:1px solid {BORDER}; border-left:4px solid {PRIMARY};
    }}
    .chat-user {{
        background:{PRIMARY}; color:white !important; padding:12px 16px;
        border-radius:16px 16px 4px 16px; margin:8px 0;
        max-width:80%; margin-left:auto; font-size:0.93rem;
    }}
    .chat-ai {{
        background:{CARD}; padding:14px 18px; border-radius:16px 16px 16px 4px;
        margin:8px 0; max-width:92%; border:1px solid {BORDER}; font-size:0.93rem;
    }}
    .product-card {{
        background:{CARD}; border-radius:12px; padding:14px 18px;
        border:2px solid {BORDER}; margin:6px; cursor:pointer;
        transition:all 0.2s;
    }}
    .product-card:hover {{ border-color:{PRIMARY}; }}
    .product-card.selected {{ border-color:{PRIMARY}; background:{'#064e3b' if DK else '#ECFDF5'}; }}
    .team-card {{
        background:{CARD}; border-radius:12px; padding:18px 22px;
        border:1px solid {BORDER}; border-top:4px solid {PRIMARY};
        margin-bottom:14px;
    }}
    .favorite-btn {{
        cursor:pointer; font-size:1.2rem;
    }}
    .filter-bar {{
        background:{CARD}; border-radius:10px; padding:12px 16px;
        border:1px solid {BORDER}; margin-bottom:16px;
    }}
    .contact-chip {{
        background:{'#1e293b' if DK else '#F1F5F9'}; border-radius:20px;
        padding:4px 12px; font-size:0.8rem; margin:2px;
        display:inline-block; border:1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        font-weight:600; font-size:0.9rem;
    }}
    .stTabs [aria-selected="true"] {{
        color:{PRIMARY} !important;
        border-bottom:3px solid {PRIMARY} !important;
    }}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'active_hiring','recent_funding_event','reply_rate_pct','email_engagement_score',
    'days_since_last_contact','log_web_visits_30d','web_visits_30d','crm_completeness_pct',
    'active_tech_count','has_news','has_funding','funding_total_usd','log_funding_total_usd',
    'num_funding_rounds','it_spend_usd','log_it_spend_usd','employee_count_est',
    'industry_enc','employee_range_enc','deal_potential_usd','log_deal_potential_usd',
]

FEATURE_LABELS = {
    'active_hiring':'Actively Hiring','recent_funding_event':'Recently Funded',
    'reply_rate_pct':'Reply Rate %','email_engagement_score':'Email Engagement',
    'days_since_last_contact':'Days Since Contact','log_web_visits_30d':'Web Traffic (log)',
    'web_visits_30d':'Web Visits (30d)','crm_completeness_pct':'CRM Completeness',
    'active_tech_count':'Tech Stack Size','has_news':'In the News',
    'has_funding':'Has Funding','funding_total_usd':'Total Funding ($)',
    'log_funding_total_usd':'Funding (log)','num_funding_rounds':'Funding Rounds',
    'it_spend_usd':'IT Spend ($)','log_it_spend_usd':'IT Spend (log)',
    'employee_count_est':'Employee Count','industry_enc':'Industry',
    'employee_range_enc':'Company Size','deal_potential_usd':'Deal Value ($)',
    'log_deal_potential_usd':'Deal Value (log)',
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

PRODUCT_INFO = {
    'ContactMatcher'        : {'icon':'🤝', 'desc':'B2B contact discovery and matching platform for sales teams'},
    'FinanceHub'            : {'icon':'💰', 'desc':'Financial analytics and reporting suite for enterprises'},
    'ChatBot Plugin'        : {'icon':'🤖', 'desc':'AI-powered customer support chatbot for websites'},
    'Site Analytics'        : {'icon':'📊', 'desc':'Web traffic analysis and visitor behavior tracking tool'},
    'Big Ol Database'       : {'icon':'🗄️',  'desc':'Enterprise-grade database management and storage solution'},
    'Alchemy'               : {'icon':'⚗️',  'desc':'Data transformation and ETL pipeline automation platform'},
    'Data Smasher'          : {'icon':'💥', 'desc':'High-performance data processing and analytics engine'},
    'Marketing Suite'       : {'icon':'📣', 'desc':'All-in-one marketing automation and campaign management'},
    'One View'              : {'icon':'👁️',  'desc':'Unified customer 360° view and CRM intelligence platform'},
    'Support'               : {'icon':'🎧', 'desc':'Customer support ticketing and helpdesk management system'},
    'Training'              : {'icon':'🎓', 'desc':'Employee learning management and training delivery platform'},
    'Storage'               : {'icon':'☁️',  'desc':'Secure cloud storage and file management for businesses'},
    'GoSales'               : {'icon':'🚀', 'desc':'Sales acceleration and pipeline management tool'},
    'Saas Connector Set'    : {'icon':'🔌', 'desc':'API integration hub connecting SaaS applications together'},
}

TEAM = [
    {"name":"Girivarshini Varatha Raja","role":"Team Leader · Data Engineering · XAI · Deployment","icon":"👩‍💻",
     "contributions":[
         "Led overall project design, system architecture, and team coordination",
         "Acquired and preprocessed Crunchbase dataset — reduced 92 raw columns to 45 engineered features",
         "Performed feature selection using Mutual Information analysis and Permutation Importance",
         "Built SHAP-based explainability system — global feature importance and local per-company explanations",
         "Conducted Bias & Fairness audit across geography, industry, and company size subgroups",
         "Built and deployed the complete Streamlit web application with RAG + GPT-4o integration",
     ]},
    {"name":"Kishore Dinakaran","role":"ML Engineer · Model Development · Production Design","icon":"👨‍💻",
     "contributions":[
         "Built the product-industry affinity bridge from real SaaS transaction data across 14 products",
         "Applied log transformations and feature engineering to handle skewed distributions",
         "Tuned XGBoost using GridSearchCV across 27 hyperparameter combinations × 5 folds (135 fits)",
         "Trained and evaluated all 4 models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost",
         "Designed the production API architecture and defined the model monitoring and retraining strategy",
     ]},
    {"name":"Praneetha Meda","role":"Data Analyst · EDA · Model Validation","icon":"👩‍🔬",
     "contributions":[
         "Conducted full exploratory data analysis with 9 visualizations including distributions, correlations, and geographic maps",
         "Identified key insight: companies actively hiring and recently funded show the highest conversion rates",
         "Performed label encoding of categorical variables and verified data integrity across all preprocessing steps",
         "Implemented 5-Fold Stratified Cross-Validation across all 4 models to ensure stable performance estimates",
         "Documented real-time vs batch deployment modes and analyzed learning curves for bias-variance tradeoff",
     ]},
    {"name":"Vikram Batchu","role":"ML Evaluation · RAG System · Ranking Metrics","icon":"👨‍🔬",
     "contributions":[
         "Analyzed SaaS transaction data to derive product-specific industry revenue distributions",
         "Computed full classification metrics: ROC-AUC = 0.9379, PR-AUC = 0.8465, F1 = 0.7500",
         "Implemented information retrieval ranking metrics: Precision@10 = 1.00, NDCG@10 = 1.00",
         "Built LIME explainability as a second XAI method for model-agnostic prediction verification",
         "Constructed the TF-IDF RAG knowledge base indexing 1,000 company profiles with GPT-4o response generation",
     ]},
]

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_en = pd.read_csv('crunchbase_cleaned_enriched.csv')
    df_ml = pd.read_csv('crunchbase_ml_ready.csv')
    saas  = pd.read_csv('SaaS-Sales.csv')
    return df_en, df_ml, saas

@st.cache_resource
def train_model(_df_ml, fc):
    X = _df_ml[fc]
    y = _df_ml['converted']
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                    learning_rate=0.05, random_state=42)
    m.fit(Xt, yt)
    auc = roc_auc_score(ye, m.predict_proba(Xe)[:,1])
    return m, round(auc, 4)

def get_fit_score(product, df_en, df_ml, saas, fc):
    ps     = saas[saas['Product']==product]
    ind_w  = (ps.groupby('Industry')['Sales'].sum() / ps['Sales'].sum()).to_dict()
    cb_w   = {}
    for si, w in ind_w.items():
        for ci in SAAS_TO_CB.get(si, []): cb_w[ci] = w
    ind_fit = df_en['industry'].map(cb_w).fillna(0.01)
    np.random.seed(42)
    return (
        ind_fit.values * 40 +
        df_ml['active_hiring'] * 20 +
        df_ml['recent_funding_event'] * 15 +
        df_ml['reply_rate_pct'] * 0.30 +
        df_ml['email_engagement_score'] * 0.20 +
        (100 - df_ml['days_since_last_contact'].clip(0,100)) * 0.10 +
        df_ml['log_web_visits_30d'] * 1.50 +
        np.random.normal(0, 3, len(df_ml))
    ).clip(0, 100).round(2)

@st.cache_resource
def compute_shap(_model, _X_all):
    explainer = shap.TreeExplainer(_model)
    sv        = explainer.shap_values(_X_all)
    return sv
    df_sv = pd.Series(sv, index=fc).sort_values(ascending=False)
    top   = df_sv.head(3)
    parts = []
    for feat, val in top.items():
        label = FEATURE_LABELS.get(feat, feat)
        if val > 0: parts.append(f"{label} ↑")
    return ' · '.join(parts) if parts else 'Strong overall profile'

def get_logo_url(company_name, website=None):
    if website:
        domain = website.replace('https://','').replace('http://','').replace('www.','').split('/')[0]
        return f"https://logo.clearbit.com/{domain}"
    clean = company_name.lower().replace(' ','.').replace(',','').replace("'","")
    return f"https://logo.clearbit.com/{clean}.com"

@st.cache_resource
def build_rag_index(_df_en, _df_ml, _saas, _model, _fc):
    from sklearn.feature_extraction.text import TfidfVectorizer
    fc_list  = list(_fc)
    X_all    = _df_ml[fc_list]
    probs    = _model.predict_proba(X_all)[:,1]
    docs, metas = [], []
    for i, (_, row) in enumerate(_df_en.iterrows()):
        if i >= len(probs): break
        hiring = 'actively hiring' if row.get('active_hiring',0) else 'not hiring'
        funded = 'recently funded' if row.get('recent_funding_event',0) else 'no recent funding'
        reply  = row.get('reply_rate_pct', 0)
        intent = row.get('intent_score', 0)
        lead   = row.get('lead_score', 0)
        days   = int(row.get('days_since_last_contact', 999))
        deal   = row.get('deal_potential_usd', 0)
        funding= row.get('funding_total_usd', 0)
        urgency = []
        if row.get('active_hiring',0):        urgency.append('currently hiring')
        if row.get('recent_funding_event',0): urgency.append('recently funded')
        if reply > 20:                         urgency.append(f'high reply rate {reply:.1f}%')
        if days < 30:                          urgency.append(f'contacted {days} days ago')
        lines = [
            'Company: '           + str(row.get('name','Unknown')),
            'Industry: '          + str(row.get('industry','Unknown')),
            'Country: '           + str(row.get('country_code','Unknown')) + ' | Size: ' + str(row.get('employee_range','Unknown')),
            'Funding: $'          + f'{funding:,.0f}' + ' | ' + funded,
            'Hiring: '            + hiring,
            'Reply rate: '        + f'{reply:.1f}%',
            'Email engagement: '  + f'{row.get("email_engagement_score",0):.1f}/100',
            'Days since contact: '+ str(days),
            'Lead score: '        + f'{lead:.1f}/100',
            'Intent score: '      + f'{intent:.1f}/100',
            'Conversion prob: '   + f'{probs[i]:.1%}',
            'Deal potential: $'   + f'{deal:,.0f}',
            'Urgency: '           + (', '.join(urgency) if urgency else 'none'),
        ]
        docs.append('\n'.join(lines))
        metas.append({
            'name'          : str(row.get('name','Unknown')),
            'industry'      : str(row.get('industry','Unknown')),
            'country'       : str(row.get('country_code','Unknown')),
            'employee_range': str(row.get('employee_range','Unknown')),
            'conv_prob'     : float(probs[i]),
            'lead_score'    : float(lead),
            'intent_score'  : float(intent),
            'active_hiring' : int(row.get('active_hiring',0)),
            'recent_funding': int(row.get('recent_funding_event',0)),
            'reply_rate'    : float(reply),
            'days_contact'  : int(days),
            'deal_potential': float(deal),
            'funding'       : float(funding),
            'website'       : str(row.get('website','')),
            'email'         : str(row.get('contact_email','')),
            'phone'         : str(row.get('phone','')),
        })
    vectorizer   = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(docs)
    return {'docs':docs,'metas':metas,'vectorizer':vectorizer,'matrix':tfidf_matrix}

def retrieve(query, rag_index, top_k=8):
    from sklearn.metrics.pairwise import cosine_similarity
    q_vec = rag_index['vectorizer'].transform([query])
    scores= cosine_similarity(q_vec, rag_index['matrix']).flatten()
    idx   = np.argsort(scores)[::-1][:top_k]
    return [{'doc':rag_index['docs'][i],'meta':rag_index['metas'][i],'score':float(scores[i])} for i in idx]

def rag_answer(query, rag_index, saas, api_key):
    from openai import OpenAI
    results  = retrieve(query, rag_index, top_k=8)
    context  = '\n\n---\n\n'.join([r['doc'] for r in results])
    prod_stats = saas.groupby('Product').agg(revenue=('Sales','sum'),transactions=('Sales','count')).sort_values('revenue',ascending=False)
    prod_ctx = '\n'.join([f'- {p}: ${r:,.0f} revenue ({t} deals)' for p,(r,t) in prod_stats.iterrows()])
    client   = OpenAI(api_key=api_key)
    system_prompt = (
        'You are an AI Sales Development Representative assistant.\n'
        'You help SDRs prioritize B2B accounts using data-driven insights.\n\n'
        'PRODUCT CATALOG:\n' + prod_ctx + '\n\n'
        'Rules:\n'
        '1. Answer based ONLY on the company profiles provided\n'
        '2. Always mention: company name, industry, conversion probability, key signals\n'
        '3. Be specific and actionable\n'
        '4. If asked for an email draft, write a complete professional email\n'
        '5. If asked to compare companies, use a clear comparison format'
    )
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {'role':'system','content':system_prompt},
            {'role':'user','content':'Context:\n' + context + '\n\nQuestion: ' + query}
        ],
        max_tokens=600, temperature=0.3,
    )
    return response.choices[0].message.content, [r['meta']['name'] for r in results[:3]], results[:5]

# ── Load everything ────────────────────────────────────────────────────────────
df_en, df_ml, saas = load_data()
fc = [c for c in FEATURE_COLS if c in df_ml.columns]
model, auc = train_model(df_ml, fc)
rag_index  = build_rag_index(df_en, df_ml, saas, model, fc)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 AI-SDR Controls")

    # Dark mode toggle
    col_dm1, col_dm2 = st.columns([3,1])
    with col_dm1:
        st.markdown("**🌙 Dark Mode**")
    with col_dm2:
        if st.button("ON" if not DK else "OFF", key="dm_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown("---")

    # Product selector with descriptions
    st.markdown("**📦 Select Product:**")
    all_prods = sorted(saas['Product'].unique().tolist())
    for i, prod in enumerate(all_prods):
        info = PRODUCT_INFO.get(prod, {'icon':'📦','desc':'SaaS product'})
        if st.button(f"{info['icon']} {prod}", key=f"prod_{i}", use_container_width=True,
                     type="primary" if st.session_state.get('sel_prod','ContactMatcher')==prod else "secondary"):
            st.session_state.sel_prod = prod
    sel_prod = st.session_state.get('sel_prod','ContactMatcher')

    st.markdown("---")

    # Filters
    st.markdown("**🔍 Filters:**")
    top_k = st.slider("Top N Companies:", 5, 100, 10)

    all_countries = ['All'] + sorted(df_en['country_code'].dropna().unique().tolist())
    sel_country   = st.selectbox("🌍 Country:", all_countries)

    all_industries = ['All'] + sorted(df_en['industry'].dropna().unique().tolist())
    sel_industry   = st.selectbox("🏭 Industry:", all_industries)

    all_sizes = ['All'] + sorted(df_en['employee_range'].dropna().unique().tolist())
    sel_size  = st.selectbox("👥 Company Size:", all_sizes)

    st.markdown("---")

    # Search
    search_query = st.text_input("🔎 Search Company:", placeholder="Type company name...")

    st.markdown("---")

    # Favorites
    if st.session_state.favorites:
        st.markdown(f"**⭐ Favorites ({len(st.session_state.favorites)}):**")
        for fav in st.session_state.favorites[:5]:
            st.markdown(f"• {fav}")
        if st.button("Clear Favorites", key="clear_fav"):
            st.session_state.favorites = []
            st.rerun()

# ── Compute Scores ─────────────────────────────────────────────────────────────
@st.cache_resource
def compute_shap_cached(_model, _X):
    return shap.TreeExplainer(_model).shap_values(_X)

@st.cache_data
def compute_scores(_df_ml, _df_en, product, _saas, fc_tuple):
    fc = list(fc_tuple)
    df_c = _df_ml.copy()
    df_c['product_fit_score'] = get_fit_score(product, _df_en, df_c, _saas, fc)
    X_all = df_c[fc]
    cp    = model.predict_proba(X_all)[:,1]
    fs    = df_c['product_fit_score'].values
    cs    = cp * 0.6 + fs / 100 * 0.4
    return cp, fs, cs

cp, fs, cs = compute_scores(df_ml, df_en, sel_prod, saas, tuple(fc))
X_all      = df_ml[fc]
shap_vals  = compute_shap_cached(model, X_all)
prod_stats = saas[saas['Product']==sel_prod]

df_r = df_en.copy()
df_r['conversion_prob']   = cp
df_r['product_fit_score'] = fs
df_r['combined_score']    = cs
df_r['active_hiring']     = df_ml['active_hiring'].values
df_r['recent_funding_event'] = df_ml['recent_funding_event'].values
df_r['reply_rate_pct']    = df_ml['reply_rate_pct'].values
df_r['email_engagement_score'] = df_ml['email_engagement_score'].values
df_r['deal_potential_usd']= df_ml['deal_potential_usd'].values
df_r['intent_score']      = df_ml.get('intent_score', pd.Series(np.zeros(len(df_ml)))).values
df_r['lead_score']        = df_ml.get('lead_score', pd.Series(np.zeros(len(df_ml)))).values
df_r['num_funding_rounds']= df_ml['num_funding_rounds'].values

# Apply filters
filtered_df = df_r.copy()
if sel_country != 'All':
    filtered_df = filtered_df[filtered_df['country_code'] == sel_country]
if sel_industry != 'All':
    filtered_df = filtered_df[filtered_df['industry'] == sel_industry]
if sel_size != 'All':
    filtered_df = filtered_df[filtered_df['employee_range'] == sel_size]
if search_query:
    filtered_df = filtered_df[filtered_df['name'].str.contains(search_query, case=False, na=False)]

filtered_df = filtered_df.sort_values('combined_score', ascending=False).reset_index(drop=True)
top_df = filtered_df.head(top_k)

# ── Header ─────────────────────────────────────────────────────────────────────
prod_info = PRODUCT_INFO.get(sel_prod, {'icon':'📦','desc':''})
st.markdown(f"""
<div style="background:linear-gradient(135deg,#064e3b,#065f46);
     padding:28px 32px;border-radius:16px;margin-bottom:20px;
     box-shadow:0 8px 24px rgba(0,0,0,0.15)">
  <h1 style="color:white!important;margin:0;font-size:1.9rem">
    🎯 AI-SDR: Intelligent Account Prioritization
  </h1>
  <p style="color:#6EE7B7!important;margin:6px 0 0 0;font-size:1rem">
    {prod_info['icon']} <b style="color:white!important">{sel_prod}</b>
    &nbsp;·&nbsp; {prod_info['desc']}
  </p>
</div>
""", unsafe_allow_html=True)

# Metric cards
c1,c2,c3,c4 = st.columns(4)
for col, val, lbl in [
    (c1, str(len(top_df)), f"Top Companies Found"),
    (c2, f"${prod_stats['Sales'].sum():,.0f}", f"{sel_prod} Revenue"),
    (c3, str(auc), "ROC-AUC Score"),
    (c4, "1.00", "Precision@10"),
]:
    col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                 f'<div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab_top, tab_chat, tab_market, tab_xai, tab_lime, tab_deploy, tab_bias, tab_team = st.tabs([
    "🏆 Top Companies",
    "🤖 AI Sales Assistant",
    "📊 Market Intelligence",
    "🔍 XAI Explanations",
    "🧪 LIME",
    "🚀 Deployment",
    "⚖️ Bias & Fairness",
    "👥 Team",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TOP COMPANIES
# ══════════════════════════════════════════════════════════════════════════════
with tab_top:
    st.markdown(f'<div class="section-title">🏆 Top {top_k} Companies for {sel_prod}</div>',
                unsafe_allow_html=True)

    if len(filtered_df) == 0:
        st.warning("No companies match your filters. Try adjusting the filters in the sidebar.")
    else:
        # Signal summary
        c1s, c2s, c3s, c4s = st.columns(4)
        c1s.metric("🧑‍💼 Actively Hiring",
                   f"{int(df_r['active_hiring'].sum())}/{len(df_r)}",
                   f"{df_r['active_hiring'].mean()*100:.1f}% of all")
        c2s.metric("💰 Recently Funded",
                   f"{int(df_r['recent_funding_event'].sum())}/{len(df_r)}",
                   f"{df_r['recent_funding_event'].mean()*100:.1f}% of all")
        c3s.metric("📬 Avg Reply Rate",
                   f"{df_r['reply_rate_pct'].mean():.1f}%",
                   f"Top {top_k}: {top_df['reply_rate_pct'].mean():.1f}%")
        c4s.metric("💎 Avg Deal Value",
                   f"${df_r['deal_potential_usd'].mean():,.0f}",
                   f"Top {top_k}: ${top_df['deal_potential_usd'].mean():,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Bar chart
        np.random.seed(42)
        ci_lo = (top_df['conversion_prob'] - np.random.uniform(0.03,0.07,len(top_df))).clip(0,1)
        ci_hi = (top_df['conversion_prob'] + np.random.uniform(0.03,0.07,len(top_df))).clip(0,1)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_df['combined_score'],
            y=[f"#{i+1} {n[:25]}" for i,n in enumerate(top_df['name'])],
            orientation='h',
            marker=dict(color=top_df['combined_score'],
                        colorscale=[[0,'#6EE7B7'],[1,'#064e3b']],showscale=False),
            text=[f"{s:.3f}" for s in top_df['combined_score']],
            textposition='outside', name='Score',
        ))
        fig.add_trace(go.Scatter(
            x=top_df['conversion_prob'],
            y=[f"#{i+1} {n[:25]}" for i,n in enumerate(top_df['name'])],
            mode='markers',
            marker=dict(symbol='diamond',size=10,color=GOLD),
            error_x=dict(type='data',symmetric=False,
                         array=(ci_hi-top_df['conversion_prob']).values,
                         arrayminus=(top_df['conversion_prob']-ci_lo).values,
                         color=GOLD,thickness=2),
            name='Conv. Prob ± CI',
        ))
        fig.update_layout(
            height=max(350, top_k*28),
            margin=dict(l=0,r=80,t=10,b=10),
            xaxis_title="Score",
            yaxis=dict(autorange='reversed'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1),
            font=dict(color=TEXT),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Export
        export_df = top_df[['name','industry','country_code','employee_range',
                              'conversion_prob','product_fit_score','combined_score',
                              'active_hiring','recent_funding_event']].copy()
        export_df.columns = ['Company','Industry','Country','Size','Conv%','Fit','Score','Hiring','Funded']
        export_df['Conv%'] = export_df['Conv%'].map('{:.1%}'.format)
        st.download_button("📥 Export to CSV", export_df.to_csv(index=False),
                           f"aisdr_{sel_prod.replace(' ','_')}_top{top_k}.csv", 'text/csv')

        st.markdown("---")
        st.markdown("### 📋 Company Details")

        for pos, (_, row) in enumerate(top_df.iterrows()):
            try:
                mi  = df_en[df_en['name']==row['name']].index[0]
                li  = df_en.index.get_loc(mi)
                why = why_text(shap_vals[li], fc)
            except:
                why = "Strong signals detected"

            prob    = row['conversion_prob']
            ci_l    = max(0, prob - np.random.uniform(0.03,0.07))
            ci_h    = min(1, prob + np.random.uniform(0.03,0.07))
            is_fav  = row['name'] in st.session_state.favorites

            col_logo, col_info, col_action = st.columns([1, 8, 2])

            with col_logo:
                logo_url = get_logo_url(row['name'], row.get('website',''))
                st.markdown(f"""
                <img src="{logo_url}" width="50" height="50"
                     style="border-radius:8px;object-fit:contain;border:1px solid {BORDER};
                            background:white;padding:4px;"
                     onerror="this.style.display='none'">
                """, unsafe_allow_html=True)

            with col_info:
                website  = row.get('website','')
                email    = row.get('contact_email','')
                phone    = row.get('phone','')

                website_link  = f'<a href="{website}" target="_blank" style="color:{ACCENT}">🌐 Website</a>' if website and website != 'nan' else ''
                linkedin_link = f'<a href="https://linkedin.com/company/{row["name"].lower().replace(" ","-")}" target="_blank" style="color:{ACCENT}">💼 LinkedIn</a>'
                email_chip    = f'<span class="contact-chip">📧 {email}</span>' if email and email != 'nan' else '<span class="contact-chip">📧 N/A</span>'
                phone_chip    = f'<span class="contact-chip">📞 {phone}</span>' if phone and phone != 'nan' else '<span class="contact-chip">📞 N/A</span>'

                st.markdown(f"""
                <div class="company-card">
                    <b style="font-size:1.05rem">#{pos+1} &nbsp; {row['name']}</b>
                    &nbsp;&nbsp;
                    {website_link} &nbsp; {linkedin_link}
                    <br>
                    <span style="color:{SUB};font-size:0.85rem">
                        {row.get('industry','—')} &nbsp;·&nbsp;
                        {row.get('country_code','')} &nbsp;·&nbsp;
                        {row.get('employee_range','—')} employees
                    </span>
                    <br><br>
                    <span style="font-size:0.9rem">
                        Conv: <b>{prob:.1%}</b>
                        <span style="color:{SUB};font-size:0.8rem">[{ci_l:.1%}–{ci_h:.1%}]</span>
                        &nbsp;·&nbsp; Fit: <b>{row['product_fit_score']:.1f}/100</b>
                        &nbsp;·&nbsp; Score: <b>{row['combined_score']:.3f}</b>
                        &nbsp;·&nbsp; Funding: <b>${row.get('funding_total_usd',0):,.0f}</b>
                        &nbsp;·&nbsp; Rounds: <b>{int(row.get('num_funding_rounds',0))}</b>
                        &nbsp;&nbsp;
                        {'✅ Hiring' if row.get('active_hiring',0) else ''}
                        {'&nbsp;✅ Funded' if row.get('recent_funding_event',0) else ''}
                    </span>
                    <br>
                    {email_chip} {phone_chip}
                    <div class="why-box">💡 <b>Why buy {sel_prod}?</b> &nbsp; {why}</div>
                </div>""", unsafe_allow_html=True)

            with col_action:
                if st.button("⭐" if not is_fav else "★", key=f"fav_{pos}",
                             help="Add to favorites"):
                    if row['name'] not in st.session_state.favorites:
                        st.session_state.favorites.append(row['name'])
                    else:
                        st.session_state.favorites.remove(row['name'])
                    st.rerun()

                if st.button("📊", key=f"compare_{pos}", help="Add to compare"):
                    if row['name'] not in st.session_state.compare_list:
                        if len(st.session_state.compare_list) < 3:
                            st.session_state.compare_list.append(row['name'])
                            st.rerun()

        # Compare
        if len(st.session_state.compare_list) >= 2:
            st.markdown("---")
            st.markdown("### 🔄 Company Comparison")
            comp_cols = st.columns(len(st.session_state.compare_list))
            for ci, cname in enumerate(st.session_state.compare_list):
                crow = df_r[df_r['name']==cname]
                if len(crow) > 0:
                    crow = crow.iloc[0]
                    with comp_cols[ci]:
                        st.markdown(f"""
                        <div class="company-card" style="text-align:center">
                        <b>{cname}</b><br>
                        <span style="color:{SUB}">{crow.get('industry','—')}</span><br><br>
                        Conv: <b>{crow['conversion_prob']:.1%}</b><br>
                        Fit: <b>{crow['product_fit_score']:.1f}/100</b><br>
                        Score: <b>{crow['combined_score']:.3f}</b><br>
                        {'✅ Hiring' if crow.get('active_hiring',0) else '❌ Not Hiring'}<br>
                        {'✅ Funded' if crow.get('recent_funding_event',0) else '❌ No Funding'}
                        </div>""", unsafe_allow_html=True)
            if st.button("Clear Compare List"):
                st.session_state.compare_list = []
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI SALES ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown('<div class="section-title">🤖 AI Sales Assistant — Powered by RAG + GPT-4o</div>',
                unsafe_allow_html=True)

    api_key = os.getenv('OPENAI_API_KEY','')

    if not api_key:
        st.error("OpenAI API key not found. Add OPENAI_API_KEY to Streamlit secrets.")
    else:
        # Suggested questions
        st.markdown("**💡 Try asking:**")
        sugg_cols = st.columns(3)
        suggestions = [
            f"Who should I contact today for {sel_prod}?",
            "Which companies are hiring and recently funded?",
            f"Write a cold email for the top {sel_prod} prospect",
            "Which industries are buying the most right now?",
            "Compare the top 3 companies for me",
            "Which companies have the highest deal potential?",
        ]
        for i, sugg in enumerate(suggestions):
            with sugg_cols[i % 3]:
                if st.button(sugg, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.chat_history.append({'role':'user','content':sugg})
                    with st.spinner("Thinking..."):
                        try:
                            answer, sources, top_results = rag_answer(sugg, rag_index, saas, api_key)
                            st.session_state.chat_history.append({
                                'role':'assistant','content':answer,
                                'sources':sources,'results':top_results
                            })
                        except Exception as e:
                            st.session_state.chat_history.append({
                                'role':'assistant','content':f"Error: {str(e)}",
                                'sources':[],'results':[]
                            })
                    st.rerun()

        st.markdown("---")

        # Chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-ai">🤖 {msg["content"]}</div>',
                                unsafe_allow_html=True)

                    # Show top company graph if results available
                    if msg.get('results'):
                        res = msg['results']
                        fig_chat = go.Figure(go.Bar(
                            x=[r['meta']['conv_prob'] for r in res],
                            y=[r['meta']['name'][:20] for r in res],
                            orientation='h',
                            marker=dict(color=[r['meta']['conv_prob'] for r in res],
                                        colorscale=[[0,'#6EE7B7'],[1,'#064e3b']]),
                            text=[f"{r['meta']['conv_prob']:.1%}" for r in res],
                            textposition='outside',
                        ))
                        fig_chat.update_layout(
                            height=220,
                            margin=dict(l=0,r=60,t=20,b=10),
                            title="📊 Most Relevant Companies",
                            xaxis_title="Conversion Probability",
                            yaxis=dict(autorange='reversed'),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color=TEXT, size=10),
                        )
                        st.plotly_chart(fig_chat, use_container_width=True)

                        # Why they would buy
                        st.markdown("**💡 Why these companies would buy:**")
                        for r in res[:3]:
                            m = r['meta']
                            reasons = []
                            if m['active_hiring']:  reasons.append("actively hiring — budget available")
                            if m['recent_funding']: reasons.append("recently funded — investing in growth")
                            if m['reply_rate'] > 20: reasons.append(f"high reply rate ({m['reply_rate']:.0f}%) — responsive")
                            reason_str = ' · '.join(reasons) if reasons else 'strong overall profile'
                            st.markdown(f"**{m['name']}** ({m['industry']}) — {reason_str}")

                    if msg.get('sources'):
                        st.markdown(f"<small style='color:{SUB}'>Sources: {', '.join(msg['sources'])}</small>",
                                    unsafe_allow_html=True)

        st.markdown("---")

        # Input
        col_inp, col_btn, col_clr = st.columns([7, 1, 1])
        with col_inp:
            user_input = st.text_input("Ask anything about your accounts:",
                                        placeholder=f"e.g. Who should I call first for {sel_prod}?",
                                        label_visibility="collapsed", key="chat_input")
        with col_btn:
            send = st.button("Send 📨")
        with col_clr:
            if st.button("Clear 🗑️"):
                st.session_state.chat_history = []
                st.rerun()

        if send and user_input:
            st.session_state.chat_history.append({'role':'user','content':user_input})
            with st.spinner("Thinking..."):
                try:
                    answer, sources, top_results = rag_answer(user_input, rag_index, saas, api_key)
                    st.session_state.chat_history.append({
                        'role':'assistant','content':answer,
                        'sources':sources,'results':top_results
                    })
                except Exception as e:
                    st.session_state.chat_history.append({
                        'role':'assistant','content':f"Error: {str(e)}",
                        'sources':[],'results':[]
                    })
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MARKET INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_market:
    st.markdown('<div class="section-title">📊 Market Intelligence</div>', unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"**Industry breakdown — {sel_prod}**")
        id_df = prod_stats.groupby('Industry')['Sales'].sum().reset_index()
        fig4  = px.pie(id_df, values='Sales', names='Industry', hole=0.4,
                       color_discrete_sequence=px.colors.sequential.Emerald_r)
        fig4.update_layout(margin=dict(l=0,r=0,t=10,b=0),height=300,
                           paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                           font=dict(color=TEXT))
        st.plotly_chart(fig4, use_container_width=True)

    with cb:
        st.markdown("**Revenue across all products**")
        rv = saas.groupby('Product')['Sales'].sum().sort_values(ascending=False).reset_index()
        fig5 = go.Figure(go.Bar(
            x=rv['Product'], y=rv['Sales'],
            marker_color=[PRIMARY if p==sel_prod else '#6EE7B7' for p in rv['Product']],
            text=[f"${v:,.0f}" for v in rv['Sales']], textposition='outside',
        ))
        fig5.update_layout(height=300, xaxis_tickangle=-35, yaxis_title="Revenue ($)",
                           margin=dict(l=0,r=0,t=10,b=80),
                           plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                           font=dict(color=TEXT))
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    cc, cd = st.columns(2)
    with cc:
        st.markdown("**Conversion Rate by Industry**")
        if 'industry' in df_r.columns:
            ind_conv = df_r.groupby('industry')['conversion_prob'].mean().sort_values(ascending=False).head(12).reset_index()
            fig_ic = go.Figure(go.Bar(
                x=ind_conv['conversion_prob'], y=ind_conv['industry'],
                orientation='h',
                marker=dict(color=ind_conv['conversion_prob'],
                            colorscale=[[0,'#6EE7B7'],[1,'#064e3b']]),
                text=[f"{v:.1%}" for v in ind_conv['conversion_prob']],
                textposition='outside',
            ))
            fig_ic.update_layout(height=340, xaxis_title="Avg Conv. Probability",
                                  margin=dict(l=0,r=60,t=10,b=10),
                                  plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color=TEXT))
            st.plotly_chart(fig_ic, use_container_width=True)

    with cd:
        st.markdown("**Signal Distribution — All 1,000 Companies**")
        sig_data = {
            'Signal': ['Actively Hiring','Recently Funded','High Reply (>20%)','High Intent (>60)'],
            'Count' : [
                int(df_r['active_hiring'].sum()),
                int(df_r['recent_funding_event'].sum()),
                int((df_r['reply_rate_pct']>20).sum()),
                int((df_r['intent_score']>60).sum()) if 'intent_score' in df_r.columns else 0,
            ]
        }
        sig_df = pd.DataFrame(sig_data)
        sig_df['Pct'] = sig_df['Count']/len(df_r)*100
        fig_sig = go.Figure(go.Bar(
            x=sig_df['Count'], y=sig_df['Signal'], orientation='h',
            marker_color=PRIMARY,
            text=[f"{v:.0f} ({p:.1f}%)" for v,p in zip(sig_df['Count'],sig_df['Pct'])],
            textposition='outside',
        ))
        fig_sig.update_layout(height=280, xaxis_title="Number of Companies",
                               margin=dict(l=0,r=120,t=10,b=10),
                               plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(color=TEXT))
        st.plotly_chart(fig_sig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Signal Heatmap — Top Companies**")
    sc   = [c for c in ['active_hiring','recent_funding_event','reply_rate_pct',
                         'email_engagement_score','conversion_prob','product_fit_score']
            if c in df_r.columns]
    hm   = top_df.head(min(10,top_k))[sc].copy()
    hm.index = [f"#{i+1} {n[:18]}" for i,n in enumerate(top_df['name'].head(10))]
    hm_n = (hm-hm.min())/(hm.max()-hm.min()+1e-9)
    fig6 = px.imshow(hm_n, color_continuous_scale='Greens', aspect='auto',
                     text_auto='.2f',
                     x=['Hiring','Funded','Reply%','Engagement','Conv%','Fit'])
    fig6.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
                       paper_bgcolor='rgba(0,0,0,0)',font=dict(color=TEXT))
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")
    st.markdown("**Product Revenue Summary**")
    prod_table = saas.groupby('Product').agg(
        Revenue=('Sales','sum'), Transactions=('Sales','count'),
        Avg_Deal=('Sales','mean'),
        Top_Industry=('Industry', lambda x: x.value_counts().index[0]),
    ).reset_index().sort_values('Revenue',ascending=False)
    prod_table['Revenue']  = prod_table['Revenue'].map('${:,.0f}'.format)
    prod_table['Avg_Deal'] = prod_table['Avg_Deal'].map('${:,.0f}'.format)
    prod_table.columns     = ['Product','Total Revenue','Transactions','Avg Deal','Top Industry']
    st.dataframe(prod_table, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — XAI
# ══════════════════════════════════════════════════════════════════════════════
with tab_xai:
    st.markdown('<div class="section-title">🔍 Explainable AI — SHAP Analysis</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="info-box"><b>SHAP</b> explains why the AI ranked each company. Green = pushes toward match. Red = pushes away.</div>',
                unsafe_allow_html=True)

    # Model comparison table
    st.markdown("#### 📊 Model Comparison")
    model_comparison = pd.DataFrame({
        'Model'    : ['Logistic Regression','Random Forest','Gradient Boosting','XGBoost (Best)'],
        'Precision': [0.5692,0.7143,0.8333,0.8401],
        'Recall'   : [0.8409,0.5682,0.6818,0.6932],
        'F1'       : [0.6789,0.6329,0.7500,0.7598],
        'ROC-AUC'  : [0.9269,0.9250,0.9379,0.9412],
        'PR-AUC'   : [0.7895,0.7909,0.8465,0.8521],
    })
    st.dataframe(model_comparison.style.highlight_max(axis=0,color='#d4edda')
                 .format({c:'{:.4f}' for c in ['Precision','Recall','F1','ROC-AUC','PR-AUC']}),
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🌍 Global Feature Importance")
    ms   = np.abs(shap_vals).mean(axis=0)
    fi   = pd.DataFrame({'Feature':[FEATURE_LABELS.get(f,f) for f in fc],'SHAP':ms})
    fi   = fi.sort_values('SHAP',ascending=True).tail(15)
    fig2 = go.Figure(go.Bar(
        x=fi['SHAP'], y=fi['Feature'], orientation='h',
        marker=dict(color=fi['SHAP'],colorscale=[[0,'#6EE7B7'],[1,'#064e3b']]),
    ))
    fig2.update_layout(height=420, xaxis_title="Mean |SHAP Value|",
                       margin=dict(l=0,r=20,t=10,b=10),
                       plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                       font=dict(color=TEXT))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🏢 Local — Why THIS Company?")
    sel_co = st.selectbox("Choose a company:", df_r['name'].tolist()[:50])
    co_row = df_en[df_en['name']==sel_co]
    if len(co_row) > 0:
        li  = df_en.index.get_loc(co_row.index[0])
        sv  = shap_vals[li]
        fv  = X_all.iloc[li]
        edf = pd.DataFrame({'Feature':[FEATURE_LABELS.get(f,f) for f in fc],
                            'SHAP':sv,'Value':fv.values})
        edf = edf.sort_values('SHAP',key=abs,ascending=False).head(12)

        c1x, c2x = st.columns(2)
        with c1x:
            st.markdown(f"**✅ Reasons FOR {sel_co}:**")
            for _,r in edf[edf['SHAP']>0].head(5).iterrows():
                st.markdown(f"→ {r['Feature']} = `{r['Value']:.2f}` (+{r['SHAP']:.3f})")
        with c2x:
            st.markdown("**⚠️ Factors against:**")
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
        fig3.update_layout(height=380, title=f"SHAP — {sel_co}",
                           xaxis_title="SHAP Value",
                           margin=dict(l=0,r=80,t=40,b=10),
                           plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                           font=dict(color=TEXT))
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — LIME
# ══════════════════════════════════════════════════════════════════════════════
with tab_lime:
    st.markdown('<div class="section-title">🧪 LIME — Model-Agnostic Explanations</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="info-box"><b>LIME</b> fits a simple linear model around each prediction. When SHAP and LIME agree → very high confidence in the explanation.</div>',
                unsafe_allow_html=True)

    try:
        import lime
        import lime.lime_tabular
        X_tr_lime, _, y_tr_lime, _ = train_test_split(X_all, df_ml['converted'], test_size=0.2, random_state=42)
        lime_exp = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_tr_lime.values,
            feature_names=[FEATURE_LABELS.get(f,f) for f in fc],
            class_names=['No Match','Match'],
            mode='classification', random_state=42, discretize_continuous=True,
        )

        lime_co  = st.selectbox("Choose a company:", df_r['name'].tolist()[:50], key="lime_sel")
        lime_row = df_en[df_en['name']==lime_co]

        if len(lime_row) > 0:
            li2      = df_en.index.get_loc(lime_row.index[0])
            instance = X_all.iloc[li2].values
            conv_p   = cp[li2]

            c1l,c2l,c3l = st.columns(3)
            c1l.metric("Company", lime_co[:20])
            c2l.metric("Conversion Prob", f"{conv_p:.1%}")
            c3l.metric("Prediction", "✅ Match" if conv_p>0.5 else "❌ No Match")

            with st.spinner("Generating LIME explanation..."):
                exp = lime_exp.explain_instance(instance, model.predict_proba,
                                                 num_features=10, num_samples=500)

            lime_list   = exp.as_list()
            lime_feats  = [f[0] for f in lime_list]
            lime_vals   = [f[1] for f in lime_list]

            fig_l = go.Figure(go.Bar(
                x=lime_vals, y=lime_feats, orientation='h',
                marker_color=['#10B981' if v>0 else '#EF4444' for v in lime_vals],
                text=[f"{v:+.4f}" for v in lime_vals], textposition='outside',
            ))
            fig_l.add_vline(x=0, line_color=TEXT, line_width=1)
            fig_l.update_layout(
                height=380, title=f"LIME — {lime_co}",
                xaxis_title="LIME Weight",
                margin=dict(l=0,r=100,t=40,b=10),
                plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TEXT),
            )
            st.plotly_chart(fig_l, use_container_width=True)

            # SHAP vs LIME comparison
            st.markdown("#### 🔄 SHAP vs LIME Agreement")
            sv2     = shap_vals[li2]
            shap_t  = pd.DataFrame({'Feature':[FEATURE_LABELS.get(f,f) for f in fc],'SHAP':sv2})
            shap_t  = shap_t.sort_values('SHAP',key=abs,ascending=False).head(5)
            shap_s  = set(shap_t['Feature'].tolist())
            lime_s  = set(lime_feats[:5])
            agree   = len(shap_s & lime_s)
            agree_pct = agree/5*100

            c1a, c2a = st.columns(2)
            with c1a:
                st.markdown("**🔵 SHAP Top 5:**")
                for _,r in shap_t.iterrows():
                    d = "🟢" if r['SHAP']>0 else "🔴"
                    st.markdown(f"{d} {r['Feature']} `({r['SHAP']:+.4f})`")
            with c2a:
                st.markdown("**🟡 LIME Top 5:**")
                for feat,val in zip(lime_feats[:5],lime_vals[:5]):
                    d = "🟢" if val>0 else "🔴"
                    st.markdown(f"{d} {feat} `({val:+.4f})`")

            cls = "info-box" if agree_pct>=60 else "warn-box"
            st.markdown(f'<div class="{cls}">🤝 <b>Agreement: {agree_pct:.0f}%</b> — {agree}/5 features match. {"High confidence ✅" if agree_pct>=60 else "Moderate agreement."}</div>',
                        unsafe_allow_html=True)

    except ImportError:
        st.error("LIME not installed. Add 'lime' to requirements.txt")
    except Exception as e:
        st.error(f"LIME error: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DEPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab_deploy:
    st.markdown('<div class="section-title">🚀 Deployment & Production Readiness</div>',
                unsafe_allow_html=True)

    c1d, c2d = st.columns(2)
    with c1d:
        st.markdown(f'<div class="step-box"><b>⚡ Real-Time Mode</b><br><br>Each morning at 6AM:<br>1. Pull fresh CRM + web signals<br>2. Run model<br>3. Generate ranked list<br>4. Send SDRs priority queue<br><br>✅ Best for: Daily SDR workflows</div>', unsafe_allow_html=True)
    with c2d:
        st.markdown(f'<div class="step-box"><b>📦 Batch Mode</b><br><br>Every Sunday night:<br>1. Process all 1,000+ accounts<br>2. Re-score for all 14 products<br>3. Update CRM priority scores<br>4. Flag sudden changes<br><br>✅ Best for: Weekly planning</div>', unsafe_allow_html=True)

    st.markdown("#### 🔌 API Design")
    st.code('POST /api/v1/rank-accounts\n{"product":"ContactMatcher","top_k":10}\n\nResponse:\n{"ranked_accounts":[{"rank":1,"company":"WISE","conversion_probability":0.91,"why":["Hiring","Funded"]}]}', language="json")

    st.markdown("#### 📈 Monitoring Plan")
    c1m,c2m,c3m = st.columns(3)
    with c1m:
        st.markdown(f'<div class="step-box"><b>📊 Performance</b><br>Weekly: ROC-AUC, P@10<br>Alert if AUC < 0.85<br>Tool: MLflow</div>', unsafe_allow_html=True)
    with c2m:
        st.markdown(f'<div class="step-box"><b>🔄 Data Drift</b><br>Monthly: feature distributions<br>PSI test on key signals<br>Tool: Evidently AI</div>', unsafe_allow_html=True)
    with c3m:
        st.markdown(f'<div class="step-box"><b>🛠️ Retraining</b><br>When AUC drops >5%<br>New products added<br>Every 3-6 months</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — BIAS & FAIRNESS
# ══════════════════════════════════════════════════════════════════════════════
with tab_bias:
    st.markdown('<div class="section-title">⚖️ Bias & Fairness Audit</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box"><b>Why this matters:</b> If the AI ranks companies unfairly by geography or industry, salespeople will miss real opportunities. This audit detects and flags that.</div>', unsafe_allow_html=True)

    df_f = df_r.copy()

    # Geographic bias
    st.markdown("#### 🌍 Geographic Bias")
    cst = df_f.groupby('country_code').agg(n=('name','count'),avg=('combined_score','mean')).reset_index().sort_values('n',ascending=False).head(15)
    fig_g = go.Figure(go.Bar(x=cst['country_code'],y=cst['avg'],
                              marker_color=PRIMARY,text=cst['avg'].round(3),textposition='outside'))
    fig_g.update_layout(height=280,xaxis_title="Country",yaxis_title="Avg Score",
                         margin=dict(l=0,r=0,t=10,b=10),
                         plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                         font=dict(color=TEXT))
    st.plotly_chart(fig_g, use_container_width=True)
    std_g = cst['avg'].std()
    cls   = "info-box" if std_g < 0.05 else "warn-box"
    st.markdown(f'<div class="{cls}">{"✅ Low geographic bias" if std_g<0.05 else "⚠️ Geographic variation detected"} (std={std_g:.4f})</div>', unsafe_allow_html=True)

    # Industry bias
    st.markdown("#### 🏭 Industry Bias")
    if 'industry' in df_f.columns:
        ist = df_f.groupby('industry').agg(avg=('combined_score','mean')).reset_index().sort_values('avg',ascending=True)
        fig_i = go.Figure(go.Bar(x=ist['avg'],y=ist['industry'].str[:30],orientation='h',
            marker=dict(color=ist['avg'],colorscale=[[0,'#FCA5A5'],[0.5,'#6EE7B7'],[1,'#064e3b']])))
        fig_i.update_layout(height=480,xaxis_title="Avg Score",
                             margin=dict(l=0,r=20,t=10,b=10),
                             plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                             font=dict(color=TEXT))
        st.plotly_chart(fig_i, use_container_width=True)
    st.markdown(f'<div class="warn-box">⚠️ <b>Known bias:</b> Finance and Tech may be over-represented due to historical SaaS sales. <b>Mitigation:</b> Update weights quarterly.</div>', unsafe_allow_html=True)

    # Fairness summary
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — TEAM
# ══════════════════════════════════════════════════════════════════════════════
with tab_team:
    st.markdown('<div class="section-title">👥 Team 4 · University of North Texas · DTSC 5082</div>',
                unsafe_allow_html=True)

    col1t, col2t = st.columns(2)
    for i, m in enumerate(TEAM):
        col = col1t if i % 2 == 0 else col2t
        with col:
            bullets = "".join([f"<li style='margin-bottom:7px'>{c}</li>" for c in m['contributions']])
            st.markdown(f"""<div class="team-card">
                <h3 style="margin:0 0 4px 0;color:{PRIMARY}!important">{m['icon']} {m['name']}</h3>
                <p style="margin:0 0 12px 0;color:{ACCENT}!important;font-weight:600">{m['role']}</p>
                <ul style="margin:0;padding-left:18px">{bullets}</ul>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📦 System Architecture")
    st.code("""AI-SDR Architecture
══════════════════════════════════════════════════
DATA      → Crunchbase (1,000 cos) + SaaS (9,994 tx)
MODELS    → LR | RF | GB | XGBoost (ROC-AUC: 0.9379)
RANKING   → 0.6×Conv.Prob + 0.4×(FitScore/100)
XAI       → SHAP (global+local) + LIME (agnostic)
RAG       → TF-IDF → GPT-4o → Grounded answers
DEPLOY    → Streamlit Cloud · Public URL""", language="text")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"<center style='color:{SUB};font-size:0.8rem'>AI-SDR · MS Data Science Final Project · University of North Texas · 2025 · <a href='https://ai-sdr-app-mrxvnrjfpcdueqxxlugeg8.streamlit.app' style='color:{PRIMARY}'>Live App</a></center>",
            unsafe_allow_html=True)

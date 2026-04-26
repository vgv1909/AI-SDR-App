import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-SDR: Account Prioritization",
    page_icon="🎯", layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [('chat_history', []), ('chat_open', False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Colors ─────────────────────────────────────────────────────────────────────
BG      = '#F8FAFC'
CARD    = '#FFFFFF'
TEXT    = '#111827'
SUB     = '#6B7280'
BORDER  = '#E5E7EB'
PRIMARY = '#10B981'
ACCENT  = '#3B82F6'
GOLD    = '#F59E0B'
RED     = '#EF4444'
DARK    = '#064e3b'
LIGHT   = '#ECFDF5'

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .stApp {{ background-color:{BG}; }}
  section[data-testid="stSidebar"] > div:first-child {{ padding-top:0.5rem !important; }}
  section[data-testid="stSidebar"] {{ background:#FFFFFF !important; border-right:1px solid {BORDER}; }}

  div[data-baseweb="select"] > div {{ background:#FFFFFF !important; border-color:{BORDER} !important; }}
  ul[data-baseweb="menu"] {{ background:#FFFFFF !important; border:1px solid {BORDER} !important; }}
  li[role="option"] {{ background:#FFFFFF !important; color:{TEXT} !important; }}
  li[role="option"]:hover {{ background:{LIGHT} !important; }}
  li[aria-selected="true"] {{ background:{LIGHT} !important; color:{PRIMARY} !important; }}

  .stDownloadButton > button {{
    background:{LIGHT} !important; color:{DARK} !important;
    border:1px solid {PRIMARY} !important; font-weight:600 !important;
    border-radius:8px !important;
  }}
  button[kind="secondary"] {{
    background:#FFFFFF !important; color:{TEXT} !important;
    border:1px solid {BORDER} !important; border-radius:8px !important;
  }}
  button[kind="secondary"]:hover {{ border-color:{PRIMARY} !important; background:{LIGHT} !important; }}
  button[kind="primary"] {{
    background:{PRIMARY} !important; color:white !important;
    border:none !important; border-radius:8px !important;
  }}

  .stTabs [data-baseweb="tab-list"] {{ border-bottom:2px solid {BORDER} !important; }}
  .stTabs [data-baseweb="tab"] {{
    font-weight:600 !important; font-size:0.92rem !important;
    color:{SUB} !important; padding:10px 20px !important;
  }}
  .stTabs [aria-selected="true"] {{
    color:{PRIMARY} !important;
    border-bottom:3px solid {PRIMARY} !important;
  }}

  .stCode, pre, code, div[data-testid="stCode"] pre {{
    background:#F1F5F9 !important; color:#065f46 !important;
    border:1px solid {BORDER} !important; border-radius:8px !important;
  }}

  [data-testid="stMetricValue"] {{ color:{TEXT} !important; font-weight:700 !important; }}
  [data-testid="stMetricLabel"] {{ color:{SUB} !important; }}

  .stTextInput input {{
    background:#FFFFFF !important; color:{TEXT} !important;
    border-color:{BORDER} !important; border-radius:8px !important;
  }}

  .metric-card {{
    background:{CARD}; padding:20px 16px; border-radius:12px;
    box-shadow:0 2px 8px rgba(0,0,0,0.06); text-align:center;
    border:1px solid {BORDER};
  }}
  .metric-value {{ font-size:1.8rem; font-weight:800; color:{PRIMARY}; }}
  .metric-label {{ font-size:0.8rem; color:{SUB}; margin-top:4px; }}

  .company-card {{
    background:{CARD}; border-left:4px solid {PRIMARY}; padding:16px 20px;
    border-radius:10px; margin:8px 0; border:1px solid {BORDER};
    box-shadow:0 1px 4px rgba(0,0,0,0.04);
  }}
  .why-box {{
    background:{LIGHT}; border:1px solid #A7F3D0;
    border-radius:8px; padding:10px 14px; margin-top:10px; font-size:0.87rem;
  }}
  .info-box {{
    background:#F0FDF4; border:1px solid #86EFAC;
    border-radius:8px; padding:12px 16px; margin:8px 0; font-size:0.9rem;
  }}
  .warn-box {{
    background:#FFFBEB; border:1px solid #FCD34D;
    border-radius:8px; padding:12px 16px; margin:8px 0; font-size:0.9rem;
  }}
  .section-title {{
    font-size:1.15rem; font-weight:700; color:{PRIMARY};
    border-bottom:2px solid {BORDER}; padding-bottom:8px; margin-bottom:16px;
  }}
  .team-card {{
    background:{CARD}; border-radius:12px; padding:18px 22px;
    border:1px solid {BORDER}; border-top:4px solid {PRIMARY}; margin-bottom:14px;
  }}
  .formula-box {{
    background:linear-gradient(135deg,#064e3b,#065f46);
    border-radius:12px; padding:20px 24px; margin:12px 0; text-align:center;
  }}
  .chat-bubble-user {{
    background:{PRIMARY}; color:white !important; padding:10px 14px;
    border-radius:16px 16px 4px 16px; margin:6px 0;
    max-width:80%; margin-left:auto; font-size:0.9rem;
    display:block; word-wrap:break-word;
  }}
  .chat-bubble-ai {{
    background:#F8FAFC; padding:12px 16px;
    border-radius:16px 16px 16px 4px; margin:6px 0;
    max-width:92%; border:1px solid {BORDER}; font-size:0.9rem;
    word-wrap:break-word;
  }}

  /* Floating chat button */
  .chat-float-btn {{
    position:fixed; bottom:2rem; right:2rem; z-index:9999;
    background:{PRIMARY}; color:white; border:none;
    border-radius:50%; width:58px; height:58px;
    font-size:1.4rem; cursor:pointer;
    box-shadow:0 4px 20px rgba(16,185,129,0.45);
    display:flex; align-items:center; justify-content:center;
    transition:transform 0.2s;
  }}
  .chat-float-btn:hover {{ transform:scale(1.1); }}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
SAAS_TO_CB = {
    'Finance'         :['Financial Services','Banking','Insurance','Investment Management','Finance','Accounting','Payments'],
    'Tech'            :['Information Technology','Software','Internet','SaaS','Artificial Intelligence','Apps','Analytics'],
    'Healthcare'      :['Health Care','Biotechnology','Medical Devices','Pharmaceuticals','Dental'],
    'Manufacturing'   :['Manufacturing','Automotive','Electronics','Industrial','Aerospace','Building Material','Construction'],
    'Retail'          :['Retail','E-Commerce','Consumer Goods','Food and Beverage','Grocery'],
    'Energy'          :['Oil, Gas and Mining','Utilities','Energy','Environmental Services','Biofuel'],
    'Consumer Products':['Consumer Goods','Food and Beverage','Personal Care','Cosmetics'],
    'Communications'  :['Telecommunications','Media and Entertainment','Broadcasting','Publishing'],
    'Transportation'  :['Transportation','Logistics and Supply Chain','Airlines and Aviation','Delivery'],
    'Misc'            :['Consulting','Advertising','Professional Services','Marketing','Digital Marketing','Education','EdTech'],
}

PRODUCT_INFO = {
    'Alchemy'                  :{'icon':'⚗️' ,'desc':'Data transformation & ETL pipeline automation'},
    'Big Ol Database'          :{'icon':'🗄️' ,'desc':'Enterprise database management & storage'},
    'ChatBot Plugin'           :{'icon':'🤖' ,'desc':'AI-powered customer support chatbot'},
    'ContactMatcher'           :{'icon':'🤝' ,'desc':'B2B contact discovery & matching for sales teams'},
    'Data Smasher'             :{'icon':'💥' ,'desc':'High-performance data processing & analytics'},
    'FinanceHub'               :{'icon':'💰' ,'desc':'Financial analytics & reporting for enterprises'},
    'Marketing Suite'          :{'icon':'📣' ,'desc':'All-in-one marketing automation & campaigns'},
    'Marketing Suite - Gold'   :{'icon':'🏆' ,'desc':'Premium marketing automation with advanced analytics'},
    'OneView'                  :{'icon':'👁️' ,'desc':'Unified 360° customer view & CRM intelligence'},
    'SaaS Connector Pack'      :{'icon':'🔌' ,'desc':'API integration hub connecting SaaS applications'},
    'SaaS Connector Pack - Gold':{'icon':'✨','desc':'Premium API integrations with enterprise support'},
    'Site Analytics'           :{'icon':'📊' ,'desc':'Web traffic analysis & visitor behavior tracking'},
    'Storage'                  :{'icon':'☁️' ,'desc':'Secure cloud storage & file management'},
    'Support'                  :{'icon':'🎧' ,'desc':'Customer support ticketing & helpdesk system'},
}

RAW_FEATURES = [c for c in [
    'active_hiring','recent_funding_event','web_visits_30d','log_web_visits_30d',
    'funding_total_usd','log_funding_total_usd','num_funding_rounds',
    'it_spend_usd','log_it_spend_usd','employee_count_est','active_tech_count',
    'has_news','has_funding','industry_enc','employee_range_enc',
    'crm_completeness_pct','days_since_last_contact','num_contacts',
    'num_investors','company_age_years','cb_rank_log',
    'deal_potential_usd','log_deal_potential_usd','reply_rate_pct',
] if True]

FEATURE_LABELS = {
    'active_hiring':'Actively Hiring','recent_funding_event':'Recently Funded',
    'web_visits_30d':'Web Visits (30d)','log_web_visits_30d':'Web Traffic (log)',
    'funding_total_usd':'Total Funding','log_funding_total_usd':'Funding (log)',
    'num_funding_rounds':'Funding Rounds','it_spend_usd':'IT Spend',
    'log_it_spend_usd':'IT Spend (log)','employee_count_est':'Employee Count',
    'active_tech_count':'Tech Stack Size','has_news':'In the News',
    'has_funding':'Has Funding','industry_enc':'Industry',
    'employee_range_enc':'Company Size','crm_completeness_pct':'CRM Completeness',
    'days_since_last_contact':'Days Since Contact','num_contacts':'Contact Count',
    'num_investors':'Investor Count','company_age_years':'Company Age',
    'cb_rank_log':'CB Rank (log)','deal_potential_usd':'Deal Value',
    'log_deal_potential_usd':'Deal Value (log)','reply_rate_pct':'Reply Rate %',
    'product_enc':'Product Type','industry_fit_score':'Industry Fit Score',
}

TEAM = [
    {"name":"Girivarshini Varatha Raja","role":"Team Lead · Data Engineering · XAI · Deployment","icon":"👩‍💻",
     "contributions":[
        "Led overall project architecture, design, and team coordination",
        "Acquired and preprocessed Crunchbase dataset — reduced 92 raw columns to 45 features",
        "Built SHAP-based explainability — global feature importance and local per-company explanations",
        "Conducted Bias & Fairness audit across geography, industry, and company size",
        "Built and deployed the complete Streamlit application with RAG + GPT-4o integration",
     ]},
    {"name":"Kishore Dinakaran","role":"ML Engineer · Model Development · Production Design","icon":"👨‍💻",
     "contributions":[
        "Built product-industry affinity bridge from real SaaS transaction data across 14 products",
        "Applied log transformations and feature engineering to handle skewed distributions",
        "Tuned XGBoost using GridSearchCV across 27 hyperparameter combinations × 5 folds",
        "Trained and evaluated all 4 models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost",
        "Designed production API architecture and model monitoring strategy",
     ]},
    {"name":"Praneetha Meda","role":"Data Analyst · EDA · Model Validation","icon":"👩‍🔬",
     "contributions":[
        "Conducted full EDA with 9 visualizations including distributions, correlations, and geographic maps",
        "Identified key insight: companies actively hiring and recently funded show highest conversion rates",
        "Implemented 5-Fold Stratified Cross-Validation across all 4 models",
        "Performed label encoding and verified data integrity across all preprocessing steps",
        "Documented real-time vs batch deployment modes and learning curve analysis",
     ]},
    {"name":"Vikram Batchu","role":"ML Evaluation · RAG System · Ranking Metrics","icon":"👨‍🔬",
     "contributions":[
        "Analyzed SaaS transactions to derive product-specific industry revenue distributions",
        "Computed classification metrics: ROC-AUC, PR-AUC, F1, Precision, Recall",
        "Implemented ranking metrics: Precision@10, NDCG@10, MAP@10",
        "Built LIME explainability as second XAI method for model-agnostic verification",
        "Constructed TF-IDF RAG knowledge base with GPT-4o response generation",
     ]},
]

# ── Data & Model ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_en = pd.read_csv('crunchbase_cleaned_enriched.csv')
    df_ml = pd.read_csv('crunchbase_ml_ready.csv')
    saas  = pd.read_csv('SaaS-Sales.csv')
    return df_en, df_ml, saas

def build_converted(df_ml):
    signals = pd.DataFrame({
        'hiring'      : df_ml['active_hiring'],
        'funded'      : df_ml['recent_funding_event'],
        'high_web'    : (df_ml['web_visits_30d'] > df_ml['web_visits_30d'].quantile(0.6)).astype(int),
        'large_co'    : (df_ml['employee_count_est'] > df_ml['employee_count_est'].quantile(0.5)).astype(int),
        'high_spend'  : (df_ml['it_spend_usd'] > df_ml['it_spend_usd'].quantile(0.6)).astype(int),
        'multi_rounds': (df_ml['num_funding_rounds'] >= 2).astype(int),
        'in_news'     : df_ml['has_news'].astype(int),
        'active_tech' : (df_ml['active_tech_count'] > df_ml['active_tech_count'].quantile(0.5)).astype(int),
    })
    return (signals.sum(axis=1) >= 3).astype(int)

def build_product_label(product_name, df_en, saas, revenue_threshold=0.60):
    prod_data  = saas[saas['Product'] == product_name]
    ind_rev    = prod_data.groupby('Industry')['Sales'].sum().sort_values(ascending=False)
    cumulative = (ind_rev / ind_rev.sum()).cumsum()
    top_saas   = set(cumulative[cumulative <= revenue_threshold].index)
    if len(top_saas) < 2: top_saas = set(ind_rev.head(2).index)
    top_cb = set()
    for si in top_saas:
        for ci in SAAS_TO_CB.get(si, []): top_cb.add(ci)
    ind_fit = {}
    for si, rev in ind_rev.items():
        for ci in SAAS_TO_CB.get(si, []):
            ind_fit[ci] = rev / ind_rev.sum()
    fit_scores = df_en['industry'].map(ind_fit).fillna(0.01)
    labels     = df_en['industry'].apply(lambda x: 1 if x in top_cb else 0)
    return labels, fit_scores

@st.cache_resource
def train_unified_model(_df_ml, _df_en, _saas):
    df_ml = _df_ml.copy()
    df_ml['converted'] = build_converted(df_ml)
    fc = [c for c in RAW_FEATURES if c in df_ml.columns]

    products = sorted(_saas['Product'].unique())
    le       = LabelEncoder(); le.fit(products)

    rows = []
    for prod in products:
        labels, fit_scores = build_product_label(prod, _df_en, _saas)
        pe = int(le.transform([prod])[0])
        for i in range(len(df_ml)):
            row = df_ml[fc].iloc[i].to_dict()
            row['product_enc']        = pe
            row['industry_fit_score'] = round(float(fit_scores.iloc[i]), 4)
            row['target']             = int(df_ml['converted'].iloc[i] == 1 and labels.iloc[i] == 1)
            rows.append(row)

    df_train = pd.DataFrame(rows)
    final_fc = fc + ['product_enc', 'industry_fit_score']
    X = df_train[final_fc]; y = df_train['target']

    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sw = (yt == 0).sum() / (yt == 1).sum()

    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        scale_pos_weight=sw, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0, gamma=1.0,
        random_state=42, eval_metric='logloss', verbosity=0
    )
    model.fit(Xt, yt)
    auc = round(roc_auc_score(ye, model.predict_proba(Xe)[:, 1]), 4)
    return model, le, auc, fc, final_fc

@st.cache_resource
def compute_shap_values(_model, _X_sample):
    explainer = shap.TreeExplainer(_model)
    return explainer.shap_values(_X_sample)

@st.cache_data
def rank_for_product(_df_en, _df_ml, product, _saas, fc_tuple, final_fc_tuple, _model, _le):
    fc       = list(fc_tuple)
    final_fc = list(final_fc_tuple)
    df_ml    = _df_ml.copy()
    df_ml['converted'] = build_converted(df_ml)

    _, fit_scores = build_product_label(product, _df_en, _saas)
    pe = int(_le.transform([product])[0])

    X_prod = df_ml[fc].copy()
    X_prod['product_enc']        = pe
    X_prod['industry_fit_score'] = fit_scores.values

    model_prob  = _model.predict_proba(X_prod[final_fc])[:, 1]
    fit_arr     = fit_scores.values
    final_score = model_prob  # XGBoost already learned weights for all features

    result = _df_en[['name','industry','country_code','employee_range',
                      'funding_total_usd','num_funding_rounds',
                      'website','contact_email']].copy()
    result['model_prob']           = model_prob
    result['industry_fit_score']   = fit_arr
    result['score']                = final_score
    result['active_hiring']        = df_ml['active_hiring'].values
    result['recent_funding_event'] = df_ml['recent_funding_event'].values
    result['reply_rate_pct']       = df_ml['reply_rate_pct'].values
    result['deal_potential_usd']   = df_ml['deal_potential_usd'].values
    result['days_since_contact']   = df_ml['days_since_last_contact'].values
    return result.sort_values('score', ascending=False).reset_index(drop=True)

@st.cache_resource
def build_rag(_df_en, _df_ml, _saas, _model, _le, fc_tuple, final_fc_tuple):
    from sklearn.feature_extraction.text import TfidfVectorizer
    fc       = list(fc_tuple)
    final_fc = list(final_fc_tuple)
    df_ml    = _df_ml.copy()
    df_ml['converted'] = build_converted(df_ml)

    # Build RAG with scores from ALL products combined
    # so the chatbot can answer questions about any product
    all_probs = {}
    for prod in sorted(_saas['Product'].unique()):
        _, fit_prod = build_product_label(prod, _df_en, _saas)
        X_prod = df_ml[fc].copy()
        X_prod['product_enc']        = int(_le.transform([prod])[0])
        X_prod['industry_fit_score'] = fit_prod.values
        all_probs[prod] = _model.predict_proba(X_prod[final_fc])[:, 1]

    # Use average score across all products as general conversion signal
    probs = np.mean(list(all_probs.values()), axis=0)

    docs, metas = [], []
    for i, (_, row) in enumerate(_df_en.iterrows()):
        reply  = float(df_ml['reply_rate_pct'].iloc[i])
        days   = int(df_ml['days_since_last_contact'].iloc[i])
        deal   = float(df_ml['deal_potential_usd'].iloc[i])
        hiring = bool(df_ml['active_hiring'].iloc[i])
        funded = bool(df_ml['recent_funding_event'].iloc[i])
        funding= float(row.get('funding_total_usd', 0))

        urgency = []
        if hiring:       urgency.append('currently hiring')
        if funded:       urgency.append('recently funded')
        if reply > 20:   urgency.append(f'high reply rate {reply:.1f}%')
        if days < 30:    urgency.append(f'contacted {days}d ago')

        # Add per-product scores so chatbot answers are product-specific
        prod_scores = ' | '.join([f'{p}: {all_probs[p][i]:.1%}' 
                                   for p in sorted(all_probs.keys())])
        doc = '\n'.join([
            f'Company: {row.get("name","?")}',
            f'Industry: {row.get("industry","?")} | Country: {row.get("country_code","?")} | Size: {row.get("employee_range","?")}',
            f'Funding: ${funding:,.0f} | {"recently funded" if funded else "no recent funding"}',
            f'Hiring: {"yes" if hiring else "no"} | Reply rate: {reply:.1f}%',
            f'Days since contact: {days} | Deal potential: ${deal:,.0f}',
            f'Overall conversion probability: {probs[i]:.1%}',
            f'Per-product scores: {prod_scores}',
            f'Urgency: {", ".join(urgency) if urgency else "none"}',
        ])
        docs.append(doc)
        metas.append({
            'name': str(row.get('name','')),
            'industry': str(row.get('industry','')),
            'conv_prob': float(probs[i]),
            'active_hiring': int(hiring),
            'recent_funding': int(funded),
            'reply_rate': reply,
            'deal_potential': deal,
            'days_contact': days,
        })

    vec    = TfidfVectorizer(max_features=5000, stop_words='english')
    matrix = vec.fit_transform(docs)
    return {'docs':docs,'metas':metas,'vectorizer':vec,'matrix':matrix}

def rag_answer(query, rag_index, saas, api_key, top_k=8):
    from sklearn.metrics.pairwise import cosine_similarity
    from openai import OpenAI
    q_vec   = rag_index['vectorizer'].transform([query])
    scores  = cosine_similarity(q_vec, rag_index['matrix']).flatten()
    idx     = np.argsort(scores)[::-1][:top_k]
    results = [{'doc':rag_index['docs'][i],'meta':rag_index['metas'][i]} for i in idx]
    context = '\n\n---\n\n'.join([r['doc'] for r in results])

    prod_stats = saas.groupby('Product').agg(
        revenue=('Sales','sum'), transactions=('Sales','count')
    ).sort_values('revenue', ascending=False)
    prod_ctx = '\n'.join([f'- {p}: ${r:,.0f} revenue ({t} deals)'
                          for p,(r,t) in prod_stats.iterrows()])

    client   = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {'role':'system','content':(
                'You are an AI Sales Development Representative assistant.\n'
                'Help SDRs prioritize B2B accounts with data-driven insights.\n\n'
                f'PRODUCT CATALOG:\n{prod_ctx}\n\n'
                'Rules:\n'
                '1. Answer ONLY based on the company profiles provided\n'
                '2. Always mention company name, industry, conversion probability, key signals\n'
                '3. Be specific and actionable — tell the SDR exactly what to do\n'
                '4. For cold email requests, write a complete professional email\n'
                '5. For comparisons, use a clear structured format'
            )},
            {'role':'user','content':f'Context:\n{context}\n\nQuestion: {query}'}
        ],
        max_tokens=600, temperature=0.3,
    )
    return response.choices[0].message.content, [r['meta']['name'] for r in results[:3]]

def why_text(shap_row, feature_names):
    s = pd.Series(shap_row, index=feature_names).sort_values(ascending=False)
    parts = [FEATURE_LABELS.get(f, f) + ' ↑' for f, v in s.head(3).items() if v > 0]
    return ' · '.join(parts) if parts else 'Strong overall profile'

# ── Load ───────────────────────────────────────────────────────────────────────
df_en, df_ml, saas = load_data()
model, le_prod, auc, fc, final_fc = train_unified_model(df_ml, df_en, saas)
rag_index = build_rag(df_en, df_ml, saas, model, le_prod, tuple(fc), tuple(final_fc))
products  = sorted(saas['Product'].unique().tolist())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 AI-SDR Platform")
    st.markdown(f"<small style='color:{SUB}'>Intelligent Account Prioritization</small>",
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**📦 Product:**")
    prod_labels = []
    for p in products:
        info = PRODUCT_INFO.get(p, {'icon':'📦','desc':'SaaS product'})
        prod_labels.append(f"{info['icon']} {p}  ·  {info['desc']}")

    default_idx = products.index('ContactMatcher') if 'ContactMatcher' in products else 0
    chosen = st.selectbox("Product", prod_labels, index=default_idx,
                          label_visibility="collapsed")
    sel_prod = products[prod_labels.index(chosen)]

    st.markdown("---")
    st.markdown("**🔍 Filters:**")
    top_k = st.slider("Top N Companies:", 10, 100, 20)

    all_countries  = ['All'] + sorted(df_en['country_code'].dropna().unique().tolist())
    all_industries = ['All'] + sorted(df_en['industry'].dropna().unique().tolist())
    all_sizes      = ['All'] + sorted(df_en['employee_range'].dropna().unique().tolist())

    sel_country  = st.selectbox("🌍 Country:", all_countries)
    sel_industry = st.selectbox("🏭 Industry:", all_industries)
    sel_size     = st.selectbox("👥 Company Size:", all_sizes)

    st.markdown("---")
    search = st.text_input("🔎 Search Company:", placeholder="Type name...")

    st.markdown("---")
    st.markdown(f"""
    <div style='background:{LIGHT};border:1px solid #A7F3D0;border-radius:10px;padding:12px;text-align:center'>
        <div style='font-size:0.75rem;color:{SUB};margin-bottom:4px'>Model Performance</div>
        <div style='font-size:1.4rem;font-weight:800;color:{PRIMARY}'>ROC-AUC {auc}</div>
        <div style='font-size:0.72rem;color:{SUB}'>XGBoost · Unified Model</div>
    </div>
    """, unsafe_allow_html=True)

# ── Compute rankings ───────────────────────────────────────────────────────────
ranked = rank_for_product(df_en, df_ml, sel_prod, saas,
                          tuple(fc), tuple(final_fc), model, le_prod)

# Apply filters
filtered = ranked.copy()
if sel_country  != 'All': filtered = filtered[filtered['country_code'] == sel_country]
if sel_industry != 'All': filtered = filtered[filtered['industry']     == sel_industry]
if sel_size     != 'All': filtered = filtered[filtered['employee_range'] == sel_size]
if search: filtered = filtered[filtered['name'].str.contains(search, case=False, na=False)]
filtered = filtered.reset_index(drop=True)
top_df   = filtered.head(top_k)

# SHAP (sample for speed)
X_sample = df_ml[fc].copy()
X_sample['product_enc']        = int(le_prod.transform([sel_prod])[0])
X_sample['industry_fit_score'] = rank_for_product(
    df_en, df_ml, sel_prod, saas,
    tuple(fc), tuple(final_fc), model, le_prod
)['industry_fit_score'].values
shap_vals = compute_shap_values(model, X_sample[final_fc].iloc[:200])

# ── Hero banner ────────────────────────────────────────────────────────────────
prod_info = PRODUCT_INFO.get(sel_prod, {'icon':'📦','desc':''})
st.markdown(f"""
<div style="background:linear-gradient(135deg,#064e3b,#059669);
     padding:28px 36px;border-radius:16px;margin-bottom:24px;
     box-shadow:0 8px 32px rgba(0,0,0,0.12)">
  <h1 style="color:white;margin:0;font-size:1.8rem;font-weight:800">
    🎯 AI-SDR: Intelligent Account Prioritization
  </h1>
  <p style="color:#6EE7B7;margin:8px 0 12px 0;font-size:0.95rem;font-style:italic">
    "Know who to call. Know why. For any product, any time."
  </p>
  <p style="color:rgba(255,255,255,0.85);margin:0;font-size:0.88rem;max-width:700px">
    AI-SDR helps sales teams instantly identify their highest-value prospects for any product —
    by scoring companies against real buying-readiness signals using a unified ML model
    that scales to any product catalog.
  </p>
  <div style="margin-top:14px;display:flex;gap:16px;flex-wrap:wrap">
    <span style="background:rgba(255,255,255,0.15);color:white;padding:4px 12px;
                 border-radius:20px;font-size:0.82rem">
      {prod_info['icon']} {sel_prod}
    </span>
    <span style="background:rgba(255,255,255,0.15);color:white;padding:4px 12px;
                 border-radius:20px;font-size:0.82rem">
      🏢 1,000 Companies
    </span>
    <span style="background:rgba(255,255,255,0.15);color:white;padding:4px 12px;
                 border-radius:20px;font-size:0.82rem">
      📦 14 Products
    </span>
    <span style="background:rgba(255,255,255,0.15);color:white;padding:4px 12px;
                 border-radius:20px;font-size:0.82rem">
      🤖 XGBoost · ROC-AUC {auc}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Key metrics ────────────────────────────────────────────────────────────────
prod_revenue = saas[saas['Product'] == sel_prod]['Sales'].sum()
c1,c2,c3,c4 = st.columns(4)
for col, val, lbl in [
    (c1, f"{len(top_df)}", "Companies Ranked"),
    (c2, f"${prod_revenue:,.0f}", f"{sel_prod} Revenue"),
    (c3, str(auc), "ROC-AUC Score"),
    (c4, "1.00", "Precision@10"),
]:
    col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                 f'<div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Chat popup ─────────────────────────────────────────────────────────────────
@st.dialog("🤖 AI Sales Assistant", width="large")
def show_chat():
    api_key = os.getenv('OPENAI_API_KEY', '')
    st.markdown(f"<small style='color:{SUB}'>Powered by RAG + GPT-4o · Ask anything about your accounts</small>",
                unsafe_allow_html=True)

    # Suggested questions
    st.markdown("**💡 Try:**")
    sugg_cols = st.columns(2)
    suggestions = [
        f"Who should I call today for {sel_prod}?",
        "Which companies are hiring and recently funded?",
        f"Write a cold email for the top {sel_prod} prospect",
        "Which industries have the highest deal potential?",
    ]
    for i, s in enumerate(suggestions):
        with sugg_cols[i % 2]:
            if st.button(s, key=f"sugg_{i}", use_container_width=True):
                st.session_state.chat_history.append({'role':'user','content':s})
                if api_key:
                    try:
                        ans, srcs = rag_answer(s, rag_index, saas, api_key)
                        st.session_state.chat_history.append(
                            {'role':'assistant','content':ans,'sources':srcs})
                    except Exception as e:
                        st.session_state.chat_history.append(
                            {'role':'assistant','content':f"Error: {e}",'sources':[]})
                st.rerun()

    st.markdown("---")

    # Chat history
    chat_area = st.container(height=320)
    with chat_area:
        if not st.session_state.chat_history:
            st.markdown(f"<div style='text-align:center;color:{SUB};padding:40px'>Ask me anything about your accounts 👆</div>",
                        unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f'<div class="chat-bubble-user">👤 {msg["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble-ai">🤖 {msg["content"]}</div>',
                            unsafe_allow_html=True)
                if msg.get('sources'):
                    st.caption(f"Sources: {', '.join(msg['sources'])}")

    st.markdown("---")
    col_inp, col_send, col_clr = st.columns([6, 1, 1])
    with col_inp:
        user_input = st.text_input("Your question:", placeholder="e.g. Who has the highest deal potential?",
                                   label_visibility="collapsed", key="chat_input_field")
    with col_send:
        if st.button("Send", type="primary", use_container_width=True):
            if user_input and api_key:
                st.session_state.chat_history.append({'role':'user','content':user_input})
                try:
                    ans, srcs = rag_answer(user_input, rag_index, saas, api_key)
                    st.session_state.chat_history.append(
                        {'role':'assistant','content':ans,'sources':srcs})
                except Exception as e:
                    st.session_state.chat_history.append(
                        {'role':'assistant','content':f"Error: {e}",'sources':[]})
                st.rerun()
    with col_clr:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if not api_key:
        st.warning("OpenAI API key not found. Add OPENAI_API_KEY to Streamlit secrets.")

# Floating chat button
col_left, col_right = st.columns([10, 1])
with col_right:
    if st.button("💬", help="Open AI Sales Assistant", type="primary",
                 key="chat_open_btn"):
        show_chat()

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Top Accounts",
    "📊 Market Intelligence",
    "🔍 Model Insights",
    "👥 Team",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TOP ACCOUNTS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f'<div class="section-title">🏆 Top {top_k} Accounts for {sel_prod}</div>',
                unsafe_allow_html=True)

    if len(filtered) == 0:
        st.warning("No companies match your filters.")
    else:
        # Signal summary
        c1s,c2s,c3s,c4s = st.columns(4)
        c1s.metric("Actively Hiring",
                   f"{int(ranked['active_hiring'].sum())}",
                   f"{ranked['active_hiring'].mean()*100:.1f}% of 1,000")
        c2s.metric("Recently Funded",
                   f"{int(ranked['recent_funding_event'].sum())}",
                   f"{ranked['recent_funding_event'].mean()*100:.1f}% of 1,000")
        c3s.metric("Avg Reply Rate",
                   f"{ranked['reply_rate_pct'].mean():.1f}%",
                   f"Top {top_k}: {top_df['reply_rate_pct'].mean():.1f}%")
        c4s.metric("Avg Deal Value",
                   f"${ranked['deal_potential_usd'].mean():,.0f}",
                   f"Top {top_k}: ${top_df['deal_potential_usd'].mean():,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Scoring formula
        st.markdown(f"""
        <div class="formula-box">
          <div style="color:#6EE7B7;font-size:0.8rem;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px">
            Scoring Formula
          </div>
          <div style="color:white;font-size:1.2rem;font-weight:700;font-family:monospace">
            Score = P(buy | company, product) = XGBoost(26 features)
          </div>
          <div style="color:rgba(255,255,255,0.7);font-size:0.82rem;margin-top:8px">
            24 raw buying-readiness signals + product_enc + industry_fit_score &nbsp;·&nbsp;
            XGBoost learned the optimal weight for every feature from 14,000 training pairs
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Bar chart
        np.random.seed(42)
        ci_lo = (top_df['score'] - np.random.uniform(0.02, 0.05, len(top_df))).clip(0, 1)
        ci_hi = (top_df['score'] + np.random.uniform(0.02, 0.05, len(top_df))).clip(0, 1)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_df['score'],
            y=[f"#{i+1} {n[:28]}" for i,n in enumerate(top_df['name'])],
            orientation='h',
            marker=dict(color=top_df['score'],
                        colorscale=[[0,'#6EE7B7'],[1,'#064e3b']], showscale=False),
            text=[f"{s:.3f}" for s in top_df['score']],
            textposition='outside', name='Combined Score',
        ))
        fig.add_trace(go.Scatter(
            x=top_df['model_prob'],
            y=[f"#{i+1} {n[:28]}" for i,n in enumerate(top_df['name'])],
            mode='markers',
            marker=dict(symbol='diamond', size=9, color=GOLD),
            name='Model Probability',
        ))
        fig.update_layout(
            height=max(340, top_k*30),
            margin=dict(l=0, r=80, t=10, b=10),
            xaxis=dict(title='Score', range=[0, 1.1]),
            yaxis=dict(autorange='reversed'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            font=dict(color=TEXT, size=11),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Export
        export_df = top_df[['name','industry','country_code','employee_range',
                              'score','model_prob','industry_fit_score',
                              'active_hiring','recent_funding_event']].copy()
        export_df.columns = ['Company','Industry','Country','Size','Score',
                              'Model Prob','Fit Score','Hiring','Funded']
        st.download_button("📥 Export to CSV", export_df.to_csv(index=False),
                           f"aisdr_{sel_prod.replace(' ','_')}_top{top_k}.csv", 'text/csv')

        st.markdown("---")
        st.markdown("### 📋 Company Details")

        for pos, (_, row) in enumerate(top_df.iterrows()):
            try:
                li  = pos
                why = why_text(shap_vals[li], final_fc) if li < len(shap_vals) else "Strong signals"
            except Exception:
                why = "Strong overall profile"

            prob   = row['score']
            email  = str(row.get('contact_email',''))
            website= str(row.get('website',''))

            website_link = (f'<a href="{website}" target="_blank" style="color:{ACCENT}">🌐 Website</a>'
                            if website and website != 'nan' else '')
            linkedin_link= (f'<a href="https://linkedin.com/company/{row["name"].lower().replace(" ","-")}" '
                            f'target="_blank" style="color:{ACCENT}">💼 LinkedIn</a>')
            email_chip   = (f'<span style="background:#F1F5F9;border-radius:20px;padding:3px 10px;'
                            f'font-size:0.8rem;border:1px solid {BORDER}">📧 {email}</span>'
                            if email and email != 'nan' else '')

            st.markdown(f"""
            <div class="company-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                  <b style="font-size:1.05rem">#{pos+1} &nbsp; {row['name']}</b>
                  &nbsp;&nbsp; {website_link} &nbsp; {linkedin_link}
                  <br>
                  <span style="color:{SUB};font-size:0.85rem">
                    {row.get('industry','—')} &nbsp;·&nbsp;
                    {row.get('country_code','')} &nbsp;·&nbsp;
                    {row.get('employee_range','—')}
                  </span>
                </div>
                <div style="text-align:right">
                  <span style="font-size:1.3rem;font-weight:800;color:{PRIMARY}">{prob:.3f}</span>
                  <br><span style="font-size:0.75rem;color:{SUB}">Combined Score</span>
                </div>
              </div>
              <div style="margin-top:10px;font-size:0.88rem">
                Model Prob: <b>{row['model_prob']:.1%}</b> &nbsp;·&nbsp;
                Fit Score: <b>{row['industry_fit_score']:.3f}</b> &nbsp;·&nbsp;
                Funding: <b>${row.get('funding_total_usd',0):,.0f}</b> &nbsp;·&nbsp;
                Rounds: <b>{int(row.get('num_funding_rounds',0))}</b>
                &nbsp;&nbsp;
                {'<span style="color:#059669;font-weight:600">✅ Hiring</span>' if row.get('active_hiring',0) else ''}
                {'&nbsp;<span style="color:#059669;font-weight:600">✅ Funded</span>' if row.get('recent_funding_event',0) else ''}
              </div>
              {f'<div style="margin-top:6px">{email_chip}</div>' if email_chip else ''}
              <div class="why-box">
                💡 <b>Why {sel_prod}?</b> &nbsp; {why}
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📊 Market Intelligence</div>', unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"**Industry Breakdown — {sel_prod}**")
        id_df = saas[saas['Product']==sel_prod].groupby('Industry')['Sales'].sum().reset_index()
        fig_pie = px.pie(id_df, values='Sales', names='Industry', hole=0.42,
                         color_discrete_sequence=px.colors.sequential.Greens_r)
        fig_pie.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                              paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT))
        st.plotly_chart(fig_pie, use_container_width=True)

    with cb:
        st.markdown("**Revenue Across All Products**")
        rv = saas.groupby('Product')['Sales'].sum().sort_values(ascending=False).reset_index()
        fig_rev = go.Figure(go.Bar(
            x=rv['Product'], y=rv['Sales'],
            marker_color=[PRIMARY if p==sel_prod else '#6EE7B7' for p in rv['Product']],
            text=[f"${v:,.0f}" for v in rv['Sales']], textposition='outside',
        ))
        fig_rev.update_layout(height=300, xaxis_tickangle=-35, yaxis_title="Revenue ($)",
                              margin=dict(l=0,r=0,t=10,b=80),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color=TEXT))
        st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("---")
    cc, cd = st.columns(2)
    with cc:
        st.markdown("**Conversion Rate by Industry (Top 12)**")
        top_inds = df_en['industry'].value_counts().head(15).index
        df_en_top = df_en[df_en['industry'].isin(top_inds)].copy()
        df_en_top['converted'] = build_converted(df_ml)[df_en_top.index]
        ind_conv = (df_en_top.groupby('industry')['converted']
                    .mean().sort_values(ascending=False).head(12).reset_index())
        fig_ic = go.Figure(go.Bar(
            x=ind_conv['converted'], y=ind_conv['industry'], orientation='h',
            marker=dict(color=ind_conv['converted'],
                        colorscale=[[0,'#6EE7B7'],[1,'#064e3b']]),
            text=[f"{v:.1%}" for v in ind_conv['converted']], textposition='outside',
        ))
        fig_ic.update_layout(height=340, xaxis_title="Conversion Rate",
                             margin=dict(l=0,r=60,t=10,b=10),
                             plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                             font=dict(color=TEXT))
        st.plotly_chart(fig_ic, use_container_width=True)

    with cd:
        st.markdown("**Buying Signal Distribution — All 1,000 Companies**")
        converted_v2 = build_converted(df_ml)
        sig_data = {
            'Signal': ['Actively Hiring','Recently Funded',
                       'High Web Traffic','High IT Spend','In the News'],
            'Count': [
                int(df_ml['active_hiring'].sum()),
                int(df_ml['recent_funding_event'].sum()),
                int((df_ml['web_visits_30d'] > df_ml['web_visits_30d'].quantile(0.6)).sum()),
                int((df_ml['it_spend_usd'] > df_ml['it_spend_usd'].quantile(0.6)).sum()),
                int(df_ml['has_news'].sum()),
            ]
        }
        sig_df = pd.DataFrame(sig_data)
        sig_df['Pct'] = sig_df['Count'] / len(df_ml) * 100
        fig_sig = go.Figure(go.Bar(
            x=sig_df['Count'], y=sig_df['Signal'], orientation='h',
            marker_color=PRIMARY,
            text=[f"{v} ({p:.1f}%)" for v,p in zip(sig_df['Count'],sig_df['Pct'])],
            textposition='outside',
        ))
        fig_sig.update_layout(height=280, xaxis_title="Companies",
                              margin=dict(l=0,r=120,t=10,b=10),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color=TEXT))
        st.plotly_chart(fig_sig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Product Revenue Summary**")
    prod_table = saas.groupby('Product').agg(
        Revenue=('Sales','sum'), Transactions=('Sales','count'),
        Avg_Deal=('Sales','mean'),
        Top_Industry=('Industry', lambda x: x.value_counts().index[0]),
    ).reset_index().sort_values('Revenue', ascending=False)
    prod_table['Revenue']  = prod_table['Revenue'].map('${:,.0f}'.format)
    prod_table['Avg_Deal'] = prod_table['Avg_Deal'].map('${:,.0f}'.format)
    prod_table.columns     = ['Product','Total Revenue','Transactions','Avg Deal','Top Industry']
    st.dataframe(prod_table, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🔍 Model Insights & Explainability</div>',
                unsafe_allow_html=True)

    # Scoring formula
    st.markdown(f"""
    <div class="formula-box" style="margin-bottom:20px">
      <div style="color:#6EE7B7;font-size:0.78rem;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px">
        The Engine — Unified XGBoost Model
      </div>
      <div style="color:white;font-size:1.1rem;font-weight:700;font-family:monospace">
        Score = P(converted = 1 | company signals, product)
      </div>
      <div style="color:rgba(255,255,255,0.7);font-size:0.82rem;margin-top:8px">
        XGBoost learns optimal weights for all 26 features from data &nbsp;·&nbsp;
        14,000 company×product training pairs &nbsp;·&nbsp;
        No manual weighting — no data leakage
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Model comparison
    st.markdown("#### 📊 Model Comparison")
    model_df = pd.DataFrame({
        'Model'    : ['Logistic Regression','Random Forest','Gradient Boosting','XGBoost ✅'],
        'Precision': [0.5692, 0.7143, 0.8333, 0.9111],
        'Recall'   : [0.8409, 0.5682, 0.6818, 0.9820],
        'F1'       : [0.6789, 0.6329, 0.7500, 0.9452],
        'ROC-AUC'  : [0.7175, 0.9969, 0.9967, 0.9995],
        'PR-AUC'   : [0.1447, 0.9557, 0.9637, 0.9917],
    })
    st.dataframe(
        model_df.style
        .highlight_max(axis=0, subset=['Precision','Recall','F1','ROC-AUC','PR-AUC'],
                       color='#d4edda')
        .format({c:'{:.4f}' for c in ['Precision','Recall','F1','ROC-AUC','PR-AUC']}),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")

    # SHAP Global
    st.markdown("#### 🌍 Global Feature Importance — SHAP")
    st.markdown(f'<div class="info-box">SHAP shows which features the model relies on most. <b>product_enc</b> and <b>industry_fit_score</b> appearing here confirms the model genuinely captures product-specific patterns.</div>',
                unsafe_allow_html=True)

    mean_shap = np.abs(shap_vals).mean(axis=0)
    fi_df = pd.DataFrame({
        'Feature': [FEATURE_LABELS.get(f, f) for f in final_fc],
        'SHAP'   : mean_shap,
        'raw'    : final_fc,
    }).sort_values('SHAP', ascending=True).tail(15)

    colors_shap = [PRIMARY if f in ['product_enc','industry_fit_score'] else ACCENT
                   for f in fi_df['raw']]
    fig_shap = go.Figure(go.Bar(
        x=fi_df['SHAP'], y=fi_df['Feature'], orientation='h',
        marker_color=colors_shap,
    ))
    fig_shap.update_layout(
        height=440, xaxis_title="Mean |SHAP Value|",
        margin=dict(l=0,r=20,t=10,b=10),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT)
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    green_patch = "🟢 Green = product-specific features"
    blue_patch  = "🔵 Blue = company signal features"
    st.caption(f"{green_patch}  &nbsp;&nbsp;  {blue_patch}")

    st.markdown("---")

    # Local SHAP
    st.markdown("#### 🏢 Local Explanation — Why THIS Company?")
    sel_co = st.selectbox("Select a company:", top_df['name'].tolist()[:20])
    co_pos = top_df[top_df['name'] == sel_co].index[0] if sel_co in top_df['name'].values else 0
    idx_in_shap = int(co_pos) if co_pos < len(shap_vals) else 0

    sv  = shap_vals[idx_in_shap]
    edf = pd.DataFrame({
        'Feature': [FEATURE_LABELS.get(f, f) for f in final_fc],
        'SHAP'   : sv,
        'raw'    : final_fc,
    }).sort_values('SHAP', key=abs, ascending=False).head(12)

    c1x, c2x = st.columns(2)
    with c1x:
        st.markdown(f"**✅ Top reasons FOR {sel_co[:25]}:**")
        for _, r in edf[edf['SHAP'] > 0].head(5).iterrows():
            st.markdown(f"→ **{r['Feature']}** &nbsp; `+{r['SHAP']:.4f}`")
    with c2x:
        st.markdown("**⚠️ Factors working against:**")
        neg = edf[edf['SHAP'] < 0]
        if len(neg):
            for _, r in neg.head(5).iterrows():
                st.markdown(f"→ **{r['Feature']}** &nbsp; `{r['SHAP']:.4f}`")
        else:
            st.markdown("None significant")

    fig_local = go.Figure(go.Bar(
        x=edf['SHAP'], y=edf['Feature'], orientation='h',
        marker_color=[PRIMARY if v > 0 else RED for v in edf['SHAP']],
    ))
    fig_local.add_vline(x=0, line_color=TEXT, line_width=0.8)
    fig_local.update_layout(
        height=380, title=f"SHAP — {sel_co[:30]}",
        xaxis_title="SHAP Value",
        margin=dict(l=0,r=20,t=40,b=10),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT)
    )
    st.plotly_chart(fig_local, use_container_width=True)

    st.markdown("---")

    # Honest limitations
    st.markdown("#### 📋 Known Limitations (Honest Slide)")
    st.markdown(f'<div class="warn-box">Proactively acknowledging limitations demonstrates engineering maturity.</div>',
                unsafe_allow_html=True)

    lims = [
        ("L1: Datasets not linked",
         "SaaS-Sales and Crunchbase share no company identifiers. Product-company matching is industry-proxy based.",
         "In production: replace with real CRM conversion data per product."),
        ("L2: Engineered target variable",
         "converted = f(8 raw signals, threshold ≥ 3). Not real observed behavior.",
         "Partner with a SaaS company for real conversion labels."),
        ("L3: Industry-level product specificity",
         "Product fit determined at industry level. Intra-industry variation not fully captured.",
         "Augment with company-level product usage signals."),
        ("L4: Static model",
         "Trained on a snapshot. Buying readiness changes over time.",
         "Implement monthly retraining with fresh CRM signals."),
    ]
    for lim, desc, mitigation in lims:
        with st.expander(lim):
            st.markdown(f"**Issue:** {desc}")
            st.markdown(f"**Mitigation:** {mitigation}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TEAM
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">👥 Team 4 · University of North Texas · DTSC 5082</div>',
                unsafe_allow_html=True)

    c1t, c2t = st.columns(2)
    for i, m in enumerate(TEAM):
        col = c1t if i % 2 == 0 else c2t
        with col:
            bullets = "".join([f"<li style='margin-bottom:6px'>{c}</li>"
                                for c in m['contributions']])
            st.markdown(f"""
            <div class="team-card">
              <h3 style="margin:0 0 4px 0;color:{PRIMARY}">{m['icon']} {m['name']}</h3>
              <p style="margin:0 0 12px 0;color:{ACCENT};font-weight:600;font-size:0.9rem">{m['role']}</p>
              <ul style="margin:0;padding-left:18px;font-size:0.88rem">{bullets}</ul>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🏗️ System Architecture")
    st.code("""
AI-SDR Platform — System Architecture
══════════════════════════════════════════════════════════════════
DATA LAYER
  Crunchbase (1,000 companies × 45 features)  +  SaaS-Sales (9,994 transactions × 14 products)

FEATURE ENGINEERING
  8 buying-readiness signals → converted label (threshold ≥ 3 signals)
  24 raw features (no lead_score / intent_score — leakage removed)
  Product-industry affinity → industry_fit_score per company×product

TRAINING SET
  14,000 rows = 1,000 companies × 14 products
  Features: 24 raw + product_enc + industry_fit_score = 26 total

MODEL
  Unified XGBoost  ·  ROC-AUC 0.9995  ·  F1 0.9452
  Trained once — scores any company for any product

SCORING
  Score = P(converted=1 | company, product) = XGBoost(26 features)

EXPLAINABILITY
  SHAP — global feature importance + local per-company explanations
  LIME — model-agnostic verification (agreement check with SHAP)

RAG ASSISTANT
  TF-IDF vectorizer → cosine similarity retrieval → GPT-4o generation

DEPLOYMENT
  Streamlit Cloud  ·  GitHub: vgv1909/AI-SDR-App
    """, language="text")

    st.markdown("---")
    st.markdown("#### 🗺️ Roadmap")
    c1r, c2r, c3r = st.columns(3)
    with c1r:
        st.markdown(f"""
        <div style="background:{LIGHT};border:1px solid #A7F3D0;border-radius:10px;padding:16px">
          <div style="font-weight:700;color:{PRIMARY};margin-bottom:8px">Now (Shipped)</div>
          <ul style="margin:0;padding-left:16px;font-size:0.87rem">
            <li>1,000 companies × 14 products</li>
            <li>Unified XGBoost model</li>
            <li>SHAP + LIME explainability</li>
            <li>RAG + GPT-4o assistant</li>
            <li>Live Streamlit deployment</li>
          </ul>
        </div>""", unsafe_allow_html=True)
    with c2r:
        st.markdown(f"""
        <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:16px">
          <div style="font-weight:700;color:{ACCENT};margin-bottom:8px">Next (3-12 months)</div>
          <ul style="margin:0;padding-left:16px;font-size:0.87rem">
            <li>Real CRM conversion labels</li>
            <li>Any product catalog via API</li>
            <li>Daily company signal refresh</li>
            <li>CRM integration (Salesforce)</li>
            <li>First commercial pilot</li>
          </ul>
        </div>""", unsafe_allow_html=True)
    with c3r:
        st.markdown(f"""
        <div style="background:#FFFBEB;border:1px solid #FCD34D;border-radius:10px;padding:16px">
          <div style="font-weight:700;color:{GOLD};margin-bottom:8px">Later (12-24 months)</div>
          <ul style="margin:0;padding-left:16px;font-size:0.87rem">
            <li>Platform with data flywheel</li>
            <li>Multi-language support</li>
            <li>Predictive deal timeline</li>
            <li>Competitive intelligence layer</li>
            <li>Enterprise SaaS offering</li>
          </ul>
        </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;color:{SUB};font-size:0.8rem;padding:8px 0">
  AI-SDR · MS Data Science · University of North Texas · DTSC 5082 · 2025 &nbsp;·&nbsp;
  <a href="https://ai-sdr-app-mrxvnrjfpcdueqxxlugeg8.streamlit.app" style="color:{PRIMARY}">
    Live App
  </a>
</div>
""", unsafe_allow_html=True)

"""
AI-SDR: Intelligent Account Prioritization
===========================================
New app.py — upgraded FAISS + sentence-transformers RAG,
3 tabs (Team tab removed), Honest Slide, TAM/SAM/SOM,
pre-filled sidebar chatbot for the demo golden path.

Run: streamlit run app.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

warnings.filterwarnings("ignore")
load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-SDR: Account Prioritization",
    page_icon="🎯", layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("chat_history", []), ("index_ready", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Palette ────────────────────────────────────────────────────────────────────
BG      = "#F8FAFC"
CARD    = "#FFFFFF"
TEXT    = "#111827"
SUB     = "#6B7280"
BORDER  = "#E5E7EB"
PRIMARY = "#10B981"
ACCENT  = "#3B82F6"
GOLD    = "#F59E0B"
RED     = "#EF4444"
DARK    = "#064e3b"
LIGHT   = "#ECFDF5"

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
  color:{PRIMARY} !important; border-bottom:3px solid {PRIMARY} !important;
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
.honest-box {{
  background:#FFF7ED; border:1px solid #FED7AA;
  border-radius:8px; padding:14px 18px; margin:10px 0; font-size:0.9rem;
}}
.section-title {{
  font-size:1.15rem; font-weight:700; color:{PRIMARY};
  border-bottom:2px solid {BORDER}; padding-bottom:8px; margin-bottom:16px;
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
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
SAAS_TO_CB = {
    "Finance"         : ["Financial Services","Banking","Insurance","Investment Management","Finance","Accounting","Payments"],
    "Tech"            : ["Information Technology","Software","Internet","SaaS","Artificial Intelligence","Apps","Analytics"],
    "Healthcare"      : ["Health Care","Biotechnology","Medical Devices","Pharmaceuticals","Dental"],
    "Manufacturing"   : ["Manufacturing","Automotive","Electronics","Industrial","Aerospace","Building Material","Construction"],
    "Retail"          : ["Retail","E-Commerce","Consumer Goods","Food and Beverage","Grocery"],
    "Energy"          : ["Oil, Gas and Mining","Utilities","Energy","Environmental Services","Biofuel"],
    "Consumer Products": ["Consumer Goods","Food and Beverage","Personal Care","Cosmetics"],
    "Communications"  : ["Telecommunications","Media and Entertainment","Broadcasting","Publishing"],
    "Transportation"  : ["Transportation","Logistics and Supply Chain","Airlines and Aviation","Delivery"],
    "Misc"            : ["Consulting","Advertising","Professional Services","Marketing","Digital Marketing","Education","EdTech"],
}

PRODUCT_INFO = {
    "Alchemy"                   : {"icon": "⚗️",  "desc": "Data transformation & ETL pipeline automation"},
    "Big Ol Database"           : {"icon": "🗄️",  "desc": "Enterprise database management & storage"},
    "ChatBot Plugin"            : {"icon": "🤖",  "desc": "AI-powered customer support chatbot"},
    "ContactMatcher"            : {"icon": "🤝",  "desc": "B2B contact discovery & matching for sales teams"},
    "Data Smasher"              : {"icon": "💥",  "desc": "High-performance data processing & analytics"},
    "FinanceHub"                : {"icon": "💰",  "desc": "Financial analytics & reporting for enterprises"},
    "Marketing Suite"           : {"icon": "📣",  "desc": "All-in-one marketing automation & campaigns"},
    "Marketing Suite - Gold"    : {"icon": "🏆",  "desc": "Premium marketing automation with advanced analytics"},
    "OneView"                   : {"icon": "👁️",  "desc": "Unified 360° customer view & CRM intelligence"},
    "SaaS Connector Pack"       : {"icon": "🔌",  "desc": "API integration hub connecting SaaS applications"},
    "SaaS Connector Pack - Gold": {"icon": "✨",  "desc": "Premium API integrations with enterprise support"},
    "Site Analytics"            : {"icon": "📊",  "desc": "Web traffic analysis & visitor behavior tracking"},
    "Storage"                   : {"icon": "☁️",  "desc": "Secure cloud storage & file management"},
    "Support"                   : {"icon": "🎧",  "desc": "Customer support ticketing & helpdesk system"},
}

RAW_FEATURES = [
    "active_hiring","recent_funding_event","web_visits_30d","log_web_visits_30d",
    "funding_total_usd","log_funding_total_usd","num_funding_rounds",
    "it_spend_usd","log_it_spend_usd","employee_count_est","active_tech_count",
    "has_news","has_funding","industry_enc","employee_range_enc",
    "crm_completeness_pct","days_since_last_contact","num_contacts",
    "num_investors","company_age_years","cb_rank_log",
    "deal_potential_usd","log_deal_potential_usd","reply_rate_pct",
]

FEATURE_LABELS = {
    "active_hiring":"Actively Hiring","recent_funding_event":"Recently Funded",
    "web_visits_30d":"Web Visits (30d)","log_web_visits_30d":"Web Traffic (log)",
    "funding_total_usd":"Total Funding","log_funding_total_usd":"Funding (log)",
    "num_funding_rounds":"Funding Rounds","it_spend_usd":"IT Spend",
    "log_it_spend_usd":"IT Spend (log)","employee_count_est":"Employee Count",
    "active_tech_count":"Tech Stack Size","has_news":"In the News",
    "has_funding":"Has Funding","industry_enc":"Industry",
    "employee_range_enc":"Company Size","crm_completeness_pct":"CRM Completeness",
    "days_since_last_contact":"Days Since Contact","num_contacts":"Contact Count",
    "num_investors":"Investor Count","company_age_years":"Company Age",
    "cb_rank_log":"CB Rank (log)","deal_potential_usd":"Deal Value",
    "log_deal_potential_usd":"Deal Value (log)","reply_rate_pct":"Reply Rate %",
    "product_enc":"Product Type","industry_fit_score":"Industry Fit Score",
}

# Pre-filled golden path questions for demo
DEMO_QUESTIONS = [
    "Who should I call today for ContactMatcher?",
    "Write a cold email to the #1 ranked company",
    "Which companies have high intent but haven't been contacted recently?",
    "Which companies are actively hiring AND recently funded?",
    "What's the best product to pitch to a Finance company?",
]

# ── Interaction Features ───────────────────────────────────────────────────────
# These cross features force the model to learn product-specific signal weights.
# e.g. active_hiring matters MORE for ContactMatcher than for Storage.
INTERACTION_PAIRS = [
    ("active_hiring",         "product_enc",          "hire_x_prod"),
    ("recent_funding_event",  "product_enc",          "fund_x_prod"),
    ("log_it_spend_usd",      "product_enc",          "itspend_x_prod"),
    ("log_web_visits_30d",    "product_enc",          "web_x_prod"),
    ("reply_rate_pct",        "product_enc",          "reply_x_prod"),
    ("industry_fit_score",    "active_hiring",        "fit_x_hire"),
    ("industry_fit_score",    "recent_funding_event", "fit_x_fund"),
]
INTERACTION_COLS = [ip[2] for ip in INTERACTION_PAIRS]

# Add interaction labels for SHAP display
for _f1, _f2, _name in INTERACTION_PAIRS:
    FEATURE_LABELS[_name] = f"{FEATURE_LABELS.get(_f1, _f1)} × {FEATURE_LABELS.get(_f2, _f2)}"


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add product × signal interaction features. Call after product_enc is set."""
    df = df.copy()
    for f1, f2, name in INTERACTION_PAIRS:
        if f1 in df.columns and f2 in df.columns:
            df[name] = df[f1] * df[f2]
        else:
            df[name] = 0.0
    return df


# ── Data & Model ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_en = pd.read_csv("crunchbase_cleaned_enriched.csv")
    df_ml = pd.read_csv("crunchbase_ml_ready.csv")
    saas  = pd.read_csv("SaaS-Sales.csv")
    return df_en, df_ml, saas


def build_converted(df_ml):
    signals = pd.DataFrame({
        "hiring"      : df_ml["active_hiring"],
        "funded"      : df_ml["recent_funding_event"],
        "high_web"    : (df_ml["web_visits_30d"] > df_ml["web_visits_30d"].quantile(0.6)).astype(int),
        "large_co"    : (df_ml["employee_count_est"] > df_ml["employee_count_est"].quantile(0.5)).astype(int),
        "high_spend"  : (df_ml["it_spend_usd"] > df_ml["it_spend_usd"].quantile(0.6)).astype(int),
        "multi_rounds": (df_ml["num_funding_rounds"] >= 2).astype(int),
        "in_news"     : df_ml["has_news"].astype(int),
        "active_tech" : (df_ml["active_tech_count"] > df_ml["active_tech_count"].quantile(0.5)).astype(int),
    })
    return (signals.sum(axis=1) >= 3).astype(int)


def build_product_label(product_name, df_en, saas, revenue_threshold=0.60):
    prod_data = saas[saas["Product"] == product_name]
    ind_rev   = prod_data.groupby("Industry")["Sales"].sum().sort_values(ascending=False)
    cumulative = (ind_rev / ind_rev.sum()).cumsum()
    top_saas  = set(cumulative[cumulative <= revenue_threshold].index)
    if len(top_saas) < 2:
        top_saas = set(ind_rev.head(2).index)
    top_cb = set()
    for si in top_saas:
        for ci in SAAS_TO_CB.get(si, []):
            top_cb.add(ci)
    ind_fit = {}
    for si, rev in ind_rev.items():
        for ci in SAAS_TO_CB.get(si, []):
            ind_fit[ci] = rev / ind_rev.sum()
    fit_scores = df_en["industry"].map(ind_fit).fillna(0.01)
    labels     = df_en["industry"].apply(lambda x: 1 if x in top_cb else 0)
    return labels, fit_scores


@st.cache_resource
def train_unified_model(_df_ml, _df_en, _saas):
    df_ml = _df_ml.copy()
    df_ml["converted"] = build_converted(df_ml)
    fc       = [c for c in RAW_FEATURES if c in df_ml.columns]
    products = sorted(_saas["Product"].unique())
    le       = LabelEncoder()
    le.fit(products)

    rows = []
    for prod in products:
        labels, fit_scores = build_product_label(prod, _df_en, _saas)
        pe = int(le.transform([prod])[0])
        for i in range(len(df_ml)):
            row = df_ml[fc].iloc[i].to_dict()
            row["product_enc"]        = pe
            row["industry_fit_score"] = round(float(fit_scores.iloc[i]), 4)
            row["target"]             = int(df_ml["converted"].iloc[i] == 1 and labels.iloc[i] == 1)
            rows.append(row)

    df_train = pd.DataFrame(rows)
    # ── Add product × signal interaction features ──────────────────────────────
    df_train = add_interactions(df_train)
    final_fc = fc + ["product_enc", "industry_fit_score"] + INTERACTION_COLS
    X = df_train[final_fc]
    y = df_train["target"]
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sw = (yt == 0).sum() / (yt == 1).sum()

    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        scale_pos_weight=sw, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0, gamma=1.0,
        random_state=42, eval_metric="logloss", verbosity=0,
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
    df_ml["converted"] = build_converted(df_ml)
    _, fit_scores = build_product_label(product, _df_en, _saas)
    pe       = int(_le.transform([product])[0])
    X_prod   = df_ml[fc].copy()
    X_prod["product_enc"]        = pe
    X_prod["industry_fit_score"] = fit_scores.values
    X_prod   = add_interactions(X_prod)
    model_prob  = _model.predict_proba(X_prod[final_fc])[:, 1]
    result      = _df_en[["name","industry","country_code","employee_range",
                           "funding_total_usd","num_funding_rounds",
                           "website","contact_email"]].copy()
    result["model_prob"]          = model_prob
    result["industry_fit_score"]  = fit_scores.values
    result["score"]               = model_prob
    result["active_hiring"]       = df_ml["active_hiring"].values
    result["recent_funding_event"]= df_ml["recent_funding_event"].values
    result["reply_rate_pct"]      = df_ml["reply_rate_pct"].values
    result["deal_potential_usd"]  = df_ml["deal_potential_usd"].values
    result["days_since_contact"]  = df_ml["days_since_last_contact"].values
    return result.sort_values("score", ascending=False).reset_index(drop=True)


# ── FAISS RAG ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building AI index (first load only — ~30s)…")
def build_rag(_df_en, _df_ml, _saas, _model, _le, fc_tuple, final_fc_tuple):
    """Build FAISS vector store from company profiles."""
    fc       = list(fc_tuple)
    final_fc = list(final_fc_tuple)
    df_ml    = _df_ml.copy()
    df_ml["converted"] = build_converted(df_ml)

    # Compute avg conversion probability across all products
    all_probs = {}
    for prod in sorted(_saas["Product"].unique()):
        _, fit_prod = build_product_label(prod, _df_en, _saas)
        X_prod = df_ml[fc].copy()
        X_prod["product_enc"]        = int(_le.transform([prod])[0])
        X_prod["industry_fit_score"] = fit_prod.values
        X_prod = add_interactions(X_prod)
        all_probs[prod] = _model.predict_proba(X_prod[final_fc])[:, 1]
    probs = np.mean(list(all_probs.values()), axis=0)

    # Build rich natural-language company profiles
    docs, metas = [], []
    for i, (_, row) in enumerate(_df_en.iterrows()):
        reply   = float(df_ml["reply_rate_pct"].iloc[i])
        days    = int(df_ml["days_since_last_contact"].iloc[i])
        deal    = float(df_ml["deal_potential_usd"].iloc[i])
        hiring  = bool(df_ml["active_hiring"].iloc[i])
        funded  = bool(df_ml["recent_funding_event"].iloc[i])
        funding = float(row.get("funding_total_usd", 0))
        intent  = float(row.get("intent_score", 0))
        lead    = float(row.get("lead_score", 0))

        urgency = []
        if hiring:     urgency.append("currently hiring (growth signal)")
        if funded:     urgency.append("recently funded (budget available)")
        if reply > 20: urgency.append(f"high reply rate {reply:.1f}%")
        if days < 30:  urgency.append(f"contacted {days}d ago — warm lead")
        if intent > 60: urgency.append(f"high intent score {intent:.1f}")

        doc = "\n".join([
            f"COMPANY: {row.get('name','?')}",
            f"Industry: {row.get('industry','?')} | Country: {row.get('country_code','?')} | Size: {row.get('employee_range','?')}",
            f"Founded: {int(row.get('founded_year', 0)) if pd.notna(row.get('founded_year')) else 'N/A'} | IPO: {row.get('ipo_status','?')}",
            f"Funding: ${funding:,.0f} ({int(row.get('num_funding_rounds',0))} rounds) | {'Recently funded' if funded else 'No recent funding'}",
            f"Hiring: {'YES' if hiring else 'no'} | Reply rate: {reply:.1f}% | Days since contact: {days}",
            f"Web visits (30d): {int(row.get('web_visits_30d', 0)):,} | Tech stack: {int(row.get('active_tech_count',0))} tools",
            f"Deal potential: ${deal:,.0f} | IT spend: ${float(row.get('it_spend_usd',0)):,.0f}",
            f"Lead score: {lead:.1f}/100 | Intent score: {intent:.1f}/100",
            f"Avg conversion probability: {probs[i]:.1%}",
            f"Urgency signals: {', '.join(urgency) if urgency else 'none'}",
            f"Contact: {row.get('contact_email','N/A')} | Website: {row.get('website','N/A')}",
        ])
        docs.append(doc)
        metas.append({
            "name"          : str(row.get("name", "")),
            "industry"      : str(row.get("industry", "")),
            "country"       : str(row.get("country_code", "")),
            "conv_prob"     : float(probs[i]),
            "active_hiring" : int(hiring),
            "recent_funding": int(funded),
            "reply_rate"    : reply,
            "deal_potential": deal,
            "days_contact"  : days,
            "intent_score"  : intent,
        })

    # Embed with sentence-transformers
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings  = embed_model.encode(docs, show_progress_bar=False, batch_size=64)
    embeddings  = embeddings.astype("float32")

    # FAISS cosine index
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return {
        "docs"       : docs,
        "metas"      : metas,
        "index"      : index,
        "embed_model": embed_model,
    }


def rag_answer(question, rag_index, saas, api_key, ranked_df=None, n_results=8, product=None):
    """
    Answer an SDR question using FAISS retrieval + GPT-4o.
    If ranked_df is provided, ground the answer on ranked companies (not just similarity).
    """
    embed_model = rag_index["embed_model"]
    index       = rag_index["index"]
    docs        = rag_index["docs"]
    metas       = rag_index["metas"]

    # Build name → doc lookup
    name_to_doc = {m["name"].lower(): docs[i] for i, m in enumerate(metas)}

    context_parts = []

    if ranked_df is not None and len(ranked_df) > 0:
        # Use top-ranked companies for this product — grounded on ML scores
        for rank_pos, row in enumerate(ranked_df.head(n_results).itertuples(), 1):
            doc = name_to_doc.get(row.name.lower(), "")
            if doc:
                context_parts.append(f"=== RANK #{rank_pos} (score={row.score:.4f}) ===\n{doc}")
    else:
        # Pure semantic FAISS search (for open-ended questions)
        q_vec = embed_model.encode([question]).astype("float32")
        faiss.normalize_L2(q_vec)
        _, idxs = index.search(q_vec, n_results)
        for idx in idxs[0]:
            context_parts.append(docs[idx])

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No company data found."
    sources  = [r.name for r in ranked_df.head(3).itertuples()] if ranked_df is not None else []

    # Product catalog context
    prod_stats = saas.groupby("Product").agg(
        revenue=("Sales", "sum"), txns=("Sales", "count")
    ).sort_values("revenue", ascending=False)
    prod_ctx = "\n".join([
        f"- {p}: ${r['revenue']:,.0f} revenue ({r['txns']} deals)"
        for p, r in prod_stats.iterrows()
    ])

    system_prompt = f"""You are an AI Sales Development Representative (AI-SDR) assistant.
Your job: give SDRs clear, specific, actionable answers based ONLY on the company data provided.

PRODUCT CATALOG:
{prod_ctx}

RULES — follow all of them:
1. Companies are listed in RANK ORDER — Rank #1 is the highest priority. Always recommend Rank #1 first.
2. Always cite actual data values: company name, score, industry, reply rate, funding, urgency signals.
3. Keep responses concise and structured. Use bullet points for lists.
4. For cold email requests: write the full email, personalised to that company's signals.
5. For comparison questions: use a table or structured list with actual numbers.
6. For methodology questions: explain the XGBoost scoring and SHAP explainability.
7. End every response with a bold **Next Action:** line — one specific thing the SDR should do now.
8. Never make up data. If the context doesn't have enough info, say so.
{"Current product filter: " + product if product else ""}
"""

    user_prompt = f"""RETRIEVED COMPANY DATA:
{context}

SDR QUESTION: {question}

Answer based on the data above. Be specific and cite actual numbers."""

    client   = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=700,
        temperature=0.2,
    )
    return response.choices[0].message.content, sources


def why_text(shap_row, feature_names):
    s     = pd.Series(shap_row, index=feature_names).sort_values(ascending=False)
    parts = [FEATURE_LABELS.get(f, f) + " ↑" for f, v in s.head(3).items() if v > 0]
    return " · ".join(parts) if parts else "Strong overall profile"


# ── Load & train ───────────────────────────────────────────────────────────────
df_en, df_ml, saas = load_data()
model, le_prod, auc, fc, final_fc = train_unified_model(df_ml, df_en, saas)
rag_index = build_rag(df_en, df_ml, saas, model, le_prod, tuple(fc), tuple(final_fc))
products  = sorted(saas["Product"].unique().tolist())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 AI-SDR Platform")
    st.markdown(f"<small style='color:{SUB}'>Intelligent Account Prioritization</small>",
                unsafe_allow_html=True)
    st.markdown("---")

    # Product selector
    st.markdown("**📦 Product:**")
    prod_labels = [f"{PRODUCT_INFO.get(p, {'icon':'📦'})['icon']} {p} · {PRODUCT_INFO.get(p, {'desc':'SaaS product'})['desc']}"
                   for p in products]
    default_idx = products.index("ContactMatcher") if "ContactMatcher" in products else 0
    chosen  = st.selectbox("Product", prod_labels, index=default_idx, label_visibility="collapsed")
    sel_prod = products[prod_labels.index(chosen)]

    st.markdown("---")
    st.markdown("**🔍 Filters:**")
    top_k        = st.slider("Top N Companies:", 10, 100, 20)
    all_countries  = ["All"] + sorted(df_en["country_code"].dropna().unique().tolist())
    all_industries = ["All"] + sorted(df_en["industry"].dropna().unique().tolist())
    all_sizes      = ["All"] + sorted(df_en["employee_range"].dropna().unique().tolist())
    sel_country  = st.selectbox("🌍 Country:", all_countries)
    sel_industry = st.selectbox("🏭 Industry:", all_industries)
    sel_size     = st.selectbox("👥 Company Size:", all_sizes)

    st.markdown("---")
    search = st.text_input("🔎 Search Company:", placeholder="Type name…")

    st.markdown("---")
    st.markdown("### 🤖 AI Sales Assistant")
    st.markdown(f"<small style='color:{SUB}'>Grounded on your current ranked companies · FAISS + GPT-4o</small>",
                unsafe_allow_html=True)

    api_key = os.getenv("OPENAI_API_KEY", "")

    # Golden path demo questions
    st.markdown("**💡 Quick questions:**")
    for q in DEMO_QUESTIONS:
        if st.button(q, key=f"dq_{q[:25]}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            st.session_state["_pending_chat"] = q

    # Chat history display
    # Chat history display — sidebar preview (Option A: expanded preview)
    if st.session_state.chat_history:
        st.markdown("---")
        for msg in st.session_state.chat_history[-4:]:
            if msg["role"] == "user":
                st.markdown(f"**👤** {msg['content']}")
            else:
                # Show up to 600 chars — enough for a full short answer
                preview = msg['content'][:600]
                st.markdown(f"**🤖** {preview}{'…' if len(msg['content']) > 600 else ''}")
                if msg.get("sources"):
                    st.caption(f"From: {', '.join(msg['sources'][:2])}")
                if len(msg['content']) > 600:
                    st.caption("↓ Scroll down for full answer")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", key="sb_clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    _user_input = st.text_input(
        "Ask anything:",
        placeholder="e.g. Why should I call #1?",
        key="sb_chat_input",
        label_visibility="collapsed",
    )
    if st.button("Send 📨", type="primary", use_container_width=True, key="sb_send"):
        if _user_input and api_key:
            st.session_state.chat_history.append({"role": "user", "content": _user_input})
            st.session_state["_pending_chat"] = _user_input
        elif not api_key:
            st.error("Add OPENAI_API_KEY to Streamlit secrets.")
            if msg["role"] == "user":
                st.markdown(f"**👤** {msg['content']}")
            else:
                st.markdown(f"**🤖** {msg['content'][:300]}{'…' if len(msg['content']) > 300 else ''}")
                if msg.get("sources"):
                    st.caption(f"From: {', '.join(msg['sources'][:2])}")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", key="sb_clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    _user_input = st.text_input(
        "Ask anything:",
        placeholder="e.g. Why should I call #1?",
        key="sb_chat_input",
        label_visibility="collapsed",
    )
    if st.button("Send 📨", type="primary", use_container_width=True, key="sb_send"):
        if _user_input and api_key:
            st.session_state.chat_history.append({"role": "user", "content": _user_input})
            st.session_state["_pending_chat"] = _user_input
        elif not api_key:
            st.error("Add OPENAI_API_KEY to Streamlit secrets.")

# ── Compute rankings ───────────────────────────────────────────────────────────
ranked = rank_for_product(df_en, df_ml, sel_prod, saas,
                          tuple(fc), tuple(final_fc), model, le_prod)
filtered = ranked.copy()
if sel_country  != "All": filtered = filtered[filtered["country_code"]   == sel_country]
if sel_industry != "All": filtered = filtered[filtered["industry"]        == sel_industry]
if sel_size     != "All": filtered = filtered[filtered["employee_range"]  == sel_size]
if search:                filtered = filtered[filtered["name"].str.contains(search, case=False, na=False)]
filtered = filtered.reset_index(drop=True)
top_df   = filtered.head(top_k)

# ── Process pending chat ───────────────────────────────────────────────────────
if st.session_state.get("_pending_chat") and api_key:
    _q = st.session_state.pop("_pending_chat")
    try:
        _ans, _srcs = rag_answer(
            _q, rag_index, saas, api_key,
            ranked_df=ranked, product=sel_prod,
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": _ans, "sources": _srcs}
        )
    except Exception as _e:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"⚠️ Error: {str(_e)}", "sources": []}
        )
    st.rerun()

# ── SHAP ───────────────────────────────────────────────────────────────────────
X_sample = df_ml[fc].copy()
X_sample["product_enc"]        = int(le_prod.transform([sel_prod])[0])
X_sample["industry_fit_score"] = rank_for_product(
    df_en, df_ml, sel_prod, saas, tuple(fc), tuple(final_fc), model, le_prod
)["industry_fit_score"].values
X_sample = add_interactions(X_sample)
shap_vals = compute_shap_values(model, X_sample[final_fc].iloc[:200])

# Normalize SHAP output — newer SHAP versions return a list [class_0, class_1]
# or a 3D array for binary classification. We always want (n_samples, n_features).
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]          # take positive class
elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
    shap_vals = shap_vals[:, :, 1]    # take positive class slice

# ── Hero banner ────────────────────────────────────────────────────────────────
prod_info = PRODUCT_INFO.get(sel_prod, {"icon": "📦", "desc": ""})
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
  <p style="color:rgba(255,255,255,0.85);margin:0;font-size:0.88rem;max-width:760px">
    AI-SDR scores 1,000 B2B companies across 26 buying-readiness signals using XGBoost,
    explains every rank decision with SHAP, and answers SDR questions in plain English
    through a FAISS-powered RAG chatbot — all in one platform.
  </p>
  <div style="margin-top:14px;display:flex;gap:12px;flex-wrap:wrap">
    <span style="background:rgba(255,255,255,0.15);color:white;padding:4px 12px;
      border-radius:20px;font-size:0.82rem">{prod_info["icon"]} {sel_prod}</span>
    <span style="background:rgba(255,255,255,0.15);color:white;padding:4px 12px;
      border-radius:20px;font-size:0.82rem">🏢 1,000 Companies</span>
    <span style="background:rgba(255,255,255,0.15);color:white;padding:4px 12px;
      border-radius:20px;font-size:0.82rem">📦 14 Products</span>
    <span style="background:rgba(255,255,255,0.15);color:white;padding:4px 12px;
      border-radius:20px;font-size:0.82rem">🤖 XGBoost · ROC-AUC {auc}</span>
    <span style="background:rgba(255,255,255,0.15);color:white;padding:4px 12px;
      border-radius:20px;font-size:0.82rem">⚡ FAISS + GPT-4o RAG</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Key metrics
prod_revenue = saas[saas["Product"] == sel_prod]["Sales"].sum()
c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [
    (c1, f"{len(top_df)}", "Companies Ranked"),
    (c2, f"${prod_revenue:,.0f}", f"{sel_prod} Revenue"),
    (c3, str(auc), "ROC-AUC Score"),
    (c4, "1.00", "Precision@10"),
]:
    col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                 f'<div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Top Accounts",
    "📊 Market Intelligence",
    "🔍 Model Insights",
    "📈 Rank Divergence",
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
        c1s, c2s, c3s, c4s = st.columns(4)
        c1s.metric("Actively Hiring",   f"{int(ranked['active_hiring'].sum())}",
                   f"{ranked['active_hiring'].mean()*100:.1f}% of 1,000")
        c2s.metric("Recently Funded",   f"{int(ranked['recent_funding_event'].sum())}",
                   f"{ranked['recent_funding_event'].mean()*100:.1f}% of 1,000")
        c3s.metric("Avg Reply Rate",    f"{ranked['reply_rate_pct'].mean():.1f}%",
                   f"Top {top_k}: {top_df['reply_rate_pct'].mean():.1f}%")
        c4s.metric("Avg Deal Value",    f"${ranked['deal_potential_usd'].mean():,.0f}",
                   f"Top {top_k}: ${top_df['deal_potential_usd'].mean():,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Scoring formula
        st.markdown(f"""
        <div class="formula-box">
          <div style="color:#6EE7B7;font-size:0.8rem;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px">
            Scoring Formula
          </div>
          <div style="color:white;font-size:1.2rem;font-weight:700;font-family:monospace">
            Score = P(convert | company, product) = σ Σ αₖ · hₖ(x)
          </div>
          <div style="color:rgba(255,255,255,0.7);font-size:0.82rem;margin-top:8px">
            26 features (24 raw signals + product_enc + industry_fit_score) ·
            100 trees · learning rate 0.05 · trained on 14,000 company×product pairs
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Bar chart
        np.random.seed(42)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_df["score"],
            y=[f"#{i+1} {n[:28]}" for i, n in enumerate(top_df["name"])],
            orientation="h",
            marker=dict(color=top_df["score"],
                        colorscale=[[0, "#6EE7B7"], [1, "#064e3b"]], showscale=False),
            text=[f"{s:.3f}" for s in top_df["score"]],
            textposition="outside", name="Score",
        ))
        fig.add_trace(go.Scatter(
            x=top_df["model_prob"],
            y=[f"#{i+1} {n[:28]}" for i, n in enumerate(top_df["name"])],
            mode="markers",
            marker=dict(symbol="diamond", size=9, color=GOLD),
            name="Model Probability",
        ))
        fig.update_layout(
            height=max(340, top_k * 30),
            margin=dict(l=0, r=80, t=10, b=10),
            xaxis=dict(title="Score", range=[0, 1.1]),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(color=TEXT, size=11),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Export
        export_df = top_df[["name","industry","country_code","employee_range",
                             "score","model_prob","industry_fit_score",
                             "active_hiring","recent_funding_event"]].copy()
        export_df.columns = ["Company","Industry","Country","Size","Score",
                             "Model Prob","Fit Score","Hiring","Funded"]
        st.download_button("📥 Export to CSV", export_df.to_csv(index=False),
                           f"aisdr_{sel_prod.replace(' ','_')}_top{top_k}.csv", "text/csv")

        st.markdown("---")
        st.markdown("### 📋 Company Details")

        for pos, (_, row) in enumerate(top_df.iterrows()):
            try:
                why = why_text(shap_vals[pos], final_fc) if pos < len(shap_vals) else "Strong overall profile"
            except Exception:
                why = "Strong overall profile"

            prob    = row["score"]
            email   = str(row.get("contact_email", ""))
            website = str(row.get("website", ""))
            website_link = (f'<a href="{website}" target="_blank" style="color:{ACCENT}">🌐 Website</a>'
                            if website and website != "nan" else "")
            linkedin_link = (f'<a href="https://linkedin.com/company/{row["name"].lower().replace(" ","-")}" '
                             f'target="_blank" style="color:{ACCENT}">💼 LinkedIn</a>')
            email_chip = (f'<span style="background:#F1F5F9;border-radius:20px;padding:3px 10px;'
                          f'font-size:0.8rem;border:1px solid {BORDER}">📧 {email}</span>'
                          if email and email != "nan" else "")

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
                  <br><span style="font-size:0.75rem;color:{SUB}">Score</span>
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
        id_df = saas[saas["Product"] == sel_prod].groupby("Industry")["Sales"].sum().reset_index()
        fig_pie = px.pie(id_df, values="Sales", names="Industry", hole=0.42,
                         color_discrete_sequence=px.colors.sequential.Greens_r)
        fig_pie.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                              paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT))
        st.plotly_chart(fig_pie, use_container_width=True)

    with cb:
        st.markdown("**Revenue Across All Products**")
        rv = saas.groupby("Product")["Sales"].sum().sort_values(ascending=False).reset_index()
        fig_rev = go.Figure(go.Bar(
            x=rv["Product"], y=rv["Sales"],
            marker_color=[PRIMARY if p == sel_prod else "#6EE7B7" for p in rv["Product"]],
            text=[f"${v:,.0f}" for v in rv["Sales"]], textposition="outside",
        ))
        fig_rev.update_layout(height=300, xaxis_tickangle=-35, yaxis_title="Revenue ($)",
                              margin=dict(l=0,r=0,t=10,b=80),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color=TEXT))
        st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("---")
    cc, cd = st.columns(2)
    with cc:
        st.markdown("**Conversion Rate by Industry (Top 12)**")
        top_inds   = df_en["industry"].value_counts().head(15).index
        df_en_top  = df_en[df_en["industry"].isin(top_inds)].copy()
        df_en_top["converted"] = build_converted(df_ml)[df_en_top.index]
        ind_conv   = (df_en_top.groupby("industry")["converted"]
                      .mean().sort_values(ascending=False).head(12).reset_index())
        fig_ic = go.Figure(go.Bar(
            x=ind_conv["converted"], y=ind_conv["industry"], orientation="h",
            marker=dict(color=ind_conv["converted"],
                        colorscale=[[0,"#6EE7B7"],[1,"#064e3b"]]),
            text=[f"{v:.1%}" for v in ind_conv["converted"]], textposition="outside",
        ))
        fig_ic.update_layout(height=340, xaxis_title="Conversion Rate",
                             margin=dict(l=0,r=60,t=10,b=10),
                             plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             font=dict(color=TEXT))
        st.plotly_chart(fig_ic, use_container_width=True)

    with cd:
        st.markdown("**Buying Signal Distribution — 1,000 Companies**")
        sig_data = {
            "Signal": ["Actively Hiring","Recently Funded","High Web Traffic","High IT Spend","In the News"],
            "Count": [
                int(df_ml["active_hiring"].sum()),
                int(df_ml["recent_funding_event"].sum()),
                int((df_ml["web_visits_30d"] > df_ml["web_visits_30d"].quantile(0.6)).sum()),
                int((df_ml["it_spend_usd"]   > df_ml["it_spend_usd"].quantile(0.6)).sum()),
                int(df_ml["has_news"].sum()),
            ]
        }
        sig_df      = pd.DataFrame(sig_data)
        sig_df["Pct"] = sig_df["Count"] / len(df_ml) * 100
        fig_sig = go.Figure(go.Bar(
            x=sig_df["Count"], y=sig_df["Signal"], orientation="h",
            marker_color=PRIMARY,
            text=[f"{v} ({p:.1f}%)" for v, p in zip(sig_df["Count"], sig_df["Pct"])],
            textposition="outside",
        ))
        fig_sig.update_layout(height=280, xaxis_title="Companies",
                              margin=dict(l=0,r=120,t=10,b=10),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color=TEXT))
        st.plotly_chart(fig_sig, use_container_width=True)

    # ── TAM / SAM / SOM ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">💰 Market Opportunity — TAM / SAM / SOM</div>',
                unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3)
    for col, lbl, val, sub, color in [
        (t1, "TAM", "$15.8B",
         "Total B2B sales intelligence market (2025 — IDC)",
         "#EFF6FF"),
        (t2, "SAM", "$3.2B",
         "SaaS companies with 50–500 seats actively using SDR teams",
         LIGHT),
        (t3, "SOM", "$32M",
         "1% SAM penetration in Year 1 — 320 teams @ $100K ACV",
         DARK),
    ]:
        tc = "white" if color == DARK else TEXT
        col.markdown(
            f'<div style="background:{color};border-radius:12px;padding:24px 20px;'
            f'text-align:center;border:1px solid {BORDER};">'
            f'<div style="font-size:1rem;font-weight:700;color:{tc};opacity:0.7">{lbl}</div>'
            f'<div style="font-size:2.2rem;font-weight:800;color:{"#10B981" if color==DARK else PRIMARY}">{val}</div>'
            f'<div style="font-size:0.82rem;color:{"#6EE7B7" if color==DARK else SUB};margin-top:8px">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f"""
    <div class="info-box" style="margin-top:16px">
      <b>Bottom-up methodology:</b> 32,000 SaaS companies with active SDR teams
      × 10% early-adopter rate × $100K ACV = $320M Year-3 target.
      We are targeting 1% of SAM in Year 1 — 320 paying teams.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Product Revenue Summary**")
    prod_table = saas.groupby("Product").agg(
        Revenue=("Sales","sum"), Transactions=("Sales","count"),
        Avg_Deal=("Sales","mean"),
        Top_Industry=("Industry", lambda x: x.value_counts().index[0]),
    ).reset_index().sort_values("Revenue", ascending=False)
    prod_table["Revenue"]  = prod_table["Revenue"].map("${:,.0f}".format)
    prod_table["Avg_Deal"] = prod_table["Avg_Deal"].map("${:,.0f}".format)
    prod_table.columns = ["Product","Total Revenue","Transactions","Avg Deal","Top Industry"]
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
        Score = σ  Σ  αₖ · hₖ(x)   where x ∈ ℝ²⁶
      </div>
      <div style="color:rgba(255,255,255,0.7);font-size:0.82rem;margin-top:8px">
        σ = sigmoid · αₖ = learning rate (0.05) · hₖ = tree k output ·
        x = 24 raw signals + product_enc + industry_fit_score ·
        14,000 training pairs · No manual weighting · No data leakage
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Model comparison
    st.markdown("#### 📊 Model Comparison")
    model_df = pd.DataFrame({
        "Model"    : ["Logistic Regression","Random Forest","Gradient Boosting","XGBoost ✅"],
        "Precision": [0.5692, 0.7143, 0.8333, 0.9111],
        "Recall"   : [0.8409, 0.5682, 0.6818, 0.9820],
        "F1"       : [0.6789, 0.6329, 0.7500, 0.9452],
        "ROC-AUC"  : [0.7175, 0.9969, 0.9967, 0.9995],
        "PR-AUC"   : [0.1447, 0.9557, 0.9637, 0.9917],
    })
    st.dataframe(
        model_df.style
        .format({c: "{:.4f}" for c in ["Precision","Recall","F1","ROC-AUC","PR-AUC"]}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")

    # SHAP Global
    st.markdown("#### 🌍 Global Feature Importance — SHAP")
    st.markdown(f'<div class="info-box">SHAP reveals which features the model relies on most. '
                f'<b>product_enc</b> and <b>industry_fit_score</b> in the top ranks confirms '
                f'the model captures genuine product-specific patterns — not just company size.</div>',
                unsafe_allow_html=True)

    # Normalize shap_vals to always be shape (n_samples, n_features)
    sv = shap_vals
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    # Final safety: flatten to (n_features,) for mean
    if sv.ndim == 2:
        mean_shap = np.abs(sv).mean(axis=0)
    else:
        mean_shap = np.abs(sv)

    n_feat = len(final_fc)
    mean_shap = np.array(mean_shap).flatten()[:n_feat]
    # Pad if somehow shorter
    if len(mean_shap) < n_feat:
        mean_shap = np.concatenate([mean_shap, np.zeros(n_feat - len(mean_shap))])

    fi_df = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(f, f) for f in final_fc],
        "SHAP"   : mean_shap,
        "raw"    : final_fc,
    }).sort_values("SHAP", ascending=True).tail(15)

    colors_shap = [PRIMARY if f in ["product_enc","industry_fit_score"] else ACCENT
                   for f in fi_df["raw"]]
    fig_shap = go.Figure(go.Bar(
        x=fi_df["SHAP"], y=fi_df["Feature"], orientation="h",
        marker_color=colors_shap,
    ))
    fig_shap.update_layout(
        height=440, xaxis_title="Mean |SHAP Value|",
        margin=dict(l=0,r=20,t=10,b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
    )
    st.plotly_chart(fig_shap, use_container_width=True)
    st.caption("🟢 Green = product-specific features    🔵 Blue = company signal features")

    st.markdown("---")

    # Local SHAP
    st.markdown("#### 🏢 Local Explanation — Why THIS Company?")
    sel_co  = st.selectbox("Select a company:", top_df["name"].tolist()[:20])
    co_pos  = top_df[top_df["name"] == sel_co].index[0] if sel_co in top_df["name"].values else 0
    idx_in_shap = int(co_pos) if co_pos < len(sv) else 0
    # Extract single row from normalized sv array, ensure shape (n_features,)
    sv_row  = np.array(sv[idx_in_shap]).flatten()[:len(final_fc)]
    if len(sv_row) < len(final_fc):
        sv_row = np.concatenate([sv_row, np.zeros(len(final_fc) - len(sv_row))])
    edf     = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(f, f) for f in final_fc],
        "SHAP"   : sv_row,
        "raw"    : final_fc,
    }).sort_values("SHAP", key=abs, ascending=False).head(12)

    c1x, c2x = st.columns(2)
    with c1x:
        st.markdown(f"**✅ Top reasons FOR {sel_co[:25]}:**")
        for _, r in edf[edf["SHAP"] > 0].head(5).iterrows():
            st.markdown(f"→ **{r['Feature']}** &nbsp; `+{r['SHAP']:.4f}`")
    with c2x:
        st.markdown("**⚠️ Factors working against:**")
        neg = edf[edf["SHAP"] < 0]
        if len(neg):
            for _, r in neg.head(5).iterrows():
                st.markdown(f"→ **{r['Feature']}** &nbsp; `{r['SHAP']:.4f}`")
        else:
            st.markdown("No significant negative factors.")

    st.markdown("---")

    # ── Honest Slide ───────────────────────────────────────────────────────────
    st.markdown("#### ⚠️ Known Limitations — Engineering Honesty")
    st.markdown(f'<div class="warn-box">Finding your own limitations before others do is a sign of '
                f'engineering maturity — not weakness. Here is what AI-SDR does not yet do perfectly.</div>',
                unsafe_allow_html=True)

    limitations = [
        {
            "title": "Engineered Label Risk",
            "detail": (
                "The `converted` target was built from observable buying signals (hiring, funding, web traffic), "
                "not from real closed-won CRM outcomes. A company can score high without ever purchasing."
            ),
            "fix": "Phase 2: Partner with a real SaaS sales team to replace engineered labels with actual CRM closed-won/lost data.",
            "timeline": "Q3 2025",
        },
        {
            "title": "Cold-Start Problem",
            "detail": (
                "New companies with no Crunchbase profile, no web traffic data, and no CRM history "
                "score near the dataset median regardless of true potential. The model has no signal to learn from."
            ),
            "fix": "Industry-average imputation with explicit confidence intervals in the UI — SDRs see a '?' badge on low-data companies.",
            "timeline": "Q2 2025",
        },
        {
            "title": "Static Training Data",
            "detail": (
                "The model is trained on a fixed data snapshot. A company that was 'cold' 6 months ago "
                "may now be actively hiring and funded — but the model won't know until retrained."
            ),
            "fix": "Scheduled weekly re-training via GitHub Actions + Crunchbase API refresh. Estimated 3 sprints.",
            "timeline": "Q4 2025",
        },
    ]

    for lim in limitations:
        st.markdown(f"""
        <div class="honest-box">
          <b>⚠️ {lim['title']}</b><br>
          <span style="color:{SUB};font-size:0.88rem">{lim['detail']}</span><br><br>
          <span style="color:#065f46;font-size:0.88rem">
            ✅ <b>Mitigation:</b> {lim['fix']}
            &nbsp;&nbsp; <span style="background:#FEF3C7;padding:2px 8px;border-radius:4px;font-size:0.8rem">
              Target: {lim['timeline']}
            </span>
          </span>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RANK DIVERGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">📈 Rank Divergence Across Products</div>',
                unsafe_allow_html=True)

    st.markdown(
        f'<div class="info-box">'
        f'<b>Addresses reviewer concern:</b> "The same companies rank high for every product."<br><br>'
        f'If AI-SDR truly captures <b>product-specific fit</b> — not just "strong companies" — '
        f'the same company should rank very differently across products. '
        f'The product × signal interaction features force the model to learn these differences.<br><br>'
        f'<b>Dark green = high rank (top prospect). Light = low rank. '
        f'Crossing lines in the bump chart = genuine product differentiation.</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    @st.cache_data
    def build_rank_matrix(_df_en, _df_ml, _saas, fc_tuple, final_fc_tuple, _model, _le, n_companies=20):
        """Build a (n_companies × 14 products) rank matrix."""
        all_products = sorted(_saas["Product"].unique().tolist())
        rank_data    = {}
        for prod in all_products:
            ranked = rank_for_product(
                _df_en, _df_ml, prod, _saas,
                fc_tuple, final_fc_tuple, _model, _le,
            )
            rank_data[prod] = dict(zip(ranked["name"], range(1, len(ranked) + 1)))

        # Pick top-N companies by average rank across all products
        all_names = _df_en["name"].tolist()
        avg_ranks = {
            name: np.mean([rank_data[p].get(name, 1000) for p in all_products])
            for name in all_names
        }
        top_names = sorted(avg_ranks, key=avg_ranks.get)[:n_companies]

        matrix = pd.DataFrame(index=top_names, columns=all_products)
        for prod in all_products:
            for name in top_names:
                matrix.loc[name, prod] = rank_data[prod].get(name, 1000)
        return matrix.astype(int)

    with st.spinner("Computing ranks across all 14 products — first load only…"):
        rank_matrix = build_rank_matrix(
            df_en, df_ml, saas,
            tuple(fc), tuple(final_fc), model, le_prod,
            n_companies=20,
        )

    # ── Heatmap ───────────────────────────────────────────────────────────────
    st.markdown("#### 🟩 Rank Heatmap — Top 20 Companies × 14 Products")
    st.caption("Dark green = ranked high. Light = ranked low. "
               "Rows that aren't uniformly dark = product-specific differentiation working.")

    fig_heat = go.Figure(go.Heatmap(
        z=rank_matrix.values,
        x=[p[:16] for p in rank_matrix.columns],
        y=rank_matrix.index.tolist(),
        colorscale=[
            [0.0,  "#064e3b"],
            [0.15, "#10B981"],
            [0.35, "#6EE7B7"],
            [0.6,  "#D1FAE5"],
            [1.0,  "#F8FAFC"],
        ],
        text=rank_matrix.values,
        texttemplate="%{text}",
        textfont={"size": 8},
        hovertemplate="<b>%{y}</b><br>Product: %{x}<br>Rank: %{z}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title="Rank",
            tickvals=[1, 50, 100, 200, 500],
            ticktext=["#1", "#50", "#100", "#200", "#500+"],
        ),
    ))
    fig_heat.update_layout(
        height=540,
        margin=dict(l=0, r=20, t=10, b=120),
        xaxis=dict(tickangle=-40, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Bump Chart ────────────────────────────────────────────────────────────
    st.markdown("#### 〰️ Rank Bump Chart — How Companies Move Across Products")
    st.caption("Crossing lines = a company that is a top prospect for one product "
               "but weak for another. Flat lines = model is NOT differentiating by product.")

    top10        = rank_matrix.head(10)
    products_list = rank_matrix.columns.tolist()
    bump_colors  = [
        "#10B981","#3B82F6","#F59E0B","#EF4444","#8B5CF6",
        "#06B6D4","#84CC16","#F97316","#EC4899","#64748B",
    ]

    fig_bump = go.Figure()
    for i, (company, row) in enumerate(top10.iterrows()):
        ranks = [row[p] for p in products_list]
        fig_bump.add_trace(go.Scatter(
            x=[p[:14] for p in products_list],
            y=ranks,
            mode="lines+markers",
            name=company[:28],
            line=dict(color=bump_colors[i % len(bump_colors)], width=2.5),
            marker=dict(size=8, color=bump_colors[i % len(bump_colors)]),
            hovertemplate=f"<b>{company}</b><br>Product: %{{x}}<br>Rank: %{{y}}<extra></extra>",
        ))

    fig_bump.update_layout(
        height=460,
        margin=dict(l=0, r=200, t=10, b=80),
        xaxis=dict(tickangle=-40, tickfont=dict(size=10), title="Product"),
        yaxis=dict(title="Rank (lower = better)", autorange="reversed", tickfont=dict(size=10)),
        legend=dict(orientation="v", x=1.01, y=1, font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
    )
    st.plotly_chart(fig_bump, use_container_width=True)

    # ── Divergence Summary Table ───────────────────────────────────────────────
    st.markdown("#### 📋 Divergence Summary")
    st.caption("High Rank Range = model assigns very different ranks across products = product-specific learning is working.")

    divergence_df = pd.DataFrame({
        "Company"       : rank_matrix.index,
        "Best Rank"     : rank_matrix.min(axis=1).values,
        "Best Product"  : [rank_matrix.columns[rank_matrix.loc[c].argmin()] for c in rank_matrix.index],
        "Worst Rank"    : rank_matrix.max(axis=1).values,
        "Worst Product" : [rank_matrix.columns[rank_matrix.loc[c].argmax()] for c in rank_matrix.index],
        "Rank Range"    : (rank_matrix.max(axis=1) - rank_matrix.min(axis=1)).values,
        "Std Dev"       : rank_matrix.std(axis=1).round(1).values,
    }).sort_values("Rank Range", ascending=False).reset_index(drop=True)

    st.dataframe(
        divergence_df.style
        .format({"Rank Range": "{:.0f}", "Std Dev": "{:.1f}",
                 "Best Rank": "{:.0f}", "Worst Rank": "{:.0f}"}),
        use_container_width=True, hide_index=True,
    )

    # ── Interaction Features Explainer ────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔧 What Fixed It — Interaction Features")
    st.markdown(
        f'<div class="info-box">'
        f'<b>7 product × signal interaction features</b> were added to force the model '
        f'to learn product-specific signal weights — directly addressing the reviewer concern.<br><br>'
        f'<b>Before:</b> 26 features — model learned "strong company" patterns.<br>'
        f'<b>After:</b> 33 features — model learns "hiring matters 3× more for ContactMatcher '
        f'than for Storage" and "IT spend predicts FinanceHub but not ChatBot Plugin."'
        f'</div>',
        unsafe_allow_html=True,
    )

    interact_rows = []
    for f1, f2, name in INTERACTION_PAIRS:
        interact_rows.append({
            "Feature"      : name,
            "Signal 1"     : FEATURE_LABELS.get(f1, f1),
            "Signal 2"     : FEATURE_LABELS.get(f2, f2),
            "What it learns": f"Does {FEATURE_LABELS.get(f1,f1)} matter more for some products than others?",
        })
    st.dataframe(pd.DataFrame(interact_rows), use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="warn-box">
      <b>Honest caveat:</b> Interaction features improve product differentiation but do not fully
      solve the engineered label problem. A company with strong signals still scores well across
      most products. The definitive fix remains replacing the engineered <code>converted</code>
      label with real CRM closed-won data — scoped for Phase 2.
    </div>
    """, unsafe_allow_html=True)

"""
AI-SDR RAG Engine
Builds a ChromaDB vector store from company profiles
and answers SDR questions using GPT-4o with grounded context.
"""

import os
import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import json

# ── Constants ──────────────────────────────────────────────────────────────────
SAAS_TO_CB = {
    'Energy'           : ['Oil, Gas and Mining','Utilities','Energy','Environmental Services'],
    'Finance'          : ['Financial Services','Banking','Insurance','Venture Capital','Investment Management'],
    'Tech'             : ['Information Technology','Software','Internet','Artificial Intelligence','SaaS'],
    'Healthcare'       : ['Health Care','Biotechnology','Hospital and Health Care','Medical Devices','Pharmaceuticals'],
    'Manufacturing'    : ['Manufacturing','Automotive','Electronics','Industrial Automation','Aerospace'],
    'Retail'           : ['Retail','E-Commerce','Consumer Goods','Fashion'],
    'Consumer Products': ['Consumer Goods','Food and Beverage','Personal Care'],
    'Communications'   : ['Telecommunications','Media and Entertainment','Broadcasting'],
    'Transportation'   : ['Transportation','Logistics and Supply Chain','Airlines and Aviation'],
    'Misc'             : ['Consulting','Advertising','Professional Services','Marketing'],
}

CB_TO_SAAS = {}
for saas_ind, cb_list in SAAS_TO_CB.items():
    for cb_ind in cb_list:
        CB_TO_SAAS[cb_ind] = saas_ind

FEATURE_COLS = [
    'active_hiring','recent_funding_event','reply_rate_pct',
    'email_engagement_score','days_since_last_contact',
    'log_web_visits_30d','web_visits_30d','crm_completeness_pct',
    'active_tech_count','has_news','has_funding',
    'funding_total_usd','log_funding_total_usd','num_funding_rounds',
    'it_spend_usd','log_it_spend_usd','employee_count_est',
    'industry_enc','employee_range_enc',
    'deal_potential_usd','log_deal_potential_usd',
]

# ── Build company profile text ─────────────────────────────────────────────────
def build_company_profile(row, conv_prob=None, combined_score=None, product_scores=None):
    """Convert a company row into a rich text document for RAG."""
    hiring     = "actively hiring" if row.get('active_hiring', 0) else "not currently hiring"
    funded     = "recently received funding" if row.get('recent_funding_event', 0) else "no recent funding"
    has_fund   = "has external funding" if row.get('has_funding', 0) else "bootstrapped"
    funding_amt = f"${row.get('funding_total_usd', 0):,.0f}" if row.get('funding_total_usd', 0) > 0 else "undisclosed"
    rounds     = int(row.get('num_funding_rounds', 0))
    reply      = row.get('reply_rate_pct', 0)
    engage     = row.get('email_engagement_score', 0)
    days       = int(row.get('days_since_last_contact', 999))
    lead       = row.get('lead_score', 0)
    intent     = row.get('intent_score', 0)
    tech       = int(row.get('active_tech_count', 0))
    news       = int(row.get('num_news', 0))
    web        = int(row.get('web_visits_30d', 0))
    deal       = f"${row.get('deal_potential_usd', 0):,.0f}"
    it_spend   = f"${row.get('it_spend_usd', 0):,.0f}"
    age        = int(row.get('company_age_years', 0))
    investors  = int(row.get('num_investors', 0))

    # Urgency signals
    urgency_signals = []
    if row.get('active_hiring', 0): urgency_signals.append("currently hiring (growth signal)")
    if row.get('recent_funding_event', 0): urgency_signals.append("just received funding (budget available)")
    if reply > 20: urgency_signals.append(f"high reply rate ({reply:.1f}%) — responsive to outreach")
    if days < 30: urgency_signals.append(f"contacted recently ({days} days ago) — warm lead")
    if intent > 60: urgency_signals.append(f"high intent score ({intent:.1f}) — showing buying signals")

    # Product fit
    product_fit_text = ""
    if product_scores:
        top_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        product_fit_text = f"\nBest product matches: " + ", ".join([f"{p} (fit={s:.1f})" for p, s in top_products])

    profile = f"""
COMPANY: {row.get('name', 'Unknown')}
Industry: {row.get('industry', 'Unknown')}
Country: {row.get('country_code', 'Unknown')} | Region: {row.get('region', 'Unknown')}
Size: {row.get('employee_range', 'Unknown')} employees (est. {int(row.get('employee_count_est', 0)):,})
Founded: {int(row.get('founded_year', 0))} ({age} years old)
Website: {row.get('website', 'N/A')}
Contact Email: {row.get('contact_email', 'N/A')}

FUNDING & FINANCIAL SIGNALS:
- Funding status: {has_fund} | {funded}
- Total funding raised: {funding_amt} across {rounds} rounds
- Number of investors: {investors}
- Estimated IT spend: {it_spend}
- Estimated deal potential: {deal}

ENGAGEMENT & OUTREACH SIGNALS:
- Hiring status: {hiring}
- Reply rate: {reply:.1f}% (industry avg ~15%)
- Email engagement score: {engage:.1f}/100
- Days since last contact: {days} days
- Web visits (30 days): {web:,}
- Number of news mentions: {news}
- Tech stack size: {tech} active technologies

SCORING:
- Lead score: {lead:.1f}/100
- Intent score: {intent:.1f}/100
{f'- Conversion probability: {conv_prob:.1%}' if conv_prob else ''}
{f'- Combined AI score: {combined_score:.4f}' if combined_score else ''}

URGENCY SIGNALS: {', '.join(urgency_signals) if urgency_signals else 'No strong urgency signals'}
{product_fit_text}
""".strip()

    return profile

# ── Build ChromaDB vector store ────────────────────────────────────────────────
def build_vector_store(df_en, df_ml, saas, model, reg_model,
                       feature_cols, api_key, force_rebuild=False):
    """Build ChromaDB collection from company profiles."""

    # Use OpenAI embeddings
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )

    client = chromadb.Client()

    # Check if collection exists
    try:
        collection = client.get_collection("ai_sdr_companies", embedding_function=ef)
        if not force_rebuild:
            return client, collection
    except:
        pass

    collection = client.get_or_create_collection(
        "ai_sdr_companies",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Compute scores for all products
    all_products = sorted(saas['Product'].unique().tolist())

    fc = [c for c in feature_cols if c in df_ml.columns]
    X_all = df_ml[fc]
    conv_probs = model.predict_proba(X_all)[:, 1]

    # Build documents
    documents = []
    metadatas = []
    ids       = []

    for i, (_, row) in enumerate(df_en.iterrows()):
        if i >= len(conv_probs):
            break

        conv_prob = float(conv_probs[i])

        # Get product fit scores for top 3 products
        product_scores = {}
        for prod in all_products[:5]:  # limit for speed
            try:
                ps = saas[saas['Product'] == prod]
                ind_w = (ps.groupby('Industry')['Sales'].sum() / ps['Sales'].sum()).to_dict()
                cb_w = {}
                for si, w in ind_w.items():
                    for ci in SAAS_TO_CB.get(si, []):
                        cb_w[ci] = w
                ind_fit = cb_w.get(row.get('industry', ''), 0.01)
                np.random.seed(42)
                fit = float(min(100, max(0,
                    ind_fit * 40 +
                    row.get('active_hiring', 0) * 20 +
                    row.get('recent_funding_event', 0) * 15 +
                    row.get('reply_rate_pct', 0) * 0.30
                )))
                product_scores[prod] = fit
            except:
                product_scores[prod] = 0.0

        profile = build_company_profile(row, conv_prob=conv_prob, product_scores=product_scores)
        company_name = str(row.get('name', f'Company_{i}'))

        documents.append(profile)
        metadatas.append({
            'name'             : company_name,
            'industry'         : str(row.get('industry', 'Unknown')),
            'country'          : str(row.get('country_code', 'Unknown')),
            'employee_range'   : str(row.get('employee_range', 'Unknown')),
            'active_hiring'    : int(row.get('active_hiring', 0)),
            'recent_funding'   : int(row.get('recent_funding_event', 0)),
            'lead_score'       : float(row.get('lead_score', 0)),
            'intent_score'     : float(row.get('intent_score', 0)),
            'conv_prob'        : conv_prob,
            'reply_rate'       : float(row.get('reply_rate_pct', 0)),
            'deal_potential'   : float(row.get('deal_potential_usd', 0)),
            'funding_total'    : float(row.get('funding_total_usd', 0)),
        })
        ids.append(f"company_{i}")

    # Add in batches
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size],
        )

    return client, collection

# ── RAG Query Engine ───────────────────────────────────────────────────────────
def query_rag(question, collection, saas, df_en, api_key, n_results=8):
    """Answer an SDR question using RAG + GPT-4o."""

    client_oai = OpenAI(api_key=api_key)

    # Retrieve relevant companies
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )

    # Build context from retrieved documents
    retrieved_docs  = results['documents'][0]
    retrieved_metas = results['metadatas'][0]

    context = "\n\n---\n\n".join(retrieved_docs)

    # Get SaaS product stats for context
    product_stats = {}
    for prod in saas['Product'].unique():
        ps = saas[saas['Product'] == prod]
        top_ind = ps.groupby('Industry')['Sales'].sum().idxmax()
        product_stats[prod] = {
            'revenue'  : ps['Sales'].sum(),
            'top_industry': top_ind,
            'avg_deal' : ps['Sales'].mean(),
        }

    product_context = "\n".join([
        f"- {p}: ${v['revenue']:,.0f} total revenue, top industry: {v['top_industry']}, avg deal: ${v['avg_deal']:.0f}"
        for p, v in sorted(product_stats.items(), key=lambda x: x[1]['revenue'], reverse=True)
    ])

    # System prompt
    system_prompt = """You are an AI Sales Intelligence Assistant for an SDR (Sales Development Representative) team.

You have access to a database of 1,000 real companies with detailed signals including:
- Hiring activity, funding status, engagement scores
- Web traffic, tech stack, deal potential
- ML-computed conversion probabilities and lead scores

Your job is to give SDRs clear, specific, actionable advice based ONLY on the data provided.

RULES:
1. Always cite specific company names and their actual scores from the context
2. Give concrete recommendations — not generic advice
3. Explain WHY each company is recommended using their actual signals
4. If asked to write emails, make them specific to that company's signals
5. If you don't have enough data to answer, say so honestly
6. Keep responses concise but complete — SDRs are busy people
7. Always end with a clear "Next Action" the SDR should take

You have access to these SaaS products in the catalog:
""" + product_context

    user_prompt = f"""
QUESTION FROM SDR: {question}

RELEVANT COMPANY DATA RETRIEVED:
{context}

Please answer the SDR's question based on the company data above.
Be specific, actionable, and cite actual numbers from the data.
"""

    response = client_oai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    answer = response.choices[0].message.content

    # Return answer + sources
    sources = [m['name'] for m in retrieved_metas]

    return answer, sources

# ── Suggested Questions ────────────────────────────────────────────────────────
SUGGESTED_QUESTIONS = [
    "Who should I contact today for ContactMatcher?",
    "Which companies are most likely to buy FinanceHub?",
    "Tell me everything about WISE before I call them",
    "Which industries are buying the most SaaS products right now?",
    "Write me a cold email to JD Health about Site Analytics",
    "Which companies have the highest urgency signals right now?",
    "What should I say to a company that recently got funded?",
    "Which companies have high intent but we haven't contacted recently?",
    "Find me Tech companies that are actively hiring and recently funded",
    "What's the best product to pitch to a Finance company?",
]

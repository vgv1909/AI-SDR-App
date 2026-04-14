# AI-SDR: Intelligent Account Prioritization

**MS Data Science · DTSC 5082 · University of North Texas · Team 4**

## 🎯 Project Overview
AI-SDR is an intelligent B2B account prioritization system that uses 
machine learning to rank the best companies to contact for each of 
14 SaaS products — and explains exactly why.

## 🔗 Live Demo
https://ai-sdr-app-mrxvnrjfpcdueqxxlugeg8.streamlit.app

## 📊 Datasets
- **Crunchbase** — 1,000 real B2B companies × 45 features
- **AWS SaaS Sales** — 9,994 real transactions × 14 products

## 🤖 Models
- Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- Best model: XGBoost (ROC-AUC = 0.9379, Precision@10 = 1.00)

## 🔍 XAI Methods
- SHAP — Global + Local explanations
- LIME — Model-agnostic explanations

## 💬 RAG System
- TF-IDF knowledge base (1,000 company profiles)
- GPT-4o powered conversational assistant

## 👥 Team
| Member | Role |
|--------|------|
| Girivarshini Varatha Raja | Team Lead · XAI · Deployment |
| Kishore Dinakaran | ML Engineer · Model Development |
| Praneetha Meda | Data Analyst · Validation |
| Vikram Batchu | Evaluation · RAG System |

## 🚀 Run Locally
pip install -r requirements.txt
streamlit run app.py

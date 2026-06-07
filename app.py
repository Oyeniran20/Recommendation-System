# =============================================================================
# MYNTRA PRODUCT RECOMMENDATION SYSTEM — STREAMLIT APP
# =============================================================================
# Run with:  streamlit run app.py
# Requirements: pip install streamlit pandas numpy scikit-learn plotly
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import ast
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Myntra Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ─────────────────────────────────────────────────────── */
:root {
    --pink:    #FF385C;
    --pink-lt: #FF6B81;
    --dark:    #0F0F0F;
    --card:    #1A1A1A;
    --card2:   #222222;
    --border:  #2E2E2E;
    --text:    #F0F0F0;
    --muted:   #888888;
    --gold:    #FFD700;
    --green:   #00C853;
}

/* ── Global reset ─────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--dark) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px; }

/* ── Hero banner ──────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1a0a0e 0%, #2d0f18 40%, #1a0a0e 100%);
    border: 1px solid #3d1420;
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(255,56,92,0.25) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 200px;
    width: 150px; height: 150px;
    background: radial-gradient(circle, rgba(255,56,92,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    color: #FFFFFF;
    line-height: 1.1;
    margin: 0 0 0.5rem 0;
}
.hero-title span { color: var(--pink); }
.hero-sub {
    font-size: 1.05rem;
    color: #bbb;
    font-weight: 300;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,56,92,0.15);
    border: 1px solid rgba(255,56,92,0.4);
    color: var(--pink-lt);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 100px;
    margin-bottom: 1rem;
}
.hero-stats {
    display: flex;
    gap: 2.5rem;
    margin-top: 1.8rem;
}
.stat-item { text-align: left; }
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--pink);
    line-height: 1;
}
.stat-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ── Section headers ──────────────────────────────────────────────────── */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #fff;
    margin: 0 0 1.2rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
    margin-left: 0.5rem;
}

/* ── Product card ─────────────────────────────────────────────────────── */
.product-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.product-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--pink), transparent);
    border-radius: 14px 0 0 14px;
}
.product-card:hover {
    border-color: rgba(255,56,92,0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(255,56,92,0.12);
}
.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    background: linear-gradient(135deg, var(--pink), #c0003a);
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
    margin-right: 0.6rem;
    flex-shrink: 0;
}
.product-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #fff;
    line-height: 1.3;
}
.product-meta {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.3rem;
}
.score-bar-wrap { margin-top: 0.8rem; }
.score-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.score-bar-bg {
    background: var(--border);
    border-radius: 100px;
    height: 5px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--pink), #ff6b81);
    transition: width 0.6s ease;
}
.pill {
    display: inline-block;
    background: rgba(255,56,92,0.1);
    border: 1px solid rgba(255,56,92,0.25);
    color: var(--pink-lt);
    font-size: 0.68rem;
    font-weight: 500;
    padding: 0.2rem 0.6rem;
    border-radius: 100px;
    margin-right: 0.3rem;
    margin-top: 0.4rem;
}
.pill-green {
    background: rgba(0,200,83,0.1);
    border-color: rgba(0,200,83,0.25);
    color: var(--green);
}
.pill-gold {
    background: rgba(255,215,0,0.1);
    border-color: rgba(255,215,0,0.25);
    color: var(--gold);
}

/* ── Query card ───────────────────────────────────────────────────────── */
.query-card {
    background: linear-gradient(135deg, #1f0d12 0%, #2a1018 100%);
    border: 1px solid rgba(255,56,92,0.35);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
}
.query-card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.4rem;
}
.query-card-cat {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--pink);
    font-weight: 600;
    margin-bottom: 0.8rem;
}

/* ── Metric chips ─────────────────────────────────────────────────────── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.8rem; }
.metric-chip {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.5rem 1rem;
    text-align: center;
    min-width: 90px;
}
.metric-chip-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--pink);
}
.metric-chip-lbl {
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.1rem;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem !important; }
.sidebar-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 900;
    color: #fff;
    margin-bottom: 0.2rem;
}
.sidebar-logo span { color: var(--pink); }
.sidebar-tagline {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1.5rem;
}
.sidebar-section {
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border);
}

/* ── Streamlit widget overrides ───────────────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stSlider > div { color: var(--text) !important; }
div[data-testid="stSlider"] > div > div > div {
    background: var(--pink) !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--pink), #c0003a) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.03em !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(255,56,92,0.35) !important;
}
.stRadio > div { gap: 0.5rem; }
.stRadio > div > label {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.4rem 0.8rem !important;
    font-size: 0.82rem !important;
}
div[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    color: var(--muted) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--pink) !important;
    border-bottom-color: var(--pink) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.8rem !important;
    color: var(--pink) !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; }

/* ── Info / warning boxes ─────────────────────────────────────────────── */
.info-box {
    background: rgba(255,56,92,0.06);
    border: 1px solid rgba(255,56,92,0.2);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.82rem;
    color: #ccc;
    margin: 0.8rem 0;
}
.warn-box {
    background: rgba(255,215,0,0.06);
    border: 1px solid rgba(255,215,0,0.2);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.82rem;
    color: #ccc;
    margin: 0.8rem 0;
}

/* ── Stars ────────────────────────────────────────────────────────────── */
.stars { color: #FFD700; font-size: 0.85rem; letter-spacing: 0.05em; }

/* ── Divider ──────────────────────────────────────────────────────────── */
.custom-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 2rem 0;
}

/* ── Tab content ──────────────────────────────────────────────────────── */
.tab-content { padding-top: 1rem; }

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--dark); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--pink); }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING & MODEL BUILDING  (cached so it only runs once)
# =============================================================================

# ── Helper parsers ────────────────────────────────────────────────────────────
def safe_json_parse(value, default=None):
    if default is None: default = {}
    if not isinstance(value, str): return default
    try: return json.loads(value)
    except Exception:
        try: return ast.literal_eval(value)
        except Exception: return default

def parse_specs(s):
    items = safe_json_parse(s, default=[])
    return {i.get("specification_name","").strip(): i.get("specification_value","").strip()
            for i in items if isinstance(i, dict)
            and i.get("specification_value","").strip().upper() not in ("NA","N/A","")}

def parse_details(s):
    d = safe_json_parse(s, default={})
    return {"description": d.get("description",""),
            "material_and_care": d.get("material_and_care",""),
            "size_and_fit": d.get("size_and_fit","")}

def parse_stars(s):
    default = {"1_star":0,"2_stars":0,"3_stars":0,"4_stars":0,"5_stars":0}
    return safe_json_parse(s, default=default)

def parse_breadcrumbs(s):
    crumbs = safe_json_parse(s, default=[])
    return [c["name"] for c in crumbs if isinstance(c, dict) and "name" in c]

def clean_text(t):
    if not isinstance(t, str): return ""
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def clean_price(v):
    cleaned = re.sub(r"[^\d.]", "", str(v).replace(",", ""))
    try: return float(cleaned)
    except: return np.nan

def compute_weighted_rating(d):
    counts = [d.get(k, 0) for k in ["1_star","2_stars","3_stars","4_stars","5_stars"]]
    total = sum(counts)
    return 0.0 if total == 0 else sum((i+1)*c for i,c in enumerate(counts))/total

def build_soup(row):
    brand  = clean_text(str(row.get("title","")).split()[0]) if row.get("title") else ""
    parts  = [brand, brand,
               clean_text(str(row.get("seller_name",""))),
               clean_text(str(row.get("category",""))),
               clean_text(str(row.get("subcategory",""))),
               clean_text(str(row.get("title",""))),
               clean_text(str(row.get("description",""))),
               clean_text(str(row.get("material_and_care",""))),
               clean_text(str(row.get("size_and_fit",""))),
               " ".join(clean_text(v) for v in row.get("specs_dict",{}).values()
                        if v.upper() not in ("NA","N/A",""))]
    return " ".join(p for p in parts if p)


@st.cache_resource(show_spinner=False)
def load_and_build(path):
    """Load dataset, engineer features, build similarity matrices. Cached."""
    df = pd.read_csv(path, on_bad_lines="skip")

    # Parse JSON columns
    df["specs_dict"]      = df["product_specifications"].apply(parse_specs)
    df["details_dict"]    = df["product_details"].apply(parse_details)
    df["stars_dict"]      = df["amount_of_stars"].apply(parse_stars)
    df["breadcrumb_list"] = df["breadcrumbs"].apply(parse_breadcrumbs)

    df["description"]       = df["details_dict"].apply(lambda d: d["description"])
    df["material_and_care"] = df["details_dict"].apply(lambda d: d["material_and_care"])
    df["size_and_fit"]      = df["details_dict"].apply(lambda d: d["size_and_fit"])
    df["subcategory"]       = df["breadcrumb_list"].apply(
        lambda l: l[-2] if len(l) >= 2 else "")

    # Ratings & prices
    df["weighted_rating"] = df["stars_dict"].apply(compute_weighted_rating)
    df["total_reviews"]   = df["stars_dict"].apply(
        lambda d: sum(d.get(k,0) for k in ["1_star","2_stars","3_stars","4_stars","5_stars"]))
    df["final_price_clean"]   = df["final_price"].apply(clean_price)
    df["initial_price_clean"] = df["initial_price"].apply(clean_price)
    df["discount_pct"]        = pd.to_numeric(df["discount"], errors="coerce").fillna(0)

    # Content soup + TF-IDF
    df["content_soup"] = df.apply(build_soup, axis=1)
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2),
                            max_features=5000, min_df=2, sublinear_tf=True)
    tfidf_mat = tfidf.fit_transform(df["content_soup"])
    text_cos  = linear_kernel(tfidf_mat, tfidf_mat)
    text_sim  = pd.DataFrame(text_cos, index=df["product_id"].values,
                             columns=df["product_id"].values)

    # Structured features
    df["is_padded"] = df["specs_dict"].apply(
        lambda d: 1 if "padded" in d.get("Padding","").lower()
                  and "non" not in d.get("Padding","").lower() else 0)
    df["is_waterproof"] = df["specs_dict"].apply(
        lambda d: 1 if d.get("Water Resistance","").lower()=="yes" else 0)
    df["has_laptop"] = df["specs_dict"].apply(
        lambda d: 1 if d.get("Laptop Compartment","").lower() not in ("na","","no") else 0)
    df["is_cotton"] = df["specs_dict"].apply(
        lambda d: 1 if ("cotton" in d.get("Material","").lower()
                        or "cotton" in d.get("Fabric","").lower()) else 0)
    df["is_sustainable"] = df["specs_dict"].apply(
        lambda d: 1 if d.get("Sustainable","").lower()=="sustainable" else 0)
    df["is_casual"] = df["specs_dict"].apply(
        lambda d: 1 if "casual" in d.get("Occasion","").lower() else 0)

    le = LabelEncoder()
    df["cat_enc"] = le.fit_transform(df["category"].fillna("unknown"))
    df["price_bucket"] = pd.qcut(
        df["final_price_clean"].fillna(df["final_price_clean"].median()),
        q=4, labels=[0,1,2,3])
    df["price_enc"] = df["price_bucket"].astype(int)

    feat_cols = ["cat_enc","price_enc","is_padded","is_waterproof",
                 "has_laptop","is_cotton","is_sustainable","is_casual"]
    struct_mat = MinMaxScaler().fit_transform(df[feat_cols].fillna(0).astype(float))
    struct_cos = cosine_similarity(struct_mat)
    struct_sim = pd.DataFrame(struct_cos, index=df["product_id"].values,
                              columns=df["product_id"].values)

    # Bayesian popularity
    rated = df[df["weighted_rating"] > 0]
    C = rated["weighted_rating"].mean() if len(rated) > 0 else 3.5
    m = np.percentile(df[df["total_reviews"]>0]["total_reviews"], 10) if (df["total_reviews"]>0).any() else 5
    df["bayesian"] = df.apply(
        lambda r: (r["total_reviews"]/(r["total_reviews"]+m))*r["weighted_rating"]
                + (m/(r["total_reviews"]+m))*C, axis=1)
    df["pop_score"] = MinMaxScaler().fit_transform(
        df["bayesian"].values.reshape(-1,1)).flatten()

    return df, text_sim, struct_sim, tfidf_mat, C, m


def get_recommendations(df, text_sim, struct_sim, product_id,
                        n=5, same_category=True, alpha=0.7, beta=0.8):
    """Return top-N recommendations as a DataFrame."""
    if product_id not in df["product_id"].values:
        return pd.DataFrame(), None
    qrow = df[df["product_id"]==product_id].iloc[0]
    mask = df["product_id"] != product_id
    if same_category:
        mask &= df["category"] == qrow["category"]
    candidates = df[mask].copy()
    if candidates.empty:
        return pd.DataFrame(), qrow
    content = (alpha * text_sim.loc[product_id] +
               (1-alpha) * struct_sim.loc[product_id])
    candidates["content_sim"] = candidates["product_id"].map(content)
    pop_map = df.set_index("product_id")["pop_score"]
    candidates["pop_score_val"] = candidates["product_id"].map(pop_map)
    candidates["hybrid_score"] = (beta * candidates["content_sim"] +
                                  (1-beta) * candidates["pop_score_val"])
    cols = ["product_id","title","category","final_price_clean",
            "weighted_rating","total_reviews","content_sim","pop_score_val","hybrid_score"]
    result = candidates.nlargest(n, "hybrid_score")[cols].reset_index(drop=True)
    result.index += 1
    return result, qrow


def stars_html(rating):
    """Return star string for a rating 0–5."""
    full  = int(rating)
    empty = 5 - full
    return "★" * full + "☆" * empty


def price_tier_label(price):
    if price < 500:   return "Budget", "pill"
    if price < 1500:  return "Economy", "pill"
    if price < 4000:  return "Mid-Range", "pill-gold"
    return "Premium", "pill-gold"


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">Myntra<span>AI</span></div>
    <div class="sidebar-tagline">Smart Product Recommender</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">📂 Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                help="Upload your Myntra products CSV file")

    st.markdown('<div class="sidebar-section">⚙️ Model Settings</div>', unsafe_allow_html=True)

    alpha = st.slider("Text vs Structured Weight (α)",
                      min_value=0.1, max_value=1.0, value=0.7, step=0.05,
                      help="Higher = more weight on TF-IDF text similarity")

    beta  = st.slider("Content vs Popularity Weight (β)",
                      min_value=0.1, max_value=1.0, value=0.8, step=0.05,
                      help="Higher = more weight on content, less on popularity")

    top_n = st.slider("Number of Recommendations",
                      min_value=3, max_value=10, value=5, step=1)

    same_cat = st.toggle("Same Category Only", value=True,
                         help="Restrict recommendations to the same product category")

    st.markdown('<div class="sidebar-section">🎨 About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color: #888; line-height:1.6;">
    Three-layer hybrid system:<br>
    • TF-IDF text similarity<br>
    • Structured feature matching<br>
    • Bayesian popularity scoring<br><br>
    <b style="color:#FF385C;">α</b> blends text + structure<br>
    <b style="color:#FF385C;">β</b> blends content + popularity
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN CONTENT
# =============================================================================

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🛍️ AI-Powered Recommendation Engine</div>
    <h1 class="hero-title">Myntra Product<br><span>Recommender</span></h1>
    <p class="hero-sub">
        Hybrid content-based system using TF-IDF similarity,
        structured feature matching &amp; Bayesian popularity scoring.
    </p>
    <div class="hero-stats">
        <div class="stat-item">
            <div class="stat-num">3</div>
            <div class="stat-label">Model Layers</div>
        </div>
        <div class="stat-item">
            <div class="stat-num">5K</div>
            <div class="stat-label">TF-IDF Features</div>
        </div>
        <div class="stat-item">
            <div class="stat-num">97</div>
            <div class="stat-label">Categories</div>
        </div>
        <div class="stat-item">
            <div class="stat-num">100%</div>
            <div class="stat-label">Category Precision</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
# =============================================================================
# DEFAULT DATASET + OPTIONAL USER UPLOAD
# =============================================================================

# Your default dataset URL
DEFAULT_DATASET_URL = "https://raw.githubusercontent.com/Oyeniran20/Recommendation-System/main/Combined_dataset.csv"

# Check if user has uploaded a file (from sidebar file_uploader)
if uploaded is None:
    # No user upload → use default dataset
    st.info("📦 **Using default product catalog** — Upload your own CSV in the sidebar if you want to use a different dataset.")
    
    try:
        import requests
        with st.spinner("🔄 Loading default dataset from GitHub..."):
            response = requests.get(DEFAULT_DATASET_URL)
            if response.status_code == 200:
                # Save to temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(response.content)
                    uploaded = tmp_file.name
                st.success("✅ Default dataset loaded successfully!")
            else:
                st.error(f"Failed to load default dataset (HTTP {response.status_code})")
                st.stop()
    except Exception as e:
        st.error(f"Error loading default dataset: {e}")
        st.stop()

# If we reach here, 'uploaded' contains either:
# - A user-uploaded file (from sidebar), OR
# - The default dataset we just downloaded
# Either way, it will work with your load_and_build function

    # Show architecture explainer when no data loaded
    st.markdown('<p class="section-header">⚡ How It Works</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, title, body in [
        (c1, "🔤", "Layer 1 — TF-IDF Text",
         "Each product's brand, description, material and specs are merged into a 'content soup'. "
         "TF-IDF converts this into a 5,000-dimensional vector. Bigrams preserve phrases like "
         "'non padded' and 'zip closure'. Cosine similarity finds the closest products."),
        (c2, "🔢", "Layer 2 — Structured Features",
         "Eight binary/categorical flags are extracted from structured spec fields: padding, "
         "waterproofing, laptop compartment, cotton content, sustainability, occasion, "
         "category and price tier. Scaled and compared via cosine similarity."),
        (c3, "⭐", "Layer 3 — Bayesian Popularity",
         "A product with 5 stars from 2 reviews is less reliable than 4.3 stars from 500 reviews. "
         "The Bayesian formula pulls low-review products toward the global mean, "
         "preventing unknown outliers from dominating recommendations."),
    ]:
        with col:
            st.markdown(f"""
            <div class="product-card">
                <div style="font-size:2rem; margin-bottom:0.6rem;">{icon}</div>
                <div style="font-family:'Playfair Display',serif; font-size:1rem;
                            font-weight:700; color:#fff; margin-bottom:0.5rem;">{title}</div>
                <div style="font-size:0.82rem; color:#999; line-height:1.6;">{body}</div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

# ── Build model ───────────────────────────────────────────────────────────────
with st.spinner("🔧 Building recommendation engine... (first load only)"):
    try:
        df, text_sim, struct_sim, tfidf_mat, C, m = load_and_build(uploaded)
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

# Dataset stats for hero update
n_products  = len(df)
n_cats      = df["category"].nunique()
n_reviewed  = (df["total_reviews"] > 0).sum()
cold_pct    = round((df["total_reviews"] == 0).mean() * 100, 1)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍  Recommend", "📊  Analytics", "📐  Evaluate"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.8], gap="large")

    with col_left:
        st.markdown('<p class="section-header">🔎 Select a Product</p>',
                    unsafe_allow_html=True)

        # Category filter
        cats = sorted(df["category"].dropna().unique())
        sel_cat = st.selectbox("Filter by Category", ["All Categories"] + cats)

        pool = df if sel_cat == "All Categories" else df[df["category"] == sel_cat]

        # Search box
        search = st.text_input("🔍 Search product title", placeholder="e.g. Tommy Hilfiger backpack...")
        if search:
            pool = pool[pool["title"].str.contains(search, case=False, na=False)]

        if pool.empty:
            st.markdown('<div class="warn-box">No products found. Try a different search.</div>',
                        unsafe_allow_html=True)
            st.stop()

        # Product selector — show brand + title
        pool_display = pool.copy()
        pool_display["display"] = pool_display.apply(
            lambda r: f"{r['title']}  ·  ₹{int(r['final_price_clean']) if pd.notna(r['final_price_clean']) else '?'}", axis=1)

        sel_display = st.selectbox("Choose Product", pool_display["display"].tolist())
        sel_idx     = pool_display["display"].tolist().index(sel_display)
        product_id  = int(pool_display.iloc[sel_idx]["product_id"])

        # Get query product info
        qrow = df[df["product_id"] == product_id].iloc[0]
        price = qrow["final_price_clean"]
        rating = qrow["weighted_rating"]
        reviews = int(qrow["total_reviews"])

        # Display selected product card
        tier_label, tier_class = price_tier_label(price) if pd.notna(price) else ("?", "pill")
        st.markdown(f"""
        <div class="query-card" style="margin-top:1rem;">
            <div class="query-card-cat">{qrow['category'].upper().replace('-',' ')}</div>
            <div class="query-card-title">{qrow['title']}</div>
            <div class="stars">{stars_html(rating)} &nbsp;
                <span style="color:#888; font-size:0.78rem; font-family:'DM Sans',sans-serif;">
                {rating:.1f} ({reviews:,} reviews)</span>
            </div>
            <div class="metric-row">
                <div class="metric-chip">
                    <div class="metric-chip-val">₹{int(price) if pd.notna(price) else '?'}</div>
                    <div class="metric-chip-lbl">Price</div>
                </div>
                <div class="metric-chip">
                    <div class="metric-chip-val">{int(qrow['discount_pct'])}%</div>
                    <div class="metric-chip-lbl">Discount</div>
                </div>
                <div class="metric-chip">
                    <div class="metric-chip-val">{reviews:,}</div>
                    <div class="metric-chip-lbl">Reviews</div>
                </div>
            </div>
            <div style="margin-top:0.8rem;">
                <span class="{tier_class} pill">{tier_label}</span>
                {"<span class='pill pill-green'>✓ Reviewed</span>" if reviews > 0 else "<span class='pill'>New Arrival</span>"}
                {"<span class='pill pill-green'>🌿 Sustainable</span>" if qrow.get('is_sustainable',0)==1 else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Recommend button
        run = st.button("✨ Get Recommendations", use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">📦 Recommendations</p>',
                    unsafe_allow_html=True)

        if not run:
            st.markdown("""
            <div class="info-box" style="text-align:center; padding:2.5rem;">
                <div style="font-size:2.5rem; margin-bottom:0.8rem;">✨</div>
                <div style="font-size:0.95rem; color:#aaa;">
                    Select a product and click<br>
                    <b style="color:#FF385C;">Get Recommendations</b>
                    to find similar items.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Finding best matches..."):
                recs, _ = get_recommendations(
                    df, text_sim, struct_sim, product_id,
                    n=top_n, same_category=same_cat,
                    alpha=alpha, beta=beta)

            if recs.empty:
                st.markdown("""
                <div class="warn-box">
                ⚠️  Not enough products in this category for recommendations.
                Try disabling <b>Same Category Only</b> in the sidebar.
                </div>""", unsafe_allow_html=True)
            else:
                max_score = recs["hybrid_score"].max()

                for i, row in recs.iterrows():
                    price_r = row["final_price_clean"]
                    rating_r = row["weighted_rating"]
                    reviews_r = int(row["total_reviews"])
                    score = row["hybrid_score"]
                    content_s = row["content_sim"]
                    pop_s = row["pop_score_val"]
                    score_pct = int((score / max_score) * 100) if max_score > 0 else 0
                    tier_l, tier_c = price_tier_label(price_r) if pd.notna(price_r) else ("?","pill")

                    st.markdown(f"""
                    <div class="product-card">
                        <div style="display:flex; align-items:flex-start; gap:0.6rem;">
                            <span class="rank-badge">{i}</span>
                            <div style="flex:1;">
                                <div class="product-title">{row['title']}</div>
                                <div class="product-meta">
                                    {row['category'].replace('-',' ').title()}
                                    &nbsp;·&nbsp;
                                    <span class="stars" style="font-size:0.78rem;">{stars_html(rating_r)}</span>
                                    &nbsp;{rating_r:.1f}&nbsp;({reviews_r:,} reviews)
                                </div>
                                <div style="margin-top:0.4rem;">
                                    <span class="{tier_c} pill">₹{int(price_r) if pd.notna(price_r) else '?'}</span>
                                    <span class="{tier_c} pill">{tier_l}</span>
                                    <span class="pill pill-green">sim {content_s:.2f}</span>
                                    <span class="pill pill-gold">pop {pop_s:.2f}</span>
                                </div>
                                <div class="score-bar-wrap">
                                    <div class="score-label">
                                        <span>Hybrid Match Score</span>
                                        <span style="color:#FF385C; font-weight:600;">{score:.3f}</span>
                                    </div>
                                    <div class="score-bar-bg">
                                        <div class="score-bar-fill" style="width:{score_pct}%;"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    # Dataset overview metrics
    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl in [
        (m1, f"{n_products:,}", "Total Products"),
        (m2, f"{n_cats}", "Categories"),
        (m3, f"{n_reviewed:,}", "Reviewed Products"),
        (m4, f"{cold_pct}%", "Cold-Start Rate"),
    ]:
        with col:
            st.metric(lbl, val)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Row 1: Category bar + Rating histogram ────────────────────────────
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<p class="section-header">📦 Top 20 Categories</p>',
                    unsafe_allow_html=True)
        top20 = df["category"].value_counts().head(20).reset_index()
        top20.columns = ["Category", "Count"]
        fig = px.bar(top20, x="Count", y="Category", orientation="h",
                     color="Count",
                     color_continuous_scale=["#2d0f18","#FF385C"],
                     template="plotly_dark")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending",
                       tickfont=dict(size=11)),
            height=420,
            font=dict(family="DM Sans"),
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">⭐ Rating Distribution</p>',
                    unsafe_allow_html=True)
        rated = df[df["weighted_rating"] > 0]["weighted_rating"]
        fig = px.histogram(rated, nbins=30,
                           color_discrete_sequence=["#FF385C"],
                           template="plotly_dark")
        fig.add_vline(x=float(rated.mean()), line_dash="dash",
                      line_color="#FFD700",
                      annotation_text=f"Mean {rated.mean():.2f}",
                      annotation_font_color="#FFD700")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            xaxis_title="Weighted Rating",
            yaxis_title="Frequency",
            height=420,
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Price dist + Review vs Rating scatter ───────────────────────
    col_c, col_d = st.columns(2, gap="large")

    with col_c:
        st.markdown('<p class="section-header">💰 Price Distribution</p>',
                    unsafe_allow_html=True)
        fig = px.histogram(df["final_price_clean"].dropna(),
                           nbins=40,
                           color_discrete_sequence=["#FF6B81"],
                           template="plotly_dark")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            xaxis_title="Final Price (₹)",
            yaxis_title="Count",
            height=350,
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown('<p class="section-header">🔍 Rating vs Review Volume</p>',
                    unsafe_allow_html=True)
        sdf = df[(df["weighted_rating"]>0) & (df["total_reviews"]>0)].copy()
        sdf["log_reviews"] = np.log1p(sdf["total_reviews"])
        fig = px.scatter(sdf, x="log_reviews", y="weighted_rating",
                         color="weighted_rating",
                         color_continuous_scale=["#c0003a","#FF385C","#FFD700"],
                         hover_data={"title": True,
                                     "weighted_rating": ":.2f",
                                     "total_reviews": True,
                                     "log_reviews": False},
                         template="plotly_dark", opacity=0.6,
                         size_max=6)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="log(1 + Reviews)",
            yaxis_title="Weighted Rating",
            coloraxis_showscale=False,
            height=350,
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Discount dist + Bayesian score dist ─────────────────────────
    col_e, col_f = st.columns(2, gap="large")

    with col_e:
        st.markdown('<p class="section-header">🏷️ Discount Distribution</p>',
                    unsafe_allow_html=True)
        disc = df[df["discount_pct"] > 0]["discount_pct"]
        fig = px.histogram(disc, nbins=25,
                           color_discrete_sequence=["#FFD700"],
                           template="plotly_dark")
        fig.add_vline(x=80, line_dash="dash", line_color="#FF385C",
                      annotation_text="80% threshold",
                      annotation_font_color="#FF385C")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            xaxis_title="Discount %",
            yaxis_title="Count",
            height=350,
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_f:
        st.markdown('<p class="section-header">🎯 Bayesian Popularity Score</p>',
                    unsafe_allow_html=True)
        fig = px.histogram(df["pop_score"], nbins=35,
                           color_discrete_sequence=["#00C853"],
                           template="plotly_dark")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            xaxis_title="Normalised Popularity Score",
            yaxis_title="Count",
            height=350,
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Category coverage table ────────────────────────────────────────────
    st.markdown('<p class="section-header">📋 Category Coverage Summary</p>',
                unsafe_allow_html=True)
    cat_summary = (df.groupby("category")
                   .agg(Products=("product_id","count"),
                        Avg_Rating=("weighted_rating","mean"),
                        Avg_Price=("final_price_clean","mean"),
                        Total_Reviews=("total_reviews","sum"),
                        Cold_Start_Pct=("total_reviews", lambda x: (x==0).mean()*100))
                   .reset_index()
                   .sort_values("Products", ascending=False))
    cat_summary["Avg_Rating"] = cat_summary["Avg_Rating"].round(2)
    cat_summary["Avg_Price"]  = cat_summary["Avg_Price"].round(0).astype("Int64")
    cat_summary["Cold_Start_Pct"] = cat_summary["Cold_Start_Pct"].round(1)
    cat_summary["Coverage"] = cat_summary["Products"].apply(
        lambda x: "✅ Good" if x >= 5 else ("⚠️ Limited" if x >= 3 else "❌ Sparse"))
    cat_summary.columns = ["Category","Products","Avg Rating","Avg Price (₹)",
                           "Total Reviews","Cold Start %","Coverage"]
    st.dataframe(cat_summary, use_container_width=True, height=320,
                 column_config={
                     "Avg Rating": st.column_config.ProgressColumn(
                         "Avg Rating", min_value=0, max_value=5, format="%.2f"),
                     "Products": st.column_config.NumberColumn("Products", format="%d"),
                 })

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    st.markdown('<p class="section-header">📐 Evaluation Dashboard</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Since there is no ground-truth user interaction data, we use four
    <b>proxy metrics</b> to measure recommendation quality. These evaluate
    relevance, diversity, quality uplift and coverage.
    </div>
    """, unsafe_allow_html=True)

    # Pick showcase products (most reviewed per category)
    showcase_cats = (df[df["total_reviews"] >= 5]
                     .sort_values("total_reviews", ascending=False)
                     .drop_duplicates("category")
                     .head(8)["product_id"].tolist())

    if not showcase_cats:
        showcase_cats = df["product_id"].head(8).tolist()

    global_bay = df["bayesian"].mean()
    eval_rows  = []

    for pid in showcase_cats:
        recs, qrow = get_recommendations(
            df, text_sim, struct_sim, pid,
            n=5, same_category=True, alpha=alpha, beta=beta)
        if recs.empty or qrow is None:
            continue

        rec_pids = recs["product_id"].values
        cat_prec = (recs["category"] == qrow["category"]).mean()

        # Intra-list diversity
        idx_list = [list(df["product_id"].values).index(i)
                    for i in rec_pids if i in df["product_id"].values]
        if len(idx_list) >= 2:
            pw    = cosine_similarity(tfidf_mat[idx_list])
            upper = pw[np.triu_indices_from(pw, k=1)]
            intra = round(float(upper.mean()), 3)
        else:
            intra = None

        avg_bay = df[df["product_id"].isin(rec_pids)]["bayesian"].mean()
        lift    = avg_bay - global_bay

        eval_rows.append({
            "Product"          : str(qrow["title"])[:38] + "...",
            "Category"         : str(qrow["category"]),
            "Cat. Precision"   : float(cat_prec),
            "Intra-List Sim"   : intra,
            "Avg Rec Rating"   : round(float(avg_bay), 3),
            "Rating Lift"      : round(float(lift), 3),
            "Avg Hybrid Score" : round(float(recs["hybrid_score"].mean()), 4),
        })

    if eval_rows:
        eval_df = pd.DataFrame(eval_rows)

        # ── Summary KPIs ──────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        valid_intra = [r for r in eval_rows if r["Intra-List Sim"] is not None]
        for col, val, lbl, help_txt in [
            (k1, f"{eval_df['Cat. Precision'].mean():.0%}",
             "Avg Category Precision",
             "Fraction of recs in same category"),
            (k2, f"{np.mean([r['Intra-List Sim'] for r in valid_intra]):.3f}" if valid_intra else "N/A",
             "Avg Intra-List Similarity",
             "Diversity among recommendations (lower = more diverse)"),
            (k3, f"{eval_df['Avg Rec Rating'].mean():.3f}",
             "Avg Recommended Rating",
             "Mean Bayesian score of recommended products"),
            (k4, f"{eval_df['Rating Lift'].mean():+.3f}",
             "Avg Rating Lift",
             "How much above global average recommended items are"),
        ]:
            with col:
                st.metric(lbl, val, help=help_txt)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Eval table ────────────────────────────────────────────────────
        st.markdown("#### Detailed Metrics per Query Product")

        display_df = eval_df.copy()
        display_df["Cat. Precision"] = display_df["Cat. Precision"].apply(
            lambda x: f"{x:.0%}")
        display_df["Intra-List Sim"] = display_df["Intra-List Sim"].apply(
            lambda x: f"{x:.3f}" if x is not None else "N/A")
        display_df["Rating Lift"] = display_df["Rating Lift"].apply(
            lambda x: f"+{x:.3f}" if x >= 0 else f"{x:.3f}")

        st.dataframe(display_df, use_container_width=True, height=320,
                     column_config={
                         "Avg Hybrid Score": st.column_config.ProgressColumn(
                             "Avg Hybrid Score", min_value=0, max_value=1, format="%.4f"),
                         "Avg Rec Rating": st.column_config.ProgressColumn(
                             "Avg Rec Rating", min_value=0, max_value=5, format="%.3f"),
                     })

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Ablation chart ────────────────────────────────────────────────
        st.markdown("#### 🔬 Ablation Study — Hybrid vs Content-Only vs Popularity-Only")
        st.markdown("""
        <div class="info-box">
        For the top reviewed product in the dataset, we compare all three modes
        to show the hybrid approach's advantage.
        </div>
        """, unsafe_allow_html=True)

        abl_pid = int(df[df["total_reviews"] > 0]
                      .sort_values("total_reviews", ascending=False)
                      .iloc[0]["product_id"])
        abl_rows = []

        for mode_name, b in [("Content Only", 1.0),
                              ("Popularity Only", 0.0),
                              ("Hybrid (β=0.8)", 0.8)]:
            recs_abl, _ = get_recommendations(
                df, text_sim, struct_sim, abl_pid,
                n=5, same_category=True, alpha=alpha, beta=b)
            if recs_abl.empty:
                continue
            rec_pids_a = recs_abl["product_id"].values
            avg_bay_a  = df[df["product_id"].isin(rec_pids_a)]["bayesian"].mean()
            idx_a = [list(df["product_id"].values).index(i)
                     for i in rec_pids_a if i in df["product_id"].values]
            intra_a = 0.0
            if len(idx_a) >= 2:
                pw_a = cosine_similarity(tfidf_mat[idx_a])
                upper_a = pw_a[np.triu_indices_from(pw_a, k=1)]
                intra_a = float(upper_a.mean())
            abl_rows.append({
                "Mode"           : mode_name,
                "Avg Content Sim": round(float(recs_abl["content_sim"].mean()), 4),
                "Avg Pop Score"  : round(float(recs_abl["pop_score_val"].mean()), 4),
                "Intra-List Sim" : round(intra_a, 4),
                "Avg Rec Rating" : round(float(avg_bay_a), 4),
            })

        if abl_rows:
            abl_df = pd.DataFrame(abl_rows)

            # Radar / grouped bar comparison
            metrics = ["Avg Content Sim","Avg Pop Score","Avg Rec Rating"]
            fig = go.Figure()
            colors = ["#FF385C","#FFD700","#00C853"]
            for row_i, (_, row_a) in enumerate(abl_df.iterrows()):
                fig.add_trace(go.Bar(
                    name=row_a["Mode"],
                    x=metrics,
                    y=[row_a[m] for m in metrics],
                    marker_color=colors[row_i],
                    marker_line_width=0,
                    opacity=0.85,
                ))

            fig.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="#ccc", size=11),
                            bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis=dict(gridcolor="#2E2E2E"),
                xaxis=dict(tickfont=dict(color="#ccc")),
                font=dict(family="DM Sans", color="#ccc"),
                height=350,
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(abl_df, use_container_width=True)

        # ── Coverage bar ──────────────────────────────────────────────────
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown("#### 📊 Category Coverage")

        cat_sizes    = df.groupby("category").size()
        good_cats    = (cat_sizes >= 5).sum()
        limited_cats = ((cat_sizes >= 3) & (cat_sizes < 5)).sum()
        sparse_cats  = (cat_sizes < 3).sum()

        c_cov1, c_cov2, c_cov3 = st.columns(3)
        for col, val, lbl, col_color in [
            (c_cov1, good_cats,    "✅ Good (≥5 products)",  "#00C853"),
            (c_cov2, limited_cats, "⚠️ Limited (3–4)",       "#FFD700"),
            (c_cov3, sparse_cats,  "❌ Sparse (<3)",          "#FF385C"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-chip" style="width:100%; border-color:{col_color}33;
                            background: {col_color}11;">
                    <div class="metric-chip-val" style="color:{col_color}; font-size:1.8rem;">
                        {val}</div>
                    <div class="metric-chip-lbl">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="warn-box">
        Not enough reviewed products to run evaluation. 
        Please ensure the dataset has products with ratings.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<hr class="custom-divider">
<div style="text-align:center; color:#555; font-size:0.75rem; padding-bottom:1rem;">
    Myntra AI Recommender &nbsp;·&nbsp;
    Hybrid TF-IDF + Structured Features + Bayesian Popularity &nbsp;·&nbsp;
    Built with Streamlit &amp; scikit-learn
</div>
""", unsafe_allow_html=True)

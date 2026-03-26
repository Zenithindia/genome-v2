"""
============================================================================
GENOME VISIBILITY PREDICTOR - STREAMLIT APP
Step 3 Upgrade:
- Competitor-aware analysis
- Runtime relative/rank feature computation
- User page vs competitor comparison
- Ranking table across analyzed pages
- Uses existing frozen models only (no retraining)
============================================================================
"""

import re
import time
import pickle
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="GENOME Visibility Predictor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    padding: 1rem 0;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
}
.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
}
.danger-box {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
}
.info-box {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
}
.metric-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.5rem 0;
}
.small-note {
    color: #666;
    font-size: 0.9rem;
}
.rank-good {
    color: #0f5132;
    font-weight: bold;
}
.rank-mid {
    color: #856404;
    font-weight: bold;
}
.rank-bad {
    color: #842029;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "else", "for", "to", "of", "in", "on",
    "at", "by", "with", "from", "as", "it", "this", "that", "these", "those",
    "you", "your", "we", "our", "they", "their", "he", "she", "his", "her",
    "what", "which", "who", "whom", "where", "when", "why", "how", "can",
    "could", "should", "would", "will", "may", "might", "do", "does", "did",
    "have", "has", "had", "not", "no", "yes", "than", "into", "about", "over",
    "under", "up", "down", "out", "more", "most", "very", "also"
}

ORIGINALITY_MARKERS = {
    "i built", "we built", "i created", "we created", "my project", "our project",
    "case study", "results", "benchmark", "lessons learned", "trade-off", "tradeoff",
    "implementation", "architecture", "challenge", "solution", "experiment",
    "evaluation", "accuracy", "precision", "recall", "f1", "latency", "deployment"
}

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load trained models, standards, and metadata"""
    try:
        model_dir = Path("model_outputs")

        with open(model_dir / "stage1_classifier.pkl", "rb") as f:
            clf = pickle.load(f)

        with open(model_dir / "stage2_regressor.pkl", "rb") as f:
            reg = pickle.load(f)

        with open(model_dir / "stage1_features.pkl", "rb") as f:
            features_s1 = pickle.load(f)

        with open(model_dir / "stage2_features.pkl", "rb") as f:
            features_s2 = pickle.load(f)

        with open(model_dir / "model_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        with open(model_dir / "feature_standards.pkl", "rb") as f:
            standards = pickle.load(f)

        with open(model_dir / "feature_thresholds.pkl", "rb") as f:
            thresholds = pickle.load(f)

        with open(model_dir / "core_extractable_features.pkl", "rb") as f:
            core_features = pickle.load(f)

        return clf, reg, features_s1, features_s2, metadata, standards, thresholds, core_features

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("Please ensure model_outputs/ directory contains all required files")
        st.stop()

# ============================================================================
# GENERAL HELPERS
# ============================================================================

def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))

def safe_div(a, b, default=0.0):
    return a / b if b else default

def normalize_whitespace(text):
    return re.sub(r"\s+", " ", (text or "")).strip()

def normalize_url(url):
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url

def parse_url_list(text_block):
    lines = [normalize_url(x.strip()) for x in (text_block or "").splitlines()]
    lines = [x for x in lines if x]
    deduped = []
    seen = set()
    for u in lines:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped

def short_domain(url):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return url

# ============================================================================
# TEXT / HTML UTILITIES
# ============================================================================

def tokenize(text):
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def jaccard_similarity(tokens_a, tokens_b):
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def containment_ratio(query_tokens, doc_tokens):
    q = set(query_tokens)
    d = set(doc_tokens)
    if not q:
        return 0.0
    return len(q & d) / len(q)

def phrase_match_score(query, text):
    q = normalize_whitespace(query).lower()
    t = normalize_whitespace(text).lower()
    if not q or not t:
        return 0.0
    if q in t:
        return 1.0

    q_tokens = [tok for tok in q.split() if tok not in STOPWORDS]
    if not q_tokens:
        return 0.0

    joined_pairs = [" ".join(q_tokens[i:i+2]) for i in range(len(q_tokens) - 1)]
    pair_hits = sum(1 for pair in joined_pairs if pair and pair in t)
    token_hits = sum(1 for tok in q_tokens if tok in t)

    pair_score = safe_div(pair_hits, max(len(joined_pairs), 1))
    token_score = safe_div(token_hits, len(q_tokens))
    return clamp(0.65 * pair_score + 0.35 * token_score)

def lexical_diversity_score(tokens):
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    return clamp((unique_ratio - 0.15) / 0.55)

def repetition_penalty(tokens):
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    repeated_mass = sum(c - 1 for c in counts.values() if c > 1)
    return clamp(repeated_mass / max(len(tokens), 1))

def originality_marker_score(text):
    t = (text or "").lower()
    hits = sum(1 for marker in ORIGINALITY_MARKERS if marker in t)
    return clamp(hits / 6)

def detect_query_type(query):
    q = (query or "").lower().strip()
    list_words = ["best", "top", "list", "examples", "types", "tools", "projects", "resources"]
    opinion_words = ["opinion", "review", "worth", "good", "better", "best", "vs", "compare"]
    is_list = int(any(w in q for w in list_words))
    is_opinion = int(any(w in q for w in opinion_words))
    is_other = int(not (is_list or is_opinion))
    return is_list, is_opinion, is_other

def get_meta_description(soup):
    for attr_name, attr_value in [("name", "description"), ("property", "og:description")]:
        tag = soup.find("meta", attrs={attr_name: attr_value})
        if tag and tag.get("content"):
            return normalize_whitespace(tag["content"])
    return ""

def get_title_text(soup):
    if soup.title and soup.title.string:
        return normalize_whitespace(soup.title.string)
    h1 = soup.find("h1")
    if h1:
        return normalize_whitespace(h1.get_text(" ", strip=True))
    return ""

def extract_headings(soup):
    headings = []
    for tag_name in ["h1", "h2", "h3"]:
        for tag in soup.find_all(tag_name):
            txt = normalize_whitespace(tag.get_text(" ", strip=True))
            if txt:
                headings.append(txt)
    return headings

def get_visible_text(soup):
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    main_candidates = []
    for selector in ["main", "article"]:
        main_candidates.extend(soup.find_all(selector))

    content_chunks = []
    if main_candidates:
        for elem in main_candidates:
            txt = normalize_whitespace(elem.get_text(" ", strip=True))
            if txt:
                content_chunks.append(txt)

    if not content_chunks:
        txt = normalize_whitespace(soup.get_text(" ", strip=True))
        content_chunks.append(txt)

    return normalize_whitespace(" ".join(content_chunks))

# ============================================================================
# FEATURE ENGINEERING HELPERS
# ============================================================================

def compute_relevance(query, title, headings, body_text):
    q_tokens = tokenize(query)
    title_tokens = tokenize(title)
    heading_tokens = tokenize(" ".join(headings))
    body_tokens = tokenize(body_text[:12000])

    title_containment = containment_ratio(q_tokens, title_tokens)
    heading_containment = containment_ratio(q_tokens, heading_tokens)
    body_containment = containment_ratio(q_tokens, body_tokens)

    title_jaccard = jaccard_similarity(q_tokens, title_tokens)
    heading_jaccard = jaccard_similarity(q_tokens, heading_tokens)
    body_jaccard = jaccard_similarity(q_tokens, body_tokens)

    title_phrase = phrase_match_score(query, title)
    heading_phrase = phrase_match_score(query, " ".join(headings[:10]))
    body_phrase = phrase_match_score(query, body_text[:3000])

    score = (
        0.30 * title_containment +
        0.18 * heading_containment +
        0.16 * body_containment +
        0.10 * title_jaccard +
        0.08 * heading_jaccard +
        0.08 * body_jaccard +
        0.06 * title_phrase +
        0.02 * heading_phrase +
        0.02 * body_phrase
    )
    return clamp(score)

def compute_diversity(structure_stats):
    heading_score = clamp(structure_stats["heading_count"] / 12)
    section_variety = clamp(structure_stats["distinct_heading_types"] / 3)
    list_score = clamp(structure_stats["list_count"] / 6)
    image_score = clamp(structure_stats["image_count"] / 6)
    table_score = clamp(structure_stats["table_count"] / 2)
    code_score = clamp(structure_stats["code_block_count"] / 3)
    paragraph_score = clamp(structure_stats["paragraph_count"] / 18)
    faq_score = clamp(structure_stats["faq_like_count"] / 4)

    score = (
        0.20 * heading_score +
        0.12 * section_variety +
        0.14 * list_score +
        0.14 * image_score +
        0.10 * table_score +
        0.10 * code_score +
        0.12 * paragraph_score +
        0.08 * faq_score
    )
    return clamp(score)

def compute_click_probability(query, title, meta_desc, relevance, diversity):
    q_tokens = tokenize(query)
    title_tokens = tokenize(title)
    meta_tokens = tokenize(meta_desc)

    title_overlap = containment_ratio(q_tokens, title_tokens)
    meta_overlap = containment_ratio(q_tokens, meta_tokens)

    title_len_words = len(title.split())
    meta_len_words = len(meta_desc.split())

    title_length_score = 1.0 if 6 <= title_len_words <= 14 else 0.7 if 4 <= title_len_words <= 18 else 0.45
    meta_presence_score = 1.0 if meta_desc else 0.35
    meta_length_score = 1.0 if 12 <= meta_len_words <= 28 else 0.7 if 8 <= meta_len_words <= 36 else 0.45
    intent_word_bonus = 1.0 if any(w in title.lower() for w in ["how", "best", "top", "guide", "portfolio", "project"]) else 0.55

    score = (
        0.28 * relevance +
        0.20 * title_overlap +
        0.12 * meta_overlap +
        0.12 * title_length_score +
        0.10 * meta_presence_score +
        0.08 * meta_length_score +
        0.05 * diversity +
        0.05 * intent_word_bonus
    )
    return clamp(score)

def compute_influence(domain, has_https, structure_stats, title, meta_desc):
    domain = (domain or "").lower()

    very_high_domains = [
        "wikipedia.org", "nih.gov", "nature.com", "science.org", ".gov", ".edu"
    ]
    medium_domains = [
        "medium.com", "forbes.com", "techcrunch.com", "nytimes.com",
        "github.io", "github.com", "substack.com", ".org"
    ]

    domain_score = 0.52
    if any(x in domain for x in very_high_domains):
        domain_score = 0.90
    elif any(x in domain for x in medium_domains):
        domain_score = 0.70

    https_bonus = 0.06 if has_https else 0.0
    metadata_bonus = 0.04 if title else 0.0
    metadata_bonus += 0.03 if meta_desc else 0.0
    structure_bonus = 0.04 if structure_stats.get("heading_count", 0) >= 4 else 0.0
    structure_bonus += 0.03 if structure_stats.get("paragraph_count", 0) >= 6 else 0.0

    return clamp(domain_score + https_bonus + metadata_bonus + structure_bonus)

def compute_uniqueness(title, headings, body_text, structure_stats):
    body_tokens = tokenize(body_text[:10000])
    title_tokens = tokenize(title)
    heading_tokens = tokenize(" ".join(headings[:20]))

    lex_score = lexical_diversity_score(body_tokens)
    rep_penalty = repetition_penalty(body_tokens)
    originality_score = originality_marker_score(f"{title} {' '.join(headings)} {body_text[:4000]}")

    number_density = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", body_text[:5000])) / max(len(body_tokens), 1)
    number_score = clamp(number_density * 20)

    title_specificity = clamp(len(set(title_tokens)) / 8) if title_tokens else 0.2
    heading_specificity = clamp(len(set(heading_tokens)) / 18) if heading_tokens else 0.2

    structure_bonus = 0.15 if structure_stats.get("table_count", 0) > 0 or structure_stats.get("code_block_count", 0) > 0 else 0.0

    score = (
        0.28 * lex_score +
        0.20 * originality_score +
        0.14 * number_score +
        0.14 * title_specificity +
        0.12 * heading_specificity +
        0.12 * structure_bonus
    ) - (0.22 * rep_penalty)

    return clamp(score)

def build_structure_stats(soup, body_text):
    headings = extract_headings(soup)
    h1_count = len(soup.find_all("h1"))
    h2_count = len(soup.find_all("h2"))
    h3_count = len(soup.find_all("h3"))
    paragraph_count = len([p for p in soup.find_all("p") if normalize_whitespace(p.get_text(" ", strip=True))])
    list_count = len(soup.find_all(["ul", "ol"]))
    image_count = len([img for img in soup.find_all("img") if img.get("src")])
    table_count = len(soup.find_all("table"))
    code_block_count = len(soup.find_all(["pre", "code"]))
    blockquote_count = len(soup.find_all("blockquote"))
    faq_like_count = len(re.findall(r"\?", body_text[:8000]))

    distinct_heading_types = sum(int(x > 0) for x in [h1_count, h2_count, h3_count])

    return {
        "heading_count": len(headings),
        "distinct_heading_types": distinct_heading_types,
        "h1_count": h1_count,
        "h2_count": h2_count,
        "h3_count": h3_count,
        "paragraph_count": paragraph_count,
        "list_count": list_count,
        "image_count": image_count,
        "table_count": table_count,
        "code_block_count": code_block_count,
        "blockquote_count": blockquote_count,
        "faq_like_count": faq_like_count
    }

# ============================================================================
# FEATURE PROVENANCE / RELIABILITY
# ============================================================================

def get_feature_provenance():
    return {
        "Relevance": {
            "source": "Estimated from query overlap with title, headings, and body text",
            "reliability": "Medium-High"
        },
        "Influence": {
            "source": "Heuristic estimate from domain pattern, HTTPS, and page structure",
            "reliability": "Medium"
        },
        "Uniqueness": {
            "source": "Proxy from lexical diversity, repetition penalty, originality markers, and structure",
            "reliability": "Medium"
        },
        "Click_Probability": {
            "source": "Heuristic from title/meta alignment, relevance, and structure",
            "reliability": "Medium"
        },
        "Diversity": {
            "source": "Estimated from headings, lists, images, tables, code blocks, and section variety",
            "reliability": "Medium-High"
        },
        "WC": {
            "source": "Extracted from fetched page text",
            "reliability": "High"
        },
        "Subjective_Position": {
            "source": "Computed from predicted PAWC ranking within analyzed group",
            "reliability": "Medium"
        },
        "Subjective_Count": {
            "source": "Computed from analyzed group size",
            "reliability": "High"
        },
        "WC_rel": {
            "source": "Computed relative to average word count of analyzed group",
            "reliability": "High"
        },
        "query_length": {
            "source": "Directly computed from input query",
            "reliability": "High"
        },
        "query_type_list": {
            "source": "Estimated from input query wording",
            "reliability": "Medium"
        },
        "query_type_opinion": {
            "source": "Estimated from input query wording",
            "reliability": "Medium"
        },
        "query_type_other": {
            "source": "Estimated from input query wording",
            "reliability": "Medium"
        },
        "num_sources": {
            "source": "Computed from analyzed group size",
            "reliability": "High"
        },
        "is_suggested_source": {
            "source": "Default assumed value",
            "reliability": "Low"
        },
        "domain_freq": {
            "source": "Computed from domain repetition in analyzed group",
            "reliability": "High"
        },
        "avg_PAWC_source": {
            "source": "Computed from analyzed group predictions",
            "reliability": "Medium"
        },
        "Influence_x_Position": {
            "source": "Derived from influence and computed position",
            "reliability": "Medium"
        },
        "Relevance_x_Uniqueness": {
            "source": "Derived from estimated relevance and uniqueness",
            "reliability": "Medium"
        },
        "Influence_rank": {
            "source": "Computed rank within analyzed group",
            "reliability": "High"
        },
        "Relevance_rank": {
            "source": "Computed rank within analyzed group",
            "reliability": "High"
        },
        "Uniqueness_rank": {
            "source": "Computed rank within analyzed group",
            "reliability": "High"
        },
        "Click_Prob_rank": {
            "source": "Computed rank within analyzed group",
            "reliability": "High"
        },
        "Diversity_rank": {
            "source": "Computed rank within analyzed group",
            "reliability": "High"
        },
        "Quality_Score": {
            "source": "Derived composite score",
            "reliability": "Medium"
        },
        "Position_weighted_Influence": {
            "source": "Derived from influence and computed position",
            "reliability": "Medium"
        },
        "Click_Prob_rel": {
            "source": "Computed relative to analyzed group average",
            "reliability": "High"
        },
        "Source_Density": {
            "source": "Derived from analyzed group size",
            "reliability": "High"
        },
        "Domain_Popularity": {
            "source": "Derived from computed domain frequency and average source PAWC",
            "reliability": "Medium"
        },
        "PAWC_rank": {
            "source": "Computed from predicted PAWC ranking in analyzed group",
            "reliability": "Medium"
        },
        "PAWC_pct": {
            "source": "Computed percentile in analyzed group",
            "reliability": "Medium"
        },
        "WC_x_Relevance": {
            "source": "Derived from extracted word count and estimated relevance",
            "reliability": "Medium-High"
        }
    }

# ============================================================================
# URL FEATURE EXTRACTION
# ============================================================================

def extract_page_features(url, query):
    """
    Extract single-page base features.
    Group/relative features are filled later in competitor-aware mode.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        has_https = parsed.scheme.lower() == "https"

        fetch_status = {
            "content_fetched": False,
            "fallback_used": False,
            "domain": domain
        }

        try:
            response = requests.get(
                url,
                timeout=12,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            title = get_title_text(soup)
            meta_desc = get_meta_description(soup)
            headings = extract_headings(soup)
            body_text = get_visible_text(soup)
            structure_stats = build_structure_stats(soup, body_text)

            body_tokens = tokenize(body_text)
            word_count = len(body_tokens)

            if word_count < 30:
                raise ValueError("Very little extractable content found on page")

            relevance = compute_relevance(query, title, headings, body_text)
            diversity = compute_diversity(structure_stats)
            influence = compute_influence(domain, has_https, structure_stats, title, meta_desc)
            uniqueness = compute_uniqueness(title, headings, body_text, structure_stats)
            click_probability = compute_click_probability(query, title, meta_desc, relevance, diversity)

            query_type_list, query_type_opinion, query_type_other = detect_query_type(query)

            base_features = {
                "Relevance": relevance,
                "Influence": influence,
                "Uniqueness": uniqueness,
                "Click_Probability": click_probability,
                "Diversity": diversity,
                "WC": float(word_count),
                "query_length": float(len(query)),
                "query_type_list": float(query_type_list),
                "query_type_opinion": float(query_type_opinion),
                "query_type_other": float(query_type_other),
                "is_suggested_source": 0.0,
            }

            diagnostics = {
                "url": url,
                "domain": domain,
                "title": title,
                "meta_description": meta_desc,
                "headings_found": len(headings),
                "body_word_count": word_count,
                "structure_stats": structure_stats
            }

            fetch_status["content_fetched"] = True
            return base_features, fetch_status, diagnostics, None

        except Exception as e:
            fetch_status["fallback_used"] = True
            query_type_list, query_type_opinion, query_type_other = detect_query_type(query)

            base_features = {
                "Relevance": 0.55,
                "Influence": compute_influence(domain, has_https, {"heading_count": 0, "paragraph_count": 0}, "", ""),
                "Uniqueness": 0.50,
                "Click_Probability": 0.52,
                "Diversity": 0.42,
                "WC": 350.0,
                "query_length": float(len(query)),
                "query_type_list": float(query_type_list),
                "query_type_opinion": float(query_type_opinion),
                "query_type_other": float(query_type_other),
                "is_suggested_source": 0.0,
            }

            diagnostics = {
                "url": url,
                "domain": domain,
                "title": "",
                "meta_description": "",
                "headings_found": 0,
                "body_word_count": 350,
                "structure_stats": {}
            }

            return base_features, fetch_status, diagnostics, None

    except Exception as e:
        return None, None, None, str(e)

# ============================================================================
# COMPETITOR-AWARE FEATURE BUILDING
# ============================================================================

def add_runtime_competitive_features(page_rows):
    """
    page_rows: list of dicts, each must contain:
      - url
      - label
      - is_user_page
      - base_features
      - domain
    """
    if not page_rows:
        return page_rows

    wc_values = [row["base_features"]["WC"] for row in page_rows]
    click_values = [row["base_features"]["Click_Probability"] for row in page_rows]

    avg_wc = float(np.mean(wc_values)) if wc_values else 1.0
    avg_click = float(np.mean(click_values)) if click_values else 1.0

    domain_counts = Counter([row["domain"] for row in page_rows])
    num_sources = len(page_rows)

    # rank helper: higher value => better rank (1 = best)
    def compute_rank_map(values_by_idx):
        ordered = sorted(values_by_idx.items(), key=lambda x: x[1], reverse=True)
        return {idx: rank + 1 for rank, (idx, _) in enumerate(ordered)}

    influence_rank_map = compute_rank_map({i: row["base_features"]["Influence"] for i, row in enumerate(page_rows)})
    relevance_rank_map = compute_rank_map({i: row["base_features"]["Relevance"] for i, row in enumerate(page_rows)})
    uniqueness_rank_map = compute_rank_map({i: row["base_features"]["Uniqueness"] for i, row in enumerate(page_rows)})
    click_rank_map = compute_rank_map({i: row["base_features"]["Click_Probability"] for i, row in enumerate(page_rows)})
    diversity_rank_map = compute_rank_map({i: row["base_features"]["Diversity"] for i, row in enumerate(page_rows)})

    # Build features required for Stage 1
    for i, row in enumerate(page_rows):
        base = row["base_features"]

        features = dict(base)
        features["Subjective_Count"] = float(num_sources)
        features["WC_rel"] = float(safe_div(base["WC"], avg_wc, 1.0))
        features["num_sources"] = float(num_sources)
        features["domain_freq"] = float(domain_counts[row["domain"]])
        features["avg_PAWC_source"] = 50.0  # provisional before first prediction pass

        features["Influence_rank"] = float(influence_rank_map[i])
        features["Relevance_rank"] = float(relevance_rank_map[i])
        features["Uniqueness_rank"] = float(uniqueness_rank_map[i])
        features["Click_Prob_rank"] = float(click_rank_map[i])
        features["Diversity_rank"] = float(diversity_rank_map[i])

        # temporary placeholders before final ranking pass
        features["Subjective_Position"] = float(i + 1)
        features["Influence_x_Position"] = float(base["Influence"] * features["Subjective_Position"])
        features["Relevance_x_Uniqueness"] = float(base["Relevance"] * base["Uniqueness"])
        features["Quality_Score"] = float(
            base["Relevance"] * 0.38 +
            base["Influence"] * 0.24 +
            base["Uniqueness"] * 0.20 +
            base["Diversity"] * 0.10 +
            base["Click_Probability"] * 0.08
        )
        features["Position_weighted_Influence"] = float(base["Influence"] / (features["Subjective_Position"] + 1))
        features["Click_Prob_rel"] = float(safe_div(base["Click_Probability"], avg_click, 1.0))
        features["Source_Density"] = float(1 / (num_sources + 1))
        features["Domain_Popularity"] = float(features["domain_freq"] * features["avg_PAWC_source"])
        features["PAWC_rank"] = float(i + 1)
        features["PAWC_pct"] = float((num_sources - i) / max(num_sources, 1))
        features["WC_x_Relevance"] = float(base["WC"] * base["Relevance"])

        row["features"] = features

    return page_rows

def predict_stage_outputs(features_dict, clf, reg, features_s1, features_s2):
    """
    First pass prediction with existing features.
    """
    X1 = np.array([features_dict[f] for f in features_s1]).reshape(1, -1)
    is_visible = bool(clf.predict(X1)[0])
    visibility_prob = float(clf.predict_proba(X1)[0][1])

    pawc_score = 0.0
    if is_visible:
        X2 = np.array([features_dict[f] for f in features_s2]).reshape(1, -1)
        log_pawc = reg.predict(X2)[0]
        pawc_score = float(np.expm1(log_pawc))

    return is_visible, visibility_prob, pawc_score

def finalize_group_features_with_predictions(page_rows):
    """
    After an initial prediction pass, use predicted PAWC to compute:
    - Subjective_Position
    - avg_PAWC_source
    - Domain_Popularity
    - PAWC_rank
    - PAWC_pct
    Then refresh dependent features.
    """
    if not page_rows:
        return page_rows

    # Sort by predicted PAWC desc, then stage1 prob desc
    sorted_rows = sorted(
        page_rows,
        key=lambda r: (r["predicted_pawc"], r["visibility_prob"]),
        reverse=True
    )

    avg_pawc = float(np.mean([r["predicted_pawc"] for r in page_rows])) if page_rows else 0.0

    for rank, row in enumerate(sorted_rows, start=1):
        row["predicted_rank"] = rank

    # push rank back to all rows
    rank_lookup = {row["url"]: row["predicted_rank"] for row in sorted_rows}

    for row in page_rows:
        row["features"]["Subjective_Position"] = float(rank_lookup[row["url"]])
        row["features"]["avg_PAWC_source"] = float(avg_pawc)
        row["features"]["Influence_x_Position"] = float(
            row["features"]["Influence"] * row["features"]["Subjective_Position"]
        )
        row["features"]["Position_weighted_Influence"] = float(
            row["features"]["Influence"] / (row["features"]["Subjective_Position"] + 1)
        )
        row["features"]["Domain_Popularity"] = float(
            row["features"]["domain_freq"] * row["features"]["avg_PAWC_source"]
        )
        row["features"]["PAWC_rank"] = float(rank_lookup[row["url"]])
        total = len(page_rows)
        row["features"]["PAWC_pct"] = float((total - rank_lookup[row["url"]] + 1) / max(total, 1))

    return page_rows

def run_group_predictions(page_rows, clf, reg, features_s1, features_s2):
    """
    Two-pass approach:
    1) build runtime competitive features
    2) first prediction pass
    3) update ranking-based fields using predictions
    4) final prediction pass
    """
    page_rows = add_runtime_competitive_features(page_rows)

    for row in page_rows:
        is_visible, visibility_prob, pawc_score = predict_stage_outputs(
            row["features"], clf, reg, features_s1, features_s2
        )
        row["is_visible"] = is_visible
        row["visibility_prob"] = visibility_prob
        row["predicted_pawc"] = pawc_score

    page_rows = finalize_group_features_with_predictions(page_rows)

    # final pass
    for row in page_rows:
        is_visible, visibility_prob, pawc_score = predict_stage_outputs(
            row["features"], clf, reg, features_s1, features_s2
        )
        row["is_visible"] = is_visible
        row["visibility_prob"] = visibility_prob
        row["predicted_pawc"] = pawc_score

    # final ranking
    sorted_rows = sorted(
        page_rows,
        key=lambda r: (r["predicted_pawc"], r["visibility_prob"]),
        reverse=True
    )
    for rank, row in enumerate(sorted_rows, start=1):
        row["final_rank"] = rank

    rank_lookup = {row["url"]: row["final_rank"] for row in sorted_rows}
    total = len(page_rows)

    for row in page_rows:
        row["features"]["Subjective_Position"] = float(rank_lookup[row["url"]])
        row["features"]["PAWC_rank"] = float(rank_lookup[row["url"]])
        row["features"]["PAWC_pct"] = float((total - rank_lookup[row["url"]] + 1) / max(total, 1))
        row["features"]["Influence_x_Position"] = float(
            row["features"]["Influence"] * row["features"]["Subjective_Position"]
        )
        row["features"]["Position_weighted_Influence"] = float(
            row["features"]["Influence"] / (row["features"]["Subjective_Position"] + 1)
        )

    return page_rows

# ============================================================================
# COMPARISON / INTERPRETATION
# ============================================================================

def compare_with_standards(features_dict, standards, thresholds, core_features):
    comparison = {}
    gaps = []

    for feature in core_features:
        if feature in features_dict and feature in standards:
            actual_val = features_dict[feature]

            if isinstance(standards[feature], dict):
                target_val = standards[feature].get("75th_percentile") or standards[feature].get("mean", 0)
            else:
                target_val = standards[feature]

            gap_pct = ((target_val - actual_val) / target_val) * 100 if target_val > 0 else 0

            comparison[feature] = {
                "actual": actual_val,
                "target": target_val,
                "gap_percentage": gap_pct,
                "meets_threshold": actual_val >= thresholds.get(feature, 0),
                "status": "✅" if gap_pct <= 10 else "⚠️" if gap_pct <= 30 else "❌"
            }

            if gap_pct > 10:
                gaps.append({
                    "feature": feature,
                    "actual": actual_val,
                    "target": target_val,
                    "gap": gap_pct
                })

    gaps.sort(key=lambda x: x["gap"], reverse=True)
    return comparison, gaps

def get_visibility_tier(probability, pawc_score, is_visible):
    if not is_visible:
        return "Low"
    if pawc_score is None:
        return "Low"
    if probability >= 0.90 and pawc_score >= 40:
        return "High"
    elif probability >= 0.70 and pawc_score >= 15:
        return "Medium"
    return "Low"

def build_interpretation_text(is_visible, visibility_prob, pawc_score, final_rank, total_pages):
    if not is_visible:
        return (
            f"The page is currently predicted as unlikely to enter the visibility set for this query. "
            f"It ranked {final_rank}/{total_pages} in the analyzed group."
        )

    if pawc_score is None:
        return (
            f"The page passed Stage 1 eligibility, but Stage 2 did not return a usable visibility-strength score."
        )

    if visibility_prob >= 0.90 and pawc_score >= 40:
        return (
            f"The page looks strongly eligible for visibility and ranks competitively in this analyzed group "
            f"at position {final_rank}/{total_pages}."
        )
    elif visibility_prob >= 0.70:
        return (
            f"The page is likely to be considered visible, but its final competitive strength is moderate. "
            f"Current rank: {final_rank}/{total_pages}."
        )
    else:
        return (
            f"The page passes eligibility, but the model confidence is not especially strong. "
            f"Current rank: {final_rank}/{total_pages}."
        )

def compare_user_vs_group(user_row, page_rows):
    """
    Returns a compact comparison dict vs competitor average.
    """
    competitor_rows = [r for r in page_rows if not r["is_user_page"]]
    if not competitor_rows:
        return {}

    compare_features = ["Relevance", "Influence", "Uniqueness", "Click_Probability", "Diversity", "WC", "WC_rel"]
    result = {}

    for feat in compare_features:
        comp_avg = float(np.mean([r["features"].get(feat, 0.0) for r in competitor_rows]))
        user_val = float(user_row["features"].get(feat, 0.0))
        diff = user_val - comp_avg
        result[feat] = {
            "user": user_val,
            "competitor_avg": comp_avg,
            "diff": diff
        }

    return result

def generate_recommendations(gaps, is_visible, user_row=None, page_rows=None):
    recommendations = []

    if not is_visible:
        recommendations.append(
            "🎯 **Primary Goal:** The page is predicted as NOT VISIBLE. Focus first on the largest feature gaps to improve eligibility."
        )
    else:
        recommendations.append(
            "✅ **Good News:** The page is predicted as VISIBLE, but the gaps below still limit competitive strength."
        )

    for gap in gaps[:4]:
        feature = gap["feature"]

        if feature == "Relevance":
            recommendations.append(
                f"📝 **Improve Relevance ({gap['gap']:.0f}% gap):** Put the query intent directly in the title, H1, and opening paragraph. Add a short answer section early in the page."
            )
        elif feature == "Influence":
            recommendations.append(
                f"🏆 **Build Influence ({gap['gap']:.0f}% gap):** Add stronger authority signals like GitHub links, citations, references, author credibility, and proof of real project work."
            )
        elif feature == "Uniqueness":
            recommendations.append(
                f"✨ **Increase Uniqueness ({gap['gap']:.0f}% gap):** Add original insights, metrics, design choices, lessons learned, benchmarks, or implementation details competitors may not have."
            )
        elif feature == "WC":
            recommendations.append(
                f"📄 **Expand Content ({gap['gap']:.0f}% gap):** Add deeper explanations, better project walkthroughs, examples, and supporting sections to move closer to the target content depth."
            )
        elif feature == "Click_Probability":
            recommendations.append(
                f"👆 **Improve Clickability ({gap['gap']:.0f}% gap):** Strengthen the title and meta description so they clearly match the query and make the page feel more worth opening."
            )
        elif feature == "Diversity":
            recommendations.append(
                f"🎨 **Add Diversity ({gap['gap']:.0f}% gap):** Include screenshots, diagrams, lists, code blocks, tables, FAQs, or multiple project sub-sections."
            )

    if user_row and page_rows:
        competitors = [r for r in page_rows if not r["is_user_page"]]
        if competitors:
            top_comp = sorted(competitors, key=lambda x: x["final_rank"])[0]
            if user_row["final_rank"] > top_comp["final_rank"]:
                recommendations.append(
                    f"📊 **Top Competitor Benchmark:** The best competitor in this analysis is `{top_comp['label']}` "
                    f"with PAWC `{top_comp['predicted_pawc']:.1f}`. Compare your page against it to prioritize the biggest feature gaps."
                )

    return recommendations

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown('<p class="main-header">🔍 GENOME Visibility Predictor</p>', unsafe_allow_html=True)
    st.markdown("### Analyze your page against competitors using the existing trained GENOME models")

    with st.spinner("Loading models and standards..."):
        clf, reg, features_s1, features_s2, metadata, standards, thresholds, core_features = load_models()

    provenance = get_feature_provenance()

    with st.sidebar:
        st.header("📊 Model Information")
        st.metric("Visibility Threshold", f"{metadata['visibility_threshold']:.2f}")
        st.metric("Stage 1 ROC AUC", f"{metadata['stage1_roc_auc']:.3f}")
        st.metric("Stage 2 R²", f"{metadata['stage2_r2']:.3f}")

        st.divider()

        st.subheader("🧠 How to Read Results")
        st.markdown("""
        **Stage 1 = Eligibility**
        - Predicts whether the page is likely to enter the visible set

        **Stage 2 = Visibility Strength**
        - Predicts relative attention strength among eligible pages

        **Step 3 change**
        - Relative and rank features are now computed across the analyzed group

        **Important**
        - PAWC is a relative strength score
        - It is **not a percentage**
        """)

        st.divider()

        st.subheader("🎯 Visibility Standards")
        st.info(f"""
        **Core Features Analyzed:** {len(core_features)}

        **Standards Based On:** Top 25% of visible sources
        """)

        with st.expander("View Standards"):
            for feature in core_features:
                if feature in standards:
                    if isinstance(standards[feature], dict):
                        target = standards[feature].get("75th_percentile", 0)
                    else:
                        target = standards[feature]
                    st.write(f"**{feature}:** {target:.3f}")

    st.divider()

    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown("""
    **⚠️ Current App Limitations**
    - This version computes competitive features only across the URLs you provide
    - Competitor discovery is still manual
    - Results depend on the quality of the analyzed competitor set
    - Models are still frozen pretrained assets; no retraining is performed here
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.header("🌐 Enter Query, Your URL, and Competitor URLs")

    query = st.text_input(
        "Search Query",
        placeholder="machine learning portfolio",
        help="The search query you want to analyze"
    )

    user_url = st.text_input(
        "Your Website URL",
        placeholder="https://example.com/your-page",
        help="Enter the main page you want to evaluate"
    )

    competitor_block = st.text_area(
        "Competitor URLs (one per line)",
        placeholder="https://competitor1.com/page\nhttps://competitor2.com/page\nhttps://competitor3.com/page",
        height=180,
        help="Add 2-5 competitor URLs for the same query"
    )

    with st.expander("⚙️ Advanced Options"):
        use_mock = st.checkbox(
            "Use mock competitor data (for testing without internet)",
            value=False,
            help="Generate realistic demo competitors without fetching all pages"
        )

    if st.button("🔍 Analyze Competitive Visibility", type="primary", use_container_width=True):
        if not query or not user_url:
            st.error("Please enter both the query and your URL.")
            return

        user_url = normalize_url(user_url)
        competitor_urls = parse_url_list(competitor_block)
        competitor_urls = [u for u in competitor_urls if u != user_url]

        if not competitor_urls and not use_mock:
            st.error("Please provide at least one competitor URL, or enable mock competitor data.")
            return

        if use_mock and not competitor_urls:
            competitor_urls = [
                "https://example-competitor-1.com",
                "https://example-competitor-2.com",
                "https://example-competitor-3.com",
            ]

        all_urls = [{"url": user_url, "label": "Your Page", "is_user_page": True}]
        for idx, cu in enumerate(competitor_urls, start=1):
            all_urls.append({"url": normalize_url(cu), "label": f"Competitor {idx}", "is_user_page": False})

        progress_bar = st.progress(0)
        status_text = st.empty()

        page_rows = []

        total_steps = len(all_urls) + 3
        current_step = 0

        for item in all_urls:
            current_step += 1
            status_text.text(f"🔍 Extracting features: {item['label']} ({current_step}/{total_steps})")
            progress_bar.progress(current_step / total_steps)

            if use_mock and not item["is_user_page"]:
                domain = short_domain(item["url"])
                base_features = {
                    "Relevance": float(np.random.uniform(0.60, 0.88)),
                    "Influence": float(np.random.uniform(0.58, 0.84)),
                    "Uniqueness": float(np.random.uniform(0.52, 0.79)),
                    "Click_Probability": float(np.random.uniform(0.55, 0.82)),
                    "Diversity": float(np.random.uniform(0.48, 0.78)),
                    "WC": float(np.random.randint(420, 1100)),
                    "query_length": float(len(query)),
                    "query_type_list": 0.0,
                    "query_type_opinion": 0.0,
                    "query_type_other": 1.0,
                    "is_suggested_source": 0.0,
                }
                diagnostics = {
                    "url": item["url"],
                    "domain": domain,
                    "title": item["label"],
                    "meta_description": "",
                    "headings_found": 4,
                    "body_word_count": int(base_features["WC"]),
                    "structure_stats": {
                        "heading_count": 4,
                        "distinct_heading_types": 3,
                        "h1_count": 1,
                        "h2_count": 2,
                        "h3_count": 1,
                        "paragraph_count": 7,
                        "list_count": 2,
                        "image_count": 1,
                        "table_count": 0,
                        "code_block_count": 0,
                        "blockquote_count": 0,
                        "faq_like_count": 1
                    }
                }
                fetch_status = {
                    "content_fetched": False,
                    "fallback_used": True,
                    "domain": domain
                }
                error = None
            else:
                base_features, fetch_status, diagnostics, error = extract_page_features(item["url"], query)

            if error:
                st.error(f"❌ Error extracting features for {item['label']}: {error}")
                return

            page_rows.append({
                "url": item["url"],
                "label": item["label"],
                "is_user_page": item["is_user_page"],
                "domain": fetch_status["domain"],
                "base_features": base_features,
                "fetch_status": fetch_status,
                "diagnostics": diagnostics
            })

        current_step += 1
        status_text.text(f"📊 Computing competitive features ({current_step}/{total_steps})")
        progress_bar.progress(current_step / total_steps)
        time.sleep(0.2)

        page_rows = run_group_predictions(page_rows, clf, reg, features_s1, features_s2)

        current_step += 1
        status_text.text(f"🎯 Comparing your page with standards ({current_step}/{total_steps})")
        progress_bar.progress(current_step / total_steps)
        time.sleep(0.2)

        user_row = [r for r in page_rows if r["is_user_page"]][0]
        comparison, gaps = compare_with_standards(user_row["features"], standards, thresholds, core_features)

        current_step += 1
        status_text.text(f"✅ Analysis complete ({current_step}/{total_steps})")
        progress_bar.progress(current_step / total_steps)
        time.sleep(0.2)

        progress_bar.empty()
        status_text.empty()

        # -------------------------------------------------------------------
        # MAIN RESULTS
        # -------------------------------------------------------------------

        st.divider()
        st.header("📊 Your Page Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if user_row["is_visible"]:
                st.markdown('<div class="success-box"><h3>✅ VISIBLE</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="danger-box"><h3>❌ NOT VISIBLE</h3></div>', unsafe_allow_html=True)

        with col2:
            st.metric("Stage 1 Probability", f"{user_row['visibility_prob']:.1%}")

        with col3:
            st.metric("Stage 2 PAWC Score", f"{user_row['predicted_pawc']:.1f}")

        with col4:
            st.metric("Final Rank", f"{user_row['final_rank']}/{len(page_rows)}")

        tier = get_visibility_tier(
            user_row["visibility_prob"],
            user_row["predicted_pawc"],
            user_row["is_visible"]
        )
        interpretation = build_interpretation_text(
            user_row["is_visible"],
            user_row["visibility_prob"],
            user_row["predicted_pawc"],
            user_row["final_rank"],
            len(page_rows)
        )

        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **Interpretation**
        - **Stage 1:** Probability that the page is considered for visibility
        - **Stage 2:** Relative visibility-strength score among eligible pages
        - **Estimated Visibility Tier:** `{tier}`
        - **Competitive Rank:** `{user_row['final_rank']} / {len(page_rows)}`

        **What this result means:**  
        {interpretation}
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # -------------------------------------------------------------------
        # RANKING TABLE
        # -------------------------------------------------------------------

        st.divider()
        st.header("🏆 Competitive Ranking Table")

        ranking_rows = []
        sorted_rows = sorted(page_rows, key=lambda r: r["final_rank"])

        for row in sorted_rows:
            ranking_rows.append({
                "Rank": row["final_rank"],
                "Page": row["label"],
                "Domain": row["domain"],
                "Visible": "Yes" if row["is_visible"] else "No",
                "Stage1_Prob": round(row["visibility_prob"], 4),
                "PAWC": round(row["predicted_pawc"], 2),
                "Relevance": round(row["features"]["Relevance"], 3),
                "Influence": round(row["features"]["Influence"], 3),
                "Uniqueness": round(row["features"]["Uniqueness"], 3),
                "Diversity": round(row["features"]["Diversity"], 3),
                "WC": round(row["features"]["WC"], 1),
                "WC_rel": round(row["features"]["WC_rel"], 3),
                "Type": "You" if row["is_user_page"] else "Competitor"
            })

        ranking_df = pd.DataFrame(ranking_rows)
        st.dataframe(ranking_df, use_container_width=True, height=320)

        # -------------------------------------------------------------------
        # USER VS COMPETITOR AVERAGE
        # -------------------------------------------------------------------

        st.divider()
        st.header("📈 Your Page vs Competitor Average")

        group_compare = compare_user_vs_group(user_row, page_rows)
        if group_compare:
            compare_rows = []
            for feat, vals in group_compare.items():
                compare_rows.append({
                    "Feature": feat,
                    "Your Value": round(vals["user"], 3),
                    "Competitor Avg": round(vals["competitor_avg"], 3),
                    "Difference": round(vals["diff"], 3),
                    "Direction": "Above Avg" if vals["diff"] > 0 else "Below Avg" if vals["diff"] < 0 else "Equal"
                })
            compare_df = pd.DataFrame(compare_rows)
            st.dataframe(compare_df, use_container_width=True, height=250)

        # -------------------------------------------------------------------
        # FEATURE ANALYSIS
        # -------------------------------------------------------------------

        st.divider()
        st.header("🎯 Feature Analysis")

        total_features = len(comparison)
        features_meeting = sum(1 for c in comparison.values() if c["gap_percentage"] <= 10)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Features Analyzed", total_features)
        with c2:
            st.metric("Meeting Target", f"{features_meeting}/{total_features}")
        with c3:
            pct_meeting = (features_meeting / total_features * 100) if total_features > 0 else 0
            st.metric("Success Rate", f"{pct_meeting:.0f}%")

        comparison_data = []
        for feature, values in comparison.items():
            provenance_info = provenance.get(feature, {"source": "Unknown", "reliability": "Unknown"})
            comparison_data.append({
                "Feature": feature,
                "Your Value": f"{values['actual']:.3f}",
                "Target (75th %ile)": f"{values['target']:.3f}",
                "Gap": f"{values['gap_percentage']:.1f}%",
                "Status": values["status"],
                "Source": provenance_info["source"],
                "Reliability": provenance_info["reliability"]
            })

        comparison_df = pd.DataFrame(comparison_data)
        st.subheader("📋 Detailed Feature Comparison")
        st.dataframe(comparison_df, use_container_width=True, height=340)

        # -------------------------------------------------------------------
        # TOP GAPS
        # -------------------------------------------------------------------

        if gaps:
            st.divider()
            st.subheader("🔴 Features Below Target (Improvement Needed)")

            for i, gap in enumerate(gaps[:5], 1):
                gap_pct = gap["gap"]

                if gap_pct > 30:
                    box_class = "danger-box"
                    icon = "🔴"
                elif gap_pct > 10:
                    box_class = "warning-box"
                    icon = "⚠️"
                else:
                    continue

                provenance_info = provenance.get(gap["feature"], {"source": "Unknown", "reliability": "Unknown"})

                st.markdown(f'<div class="{box_class}">', unsafe_allow_html=True)
                st.markdown(f"""
                **{icon} {i}. {gap['feature']}**
                - Your value: `{gap['actual']:.3f}`
                - Target value: `{gap['target']:.3f}`
                - Gap: `{gap_pct:.1f}%` below target
                - Source: `{provenance_info['source']}`
                - Reliability: `{provenance_info['reliability']}`
                """)
                st.markdown('</div>', unsafe_allow_html=True)

        # -------------------------------------------------------------------
        # RECOMMENDATIONS
        # -------------------------------------------------------------------

        st.divider()
        st.subheader("💡 Recommendations")

        recommendations = generate_recommendations(gaps, user_row["is_visible"], user_row, page_rows)
        for rec in recommendations:
            st.info(rec)

        # -------------------------------------------------------------------
        # PAGE DIAGNOSTICS
        # -------------------------------------------------------------------

        st.divider()
        st.header("🧩 Page Diagnostics")

        diag_tabs = st.tabs([row["label"] for row in sorted_rows])

        for tab, row in zip(diag_tabs, sorted_rows):
            with tab:
                st.write(f"**URL:** {row['url']}")
                st.write(f"**Domain:** {row['domain']}")
                st.write(f"**Fetched successfully:** {row['fetch_status'].get('content_fetched', False)}")
                st.write(f"**Fallback used:** {row['fetch_status'].get('fallback_used', False)}")

                st.write(f"**Title:** {row['diagnostics'].get('title', '') or 'N/A'}")
                st.write(f"**Meta description present:** {bool(row['diagnostics'].get('meta_description', ''))}")
                st.write(f"**Headings found:** {row['diagnostics'].get('headings_found', 0)}")
                st.write(f"**Body word count:** {row['diagnostics'].get('body_word_count', 0)}")

                stats = row["diagnostics"].get("structure_stats", {})
                if stats:
                    diag_df = pd.DataFrame(
                        [{"Metric": k, "Value": v} for k, v in stats.items()]
                    )
                    st.dataframe(diag_df, use_container_width=True, height=260)

        # -------------------------------------------------------------------
        # FEATURE SOURCE SUMMARY FOR USER PAGE
        # -------------------------------------------------------------------

        st.divider()
        st.header("🧾 Your Page Feature Source Summary")

        provenance_rows = []
        for feature in sorted(user_row["features"].keys()):
            info = provenance.get(feature, {"source": "Unknown", "reliability": "Unknown"})
            provenance_rows.append({
                "Feature": feature,
                "Value": f"{user_row['features'][feature]:.3f}" if isinstance(user_row["features"][feature], (float, int)) else str(user_row["features"][feature]),
                "Source": info["source"],
                "Reliability": info["reliability"]
            })

        provenance_df = pd.DataFrame(provenance_rows)
        st.dataframe(provenance_df, use_container_width=True, height=320)

if __name__ == "__main__":
    main()
"""
============================================================================
SEO PREDICTOR - STREAMLIT APP (EMBEDDING + SEO IMPROVED)
Competitive SEO analyzer with:
- manual competitor URLs
- optional auto competitor discovery
- stronger on-page, technical, snippet, and authority scoring
- embedding-based semantic relevance
- comparative ranking and recommendations

Run:
    streamlit run seo_predictor.py
============================================================================
"""

import re
import time
from urllib.parse import urlparse, unquote, quote_plus
from collections import Counter

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="SEO Visibility Predictor",
    page_icon="🌐",
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
.small-note {
    color: #666;
    font-size: 0.9rem;
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
    "under", "up", "down", "out", "more", "most", "very", "also", "using",
    "used", "use", "make", "made", "get", "got", "new", "old"
}

POWER_WORDS = {
    "best", "top", "guide", "tutorial", "complete", "ultimate", "easy", "free",
    "proven", "simple", "fast", "2025", "2026", "project", "portfolio", "examples",
    "checklist", "step", "steps", "case study", "comparison", "vs"
}

TRUST_DOMAINS_HIGH = {
    "wikipedia.org", "nih.gov", "nature.com", "science.org", "github.com", "github.io"
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

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

def tokenize(text):
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def containment_ratio(query_tokens, doc_tokens):
    q = set(query_tokens)
    d = set(doc_tokens)
    if not q:
        return 0.0
    return len(q & d) / len(q)

def jaccard_similarity(tokens_a, tokens_b):
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

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

def count_syllables(word):
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def flesch_reading_ease(text):
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r"[A-Za-z]+", text)
    if not sentences or not words:
        return 50.0
    syllables = sum(count_syllables(w) for w in words)
    wps = len(words) / len(sentences)
    spw = syllables / len(words)
    return 206.835 - 1.015 * wps - 84.6 * spw

def readability_score(text):
    score = flesch_reading_ease(text[:8000])
    if score >= 70:
        return 1.0
    if score >= 55:
        return 0.8
    if score >= 40:
        return 0.6
    if score >= 25:
        return 0.4
    return 0.2

# ============================================================================
# EMBEDDING MODEL
# ============================================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def cosine_sim(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def build_page_semantic_text(title, headings, body_text):
    heading_text = " ".join([h[1] for h in headings[:12]])
    body_excerpt = normalize_whitespace(body_text[:3000])
    return normalize_whitespace(f"{title}. {heading_text}. {body_excerpt}")

def compute_semantic_relevance(query, title, headings, body_text, emb_model):
    query_text = normalize_whitespace(query)
    page_text = build_page_semantic_text(title, headings, body_text)

    if not query_text or not page_text:
        return 0.0

    embeddings = emb_model.encode([query_text, page_text], normalize_embeddings=True)
    score = cosine_sim(embeddings[0], embeddings[1])
    score_01 = (score + 1.0) / 2.0
    return clamp(score_01)

# ============================================================================
# AUTO COMPETITOR DISCOVERY
# ============================================================================

def discover_competitors_duckduckgo(query, max_results=5):
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        resp = requests.get(search_url, timeout=12, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        urls = []
        for a in soup.select("a.result__a"):
            href = a.get("href")
            if href and href.startswith("http"):
                urls.append(normalize_url(href))

        if not urls:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http"):
                    urls.append(normalize_url(href))

        deduped = []
        seen = set()
        for u in urls:
            dom = short_domain(u)
            if dom not in seen:
                deduped.append(u)
                seen.add(dom)
            if len(deduped) >= max_results:
                break

        return deduped
    except Exception:
        return []

# ============================================================================
# HTML EXTRACTION
# ============================================================================

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
                headings.append((tag_name, txt))
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

def extract_canonical(soup):
    tag = soup.find("link", rel=lambda x: x and "canonical" in str(x).lower())
    return tag.get("href", "").strip() if tag else ""

def extract_robots_meta(soup):
    tag = soup.find("meta", attrs={"name": lambda x: x and x.lower() == "robots"})
    return tag.get("content", "").strip().lower() if tag else ""

def has_schema_markup(soup):
    scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
    return len(scripts) > 0

def has_open_graph(soup):
    return soup.find("meta", attrs={"property": "og:title"}) is not None

# ============================================================================
# SEO FEATURE FUNCTIONS
# ============================================================================

def build_structure_stats(soup, body_text, domain):
    headings = extract_headings(soup)
    h1_count = len(soup.find_all("h1"))
    h2_count = len(soup.find_all("h2"))
    h3_count = len(soup.find_all("h3"))
    paragraph_count = len([p for p in soup.find_all("p") if normalize_whitespace(p.get_text(" ", strip=True))])
    list_count = len(soup.find_all(["ul", "ol"]))
    image_count = len([img for img in soup.find_all("img") if img.get("src")])
    image_alt_count = len([img for img in soup.find_all("img") if img.get("alt") and normalize_whitespace(img.get("alt"))])
    table_count = len(soup.find_all("table"))
    code_block_count = len(soup.find_all(["pre", "code"]))
    faq_like_count = len(re.findall(r"\?", body_text[:8000]))

    internal_links = 0
    external_links = 0
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if href.startswith("#") or href.startswith("/"):
            internal_links += 1
        elif href.startswith("http"):
            link_domain = short_domain(href)
            if domain and domain in link_domain:
                internal_links += 1
            else:
                external_links += 1

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
        "image_alt_count": image_alt_count,
        "table_count": table_count,
        "code_block_count": code_block_count,
        "internal_links": internal_links,
        "external_links": external_links,
        "faq_like_count": faq_like_count
    }

def compute_lexical_relevance(query, title, headings, body_text):
    q_tokens = tokenize(query)
    title_tokens = tokenize(title)
    heading_tokens = tokenize(" ".join([h[1] for h in headings]))
    body_tokens = tokenize(body_text[:12000])

    score = (
        0.30 * containment_ratio(q_tokens, title_tokens) +
        0.18 * containment_ratio(q_tokens, heading_tokens) +
        0.16 * containment_ratio(q_tokens, body_tokens) +
        0.10 * jaccard_similarity(q_tokens, title_tokens) +
        0.08 * jaccard_similarity(q_tokens, heading_tokens) +
        0.08 * jaccard_similarity(q_tokens, body_tokens) +
        0.06 * phrase_match_score(query, title) +
        0.02 * phrase_match_score(query, " ".join([h[1] for h in headings[:10]])) +
        0.02 * phrase_match_score(query, body_text[:3000])
    )
    return clamp(score)

def compute_hybrid_relevance(query, title, headings, body_text, emb_model):
    lexical = compute_lexical_relevance(query, title, headings, body_text)
    semantic = compute_semantic_relevance(query, title, headings, body_text, emb_model)
    return clamp(0.65 * semantic + 0.35 * lexical), semantic, lexical

def compute_title_score(query, title):
    title = normalize_whitespace(title)
    q_tokens = tokenize(query)
    t_tokens = tokenize(title)

    contains_keywords = containment_ratio(q_tokens, t_tokens)
    exact_phrase = phrase_match_score(query, title)
    title_len = len(title)
    title_len_score = 1.0 if 45 <= title_len <= 65 else 0.75 if 35 <= title_len <= 75 else 0.45
    power_word_score = clamp(sum(1 for w in POWER_WORDS if w in title.lower()) / 3)
    readability = 1.0 if 6 <= len(title.split()) <= 12 else 0.7

    return clamp(
        0.35 * contains_keywords +
        0.20 * exact_phrase +
        0.20 * title_len_score +
        0.15 * power_word_score +
        0.10 * readability
    )

def compute_meta_score(query, meta_desc):
    meta_desc = normalize_whitespace(meta_desc)
    if not meta_desc:
        return 0.20

    q_tokens = tokenize(query)
    m_tokens = tokenize(meta_desc)
    overlap = containment_ratio(q_tokens, m_tokens)
    exact_phrase = phrase_match_score(query, meta_desc)
    meta_len = len(meta_desc)
    meta_len_score = 1.0 if 120 <= meta_len <= 160 else 0.75 if 90 <= meta_len <= 180 else 0.45

    return clamp(0.45 * overlap + 0.20 * exact_phrase + 0.35 * meta_len_score)

def compute_h1_score(query, headings):
    h1s = [txt for tag, txt in headings if tag == "h1"]
    if not h1s:
        return 0.15
    h1 = h1s[0]
    return clamp(
        0.55 * containment_ratio(tokenize(query), tokenize(h1)) +
        0.45 * phrase_match_score(query, h1)
    )

def compute_url_score(query, url):
    parsed = urlparse(url)
    path = unquote(parsed.path.lower().replace("-", " ").replace("_", " "))
    slug_text = f"{parsed.netloc.lower()} {path}"
    q_tokens = tokenize(query)
    slug_tokens = tokenize(slug_text)

    path_depth = len([p for p in parsed.path.split("/") if p])
    path_depth_score = 1.0 if 1 <= path_depth <= 3 else 0.75 if path_depth <= 5 else 0.45

    return clamp(
        0.55 * containment_ratio(q_tokens, slug_tokens) +
        0.20 * phrase_match_score(query, slug_text) +
        0.25 * path_depth_score
    )

def compute_keyword_density_score(query, body_text):
    q_tokens = tokenize(query)
    body_tokens = tokenize(body_text[:12000])
    if not q_tokens or not body_tokens:
        return 0.0

    counts = Counter(body_tokens)
    total_occ = sum(counts[t] for t in q_tokens)
    density = total_occ / max(len(body_tokens), 1)

    if 0.005 <= density <= 0.025:
        density_score = 1.0
    elif 0.002 <= density < 0.005 or 0.025 < density <= 0.04:
        density_score = 0.7
    elif 0.001 <= density < 0.002:
        density_score = 0.45
    else:
        density_score = 0.25

    coverage_score = containment_ratio(q_tokens, body_tokens)
    return clamp(0.55 * density_score + 0.45 * coverage_score)

def compute_content_quality_score(structure_stats, word_count, body_text):
    depth_score = 1.0 if 900 <= word_count <= 2200 else 0.8 if 600 <= word_count <= 3000 else 0.55
    heading_score = clamp(structure_stats["heading_count"] / 10)
    paragraph_score = clamp(structure_stats["paragraph_count"] / 14)
    list_score = clamp(structure_stats["list_count"] / 5)
    media_score = clamp((structure_stats["image_count"] + structure_stats["table_count"] + structure_stats["code_block_count"]) / 5)
    read_score = readability_score(body_text)

    return clamp(
        0.28 * depth_score +
        0.18 * heading_score +
        0.16 * paragraph_score +
        0.12 * list_score +
        0.10 * media_score +
        0.16 * read_score
    )

def compute_technical_seo_score(url, response_time, structure_stats, canonical_url, robots_meta, schema_present, og_present):
    parsed = urlparse(url)
    https_score = 1.0 if parsed.scheme.lower() == "https" else 0.3
    response_score = 1.0 if response_time <= 1.5 else 0.8 if response_time <= 3.0 else 0.5 if response_time <= 5.0 else 0.25
    heading_hierarchy_score = 1.0 if structure_stats["h1_count"] == 1 and structure_stats["h2_count"] >= 1 else 0.6 if structure_stats["h1_count"] >= 1 else 0.25
    image_alt_score = safe_div(structure_stats["image_alt_count"], max(structure_stats["image_count"], 1), 1.0) if structure_stats["image_count"] else 1.0
    link_score = 1.0 if structure_stats["internal_links"] >= 2 else 0.6 if structure_stats["internal_links"] >= 1 else 0.3
    canonical_score = 1.0 if canonical_url else 0.55
    robots_score = 0.0 if "noindex" in robots_meta else 1.0
    schema_score = 1.0 if schema_present else 0.55
    og_score = 1.0 if og_present else 0.6

    return clamp(
        0.18 * https_score +
        0.18 * response_score +
        0.12 * heading_hierarchy_score +
        0.10 * image_alt_score +
        0.10 * link_score +
        0.10 * canonical_score +
        0.10 * robots_score +
        0.07 * schema_score +
        0.05 * og_score
    )

def compute_authority_score(domain, has_https, structure_stats, title, meta_desc):
    domain = domain.lower()
    base = 0.50

    if domain in TRUST_DOMAINS_HIGH or ".gov" in domain or ".edu" in domain:
        base = 0.88
    elif any(x in domain for x in ["github.io", "github.com", ".org", "medium.com", "substack.com"]):
        base = 0.68

    https_bonus = 0.05 if has_https else 0.0
    metadata_bonus = 0.04 if title else 0.0
    metadata_bonus += 0.03 if meta_desc else 0.0
    structure_bonus = 0.05 if structure_stats["heading_count"] >= 4 else 0.0
    structure_bonus += 0.03 if structure_stats["paragraph_count"] >= 6 else 0.0

    return clamp(base + https_bonus + metadata_bonus + structure_bonus)

def compute_ctr_score(query, title, meta_desc, title_score, meta_score, relevance):
    q_tokens = tokenize(query)
    title_overlap = containment_ratio(q_tokens, tokenize(title))
    meta_overlap = containment_ratio(q_tokens, tokenize(meta_desc))
    power_bonus = clamp(sum(1 for w in POWER_WORDS if w in title.lower()) / 3)

    return clamp(
        0.28 * title_score +
        0.22 * meta_score +
        0.18 * relevance +
        0.16 * title_overlap +
        0.08 * meta_overlap +
        0.08 * power_bonus
    )

def repetition_mass(tokens):
    counts = Counter(tokens)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return clamp(repeated / max(len(tokens), 1))

def originality_hits_score(text):
    text = (text or "").lower()
    markers = [
        "case study", "results", "benchmark", "implementation", "architecture",
        "challenge", "solution", "deployment", "accuracy", "precision", "recall",
        "latency", "project", "portfolio", "lessons learned"
    ]
    hits = sum(1 for marker in markers if marker in text)
    return clamp(hits / 6)

def compute_uniqueness_proxy(title, headings, body_text, structure_stats):
    body_tokens = tokenize(body_text[:10000])
    title_tokens = tokenize(title)
    heading_tokens = tokenize(" ".join([h[1] for h in headings[:20]]))

    if not body_tokens:
        return 0.0

    unique_ratio = len(set(body_tokens)) / len(body_tokens)
    repeated_mass = repetition_mass(body_tokens)
    originality_hits = originality_hits_score(f"{title} {' '.join([h[1] for h in headings])} {body_text[:4000]}")
    title_specificity = clamp(len(set(title_tokens)) / 8) if title_tokens else 0.2
    heading_specificity = clamp(len(set(heading_tokens)) / 18) if heading_tokens else 0.2
    structure_bonus = 0.12 if structure_stats["table_count"] > 0 or structure_stats["code_block_count"] > 0 else 0.0

    return clamp(
        0.30 * clamp((unique_ratio - 0.15) / 0.55) +
        0.20 * originality_hits +
        0.16 * title_specificity +
        0.12 * heading_specificity +
        0.10 * structure_bonus +
        0.12 * (1 - repeated_mass)
    )

def compute_snippet_readiness(query, title, meta_desc, headings, structure_stats):
    q_tokens = tokenize(query)
    first_h2 = ""
    for tag, txt in headings:
        if tag == "h2":
            first_h2 = txt
            break

    faq_bonus = clamp(structure_stats["faq_like_count"] / 3)
    heading_match = containment_ratio(q_tokens, tokenize(first_h2))
    meta_match = containment_ratio(q_tokens, tokenize(meta_desc))
    title_match = containment_ratio(q_tokens, tokenize(title))

    return clamp(
        0.30 * title_match +
        0.25 * meta_match +
        0.20 * heading_match +
        0.15 * faq_bonus +
        0.10 * (1.0 if meta_desc else 0.4)
    )

def compute_freshness_score(title, body_text):
    years = re.findall(r"\b(202[4-9]|2030)\b", f"{title} {body_text[:3000]}")
    if years:
        return 1.0
    if re.search(r"\b(updated|latest|new|recent)\b", f"{title} {body_text[:2000]}", re.I):
        return 0.75
    return 0.45

def weighted_seo_score(feats):
    return clamp(
        0.17 * feats["Relevance"] +
        0.12 * feats["Title_Score"] +
        0.07 * feats["Meta_Score"] +
        0.05 * feats["H1_Score"] +
        0.06 * feats["URL_Score"] +
        0.07 * feats["Keyword_Density_Score"] +
        0.14 * feats["Content_Quality_Score"] +
        0.12 * feats["Technical_SEO_Score"] +
        0.08 * feats["Authority_Score"] +
        0.06 * feats["CTR_Score"] +
        0.04 * feats["Uniqueness_Proxy"] +
        0.01 * feats["Freshness_Score"] +
        0.01 * feats["Snippet_Readiness"]
    )

# ============================================================================
# FEATURE PROVENANCE
# ============================================================================

def get_feature_provenance():
    return {
        "Relevance": "Hybrid semantic + lexical relevance using SentenceTransformer embeddings and title/headings/body overlap",
        "Semantic_Relevance": "Cosine similarity between query embedding and page embedding",
        "Lexical_Relevance": "Keyword/phrase overlap with title, headings, and body",
        "Title_Score": "Keyword presence, phrase match, title length, power words",
        "Meta_Score": "Meta description presence, keyword match, length",
        "H1_Score": "Query match with H1",
        "URL_Score": "Keyword presence in URL/domain/path and path depth",
        "Keyword_Density_Score": "Query term frequency and coverage in body text",
        "Content_Quality_Score": "Depth, headings, paragraphs, lists, media, readability",
        "Technical_SEO_Score": "HTTPS, response time, heading structure, alt text, links, canonical, robots, schema, OG",
        "Authority_Score": "Domain heuristic, HTTPS, metadata, structure",
        "CTR_Score": "Snippet attractiveness and query alignment",
        "Uniqueness_Proxy": "Lexical diversity, originality markers, repetition penalty",
        "Freshness_Score": "Recency words / year mentions",
        "Snippet_Readiness": "Title/meta/heading/FAQ readiness for snippet-style display",
        "WC": "Extracted word count from visible text",
        "WC_rel": "Relative word count vs analyzed group average",
        "SEO_Score": "Weighted aggregate SEO score",
    }

# ============================================================================
# PAGE EXTRACTION
# ============================================================================

def extract_page_seo_features(url, query, emb_model):
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
            start = time.perf_counter()
            response = requests.get(
                url,
                timeout=12,
                headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
            response_time = time.perf_counter() - start

            soup = BeautifulSoup(response.content, "html.parser")

            title = get_title_text(soup)
            meta_desc = get_meta_description(soup)
            headings = extract_headings(soup)
            body_text = get_visible_text(soup)
            canonical_url = extract_canonical(soup)
            robots_meta = extract_robots_meta(soup)
            schema_present = has_schema_markup(soup)
            og_present = has_open_graph(soup)

            structure_stats = build_structure_stats(soup, body_text, domain)
            word_count = len(tokenize(body_text))

            if word_count < 30:
                raise ValueError("Very little extractable content found on page")

            hybrid_rel, semantic_rel, lexical_rel = compute_hybrid_relevance(
                query, title, headings, body_text, emb_model
            )

            features = {
                "Relevance": hybrid_rel,
                "Semantic_Relevance": semantic_rel,
                "Lexical_Relevance": lexical_rel,
                "Title_Score": compute_title_score(query, title),
                "Meta_Score": compute_meta_score(query, meta_desc),
                "H1_Score": compute_h1_score(query, headings),
                "URL_Score": compute_url_score(query, url),
                "Keyword_Density_Score": compute_keyword_density_score(query, body_text),
                "Content_Quality_Score": compute_content_quality_score(structure_stats, word_count, body_text),
                "Technical_SEO_Score": compute_technical_seo_score(
                    url, response_time, structure_stats, canonical_url, robots_meta, schema_present, og_present
                ),
                "Authority_Score": compute_authority_score(domain, has_https, structure_stats, title, meta_desc),
                "WC": float(word_count),
                "Uniqueness_Proxy": compute_uniqueness_proxy(title, headings, body_text, structure_stats),
                "Freshness_Score": compute_freshness_score(title, body_text),
                "Snippet_Readiness": compute_snippet_readiness(query, title, meta_desc, headings, structure_stats),
            }
            features["CTR_Score"] = compute_ctr_score(
                query, title, meta_desc,
                features["Title_Score"],
                features["Meta_Score"],
                features["Relevance"]
            )

            diagnostics = {
                "url": url,
                "domain": domain,
                "title": title,
                "meta_description": meta_desc,
                "canonical_url": canonical_url,
                "robots_meta": robots_meta,
                "schema_present": schema_present,
                "og_present": og_present,
                "headings_found": len(headings),
                "body_word_count": word_count,
                "response_time_sec": round(response_time, 3),
                "structure_stats": structure_stats
            }

            fetch_status["content_fetched"] = True
            return features, fetch_status, diagnostics, None

        except Exception:
            fetch_status["fallback_used"] = True
            features = {
                "Semantic_Relevance": 0.50,
                "Lexical_Relevance": 0.55,
                "Relevance": 0.53,
                "Title_Score": 0.50,
                "Meta_Score": 0.40,
                "H1_Score": 0.45,
                "URL_Score": 0.50,
                "Keyword_Density_Score": 0.45,
                "Content_Quality_Score": 0.50,
                "Technical_SEO_Score": 0.55,
                "Authority_Score": compute_authority_score(domain, has_https, {"heading_count": 0, "paragraph_count": 0}, "", ""),
                "CTR_Score": 0.48,
                "WC": 400.0,
                "Uniqueness_Proxy": 0.48,
                "Freshness_Score": 0.45,
                "Snippet_Readiness": 0.45,
            }
            diagnostics = {
                "url": url,
                "domain": domain,
                "title": "",
                "meta_description": "",
                "canonical_url": "",
                "robots_meta": "",
                "schema_present": False,
                "og_present": False,
                "headings_found": 0,
                "body_word_count": 400,
                "response_time_sec": None,
                "structure_stats": {}
            }
            return features, fetch_status, diagnostics, None

    except Exception as e:
        return None, None, None, str(e)

# ============================================================================
# GROUP COMPUTATION
# ============================================================================

def compute_group_features(rows):
    if not rows:
        return rows

    avg_wc = float(np.mean([r["features"]["WC"] for r in rows]))
    avg_ctr = float(np.mean([r["features"]["CTR_Score"] for r in rows]))
    domain_counts = Counter([r["domain"] for r in rows])

    def compute_rank_map(metric_name, descending=True):
        values = {i: r["features"][metric_name] for i, r in enumerate(rows)}
        ordered = sorted(values.items(), key=lambda x: x[1], reverse=descending)
        return {idx: rank + 1 for rank, (idx, _) in enumerate(ordered)}

    rank_metrics = [
        "Relevance", "Semantic_Relevance", "Lexical_Relevance", "Title_Score", "Meta_Score",
        "H1_Score", "URL_Score", "Keyword_Density_Score", "Content_Quality_Score",
        "Technical_SEO_Score", "Authority_Score", "CTR_Score", "Uniqueness_Proxy",
        "Freshness_Score", "Snippet_Readiness", "WC"
    ]

    for feat in rank_metrics:
        rank_map = compute_rank_map(feat, True)
        rank_field = f"{feat}_Rank"
        for i, row in enumerate(rows):
            row["features"][rank_field] = float(rank_map[i])

    num_sources = len(rows)

    for row in rows:
        row["features"]["WC_rel"] = float(safe_div(row["features"]["WC"], avg_wc, 1.0))
        row["features"]["CTR_rel"] = float(safe_div(row["features"]["CTR_Score"], avg_ctr, 1.0))
        row["features"]["Domain_Freq"] = float(domain_counts[row["domain"]])
        row["features"]["Num_Sources"] = float(num_sources)
        row["features"]["SEO_Score"] = weighted_seo_score(row["features"])

    seo_rank_map = compute_rank_map("SEO_Score", True)
    total = len(rows)
    for i, row in enumerate(rows):
        row["features"]["SEO_Rank"] = float(seo_rank_map[i])
        row["features"]["SEO_Percentile"] = float((total - seo_rank_map[i] + 1) / max(total, 1))
        row["rank"] = seo_rank_map[i]

    return rows

def compare_user_vs_group(user_row, rows):
    competitors = [r for r in rows if not r["is_user_page"]]
    if not competitors:
        return {}

    compare_features = [
        "SEO_Score", "Relevance", "Semantic_Relevance", "Lexical_Relevance",
        "Title_Score", "Meta_Score", "H1_Score", "URL_Score",
        "Keyword_Density_Score", "Content_Quality_Score", "Technical_SEO_Score",
        "Authority_Score", "CTR_Score", "Uniqueness_Proxy", "Freshness_Score",
        "Snippet_Readiness", "WC", "WC_rel"
    ]
    result = {}
    for feat in compare_features:
        comp_avg = float(np.mean([r["features"].get(feat, 0.0) for r in competitors]))
        user_val = float(user_row["features"].get(feat, 0.0))
        result[feat] = {
            "user": user_val,
            "competitor_avg": comp_avg,
            "diff": user_val - comp_avg
        }
    return result

# ============================================================================
# GAPS + RECOMMENDATIONS
# ============================================================================

def feature_target(value_name):
    targets = {
        "SEO_Score": 0.78,
        "Relevance": 0.75,
        "Semantic_Relevance": 0.78,
        "Lexical_Relevance": 0.70,
        "Title_Score": 0.80,
        "Meta_Score": 0.75,
        "H1_Score": 0.75,
        "URL_Score": 0.70,
        "Keyword_Density_Score": 0.72,
        "Content_Quality_Score": 0.78,
        "Technical_SEO_Score": 0.80,
        "Authority_Score": 0.72,
        "CTR_Score": 0.76,
        "Uniqueness_Proxy": 0.68,
        "Freshness_Score": 0.70,
        "Snippet_Readiness": 0.72,
        "WC_rel": 1.00,
    }
    return targets.get(value_name, 0.75)

def build_gap_table(user_row):
    core = [
        "SEO_Score", "Relevance", "Semantic_Relevance", "Lexical_Relevance",
        "Title_Score", "Meta_Score", "H1_Score", "URL_Score",
        "Keyword_Density_Score", "Content_Quality_Score", "Technical_SEO_Score",
        "Authority_Score", "CTR_Score", "Uniqueness_Proxy", "Freshness_Score",
        "Snippet_Readiness", "WC_rel"
    ]
    rows = []
    for feat in core:
        actual = float(user_row["features"].get(feat, 0.0))
        target = feature_target(feat)
        gap_pct = ((target - actual) / target * 100) if target > 0 else 0.0
        rows.append({
            "Feature": feat,
            "Your Value": actual,
            "Target": target,
            "Gap %": gap_pct,
            "Status": "✅" if gap_pct <= 10 else "⚠️" if gap_pct <= 30 else "❌"
        })
    rows.sort(key=lambda x: x["Gap %"], reverse=True)
    return rows

def generate_seo_recommendations(user_row, rows):
    gap_rows = build_gap_table(user_row)
    recommendations = []

    top_gaps = [g for g in gap_rows if g["Gap %"] > 10][:6]
    if not top_gaps:
        recommendations.append("✅ Your page is already close to the benchmark targets across the major SEO signals.")
        return recommendations

    for gap in top_gaps:
        feat = gap["Feature"]
        gap_pct = gap["Gap %"]

        if feat == "Title_Score":
            recommendations.append(f"📝 **Improve Title ({gap_pct:.0f}% gap):** Put the keyword earlier, keep the title near 50–60 characters, and make it more compelling.")
        elif feat == "Meta_Score":
            recommendations.append(f"📄 **Improve Meta Description ({gap_pct:.0f}% gap):** Add a clearer meta description with keyword alignment and stronger click intent.")
        elif feat == "H1_Score":
            recommendations.append(f"🔖 **Improve H1 ({gap_pct:.0f}% gap):** Use one strong H1 closely aligned with the query.")
        elif feat == "URL_Score":
            recommendations.append(f"🔗 **Improve URL SEO ({gap_pct:.0f}% gap):** Use a clean, short URL that contains the main keyword naturally.")
        elif feat == "Keyword_Density_Score":
            recommendations.append(f"🎯 **Improve Keyword Coverage ({gap_pct:.0f}% gap):** Add the query naturally into headings, intro, and body without stuffing.")
        elif feat == "Content_Quality_Score":
            recommendations.append(f"📚 **Improve Content Quality ({gap_pct:.0f}% gap):** Add depth, examples, better structure, lists, images, and clearer section hierarchy.")
        elif feat == "Technical_SEO_Score":
            recommendations.append(f"⚙️ **Improve Technical SEO ({gap_pct:.0f}% gap):** Add canonical tag, improve internal links, alt text, schema, and ensure no indexing issues.")
        elif feat == "Authority_Score":
            recommendations.append(f"🏆 **Improve Authority Signals ({gap_pct:.0f}% gap):** Add stronger proof of expertise, references, GitHub links, and better trust signals.")
        elif feat == "CTR_Score":
            recommendations.append(f"👆 **Improve CTR Signals ({gap_pct:.0f}% gap):** Make title and meta description more attractive and intent-matched.")
        elif feat == "Relevance":
            recommendations.append(f"🔍 **Improve Overall Relevance ({gap_pct:.0f}% gap):** Align title, headings, intro, and page purpose more directly with the query intent.")
        elif feat == "Semantic_Relevance":
            recommendations.append(f"🧠 **Improve Semantic Relevance ({gap_pct:.0f}% gap):** Add conceptually related terms, synonyms, and direct explanations so the page meaning better matches the query.")
        elif feat == "Lexical_Relevance":
            recommendations.append(f"🔤 **Improve Lexical Relevance ({gap_pct:.0f}% gap):** Include the exact query wording more naturally in the title, H1, and early body text.")
        elif feat == "Uniqueness_Proxy":
            recommendations.append(f"✨ **Improve Uniqueness ({gap_pct:.0f}% gap):** Add original insights, screenshots, metrics, code, and project-specific explanations.")
        elif feat == "Freshness_Score":
            recommendations.append(f"🗓️ **Improve Freshness ({gap_pct:.0f}% gap):** Update the page with recent year/context signals and clearly mark updates.")
        elif feat == "Snippet_Readiness":
            recommendations.append(f"📌 **Improve Snippet Readiness ({gap_pct:.0f}% gap):** Add concise answer-style headings, FAQ sections, and clearer summary text.")
        elif feat == "WC_rel":
            recommendations.append(f"📏 **Improve Relative Depth ({gap_pct:.0f}% gap):** Your page is thinner than the group average. Add more useful content sections.")
        elif feat == "SEO_Score":
            recommendations.append(f"🚀 **Improve Overall SEO Strength ({gap_pct:.0f}% gap):** Focus first on title, content quality, technical SEO, and CTR signals.")

    competitors = [r for r in rows if not r["is_user_page"]]
    if competitors:
        best = sorted(competitors, key=lambda x: x["rank"])[0]
        recommendations.append(
            f"📊 **Benchmark:** The strongest competitor in this analysis is `{best['label']}` with SEO score `{best['features']['SEO_Score']:.3f}`."
        )

    return recommendations

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown('<p class="main-header">🌐 SEO Visibility Predictor</p>', unsafe_allow_html=True)
    st.markdown("### Competitive SEO analysis for traditional search engines")

    provenance = get_feature_provenance()

    with st.spinner("Loading semantic relevance model..."):
        emb_model = load_embedding_model()

    with st.sidebar:
        st.header("📘 How this works")
        st.markdown("""
        This app uses:
        - on-page SEO signals
        - technical SEO signals
        - authority heuristics
        - embedding-based semantic relevance
        - competitive comparison across URLs

        It does **not** use a trained ranking model.
        It uses a rule-based weighted scoring system.
        """)

        st.divider()
        st.subheader("🎯 Core SEO Signals")
        st.markdown("""
        - Hybrid relevance
        - Title / Meta / H1 / URL
        - Keyword coverage
        - Content quality
        - Technical SEO
        - Authority
        - CTR potential
        - Freshness
        - Snippet readiness
        """)

    st.divider()

    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown("""
    **⚠️ Current App Limitations**
    - Authority is heuristic, not based on real backlink APIs
    - Auto competitor discovery is best-effort only
    - This is a ranking simulator, not a live Google ranking feed
    - Results depend on the quality of competitor URLs you provide or discover
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.header("🌐 Enter Query, Your URL, and Competitor URLs")

    query = st.text_input(
        "Search Query",
        placeholder="machine learning portfolio",
        help="The SEO query you want to analyze"
    )

    user_url = st.text_input(
        "Your Website URL",
        placeholder="https://example.com/your-page",
        help="Enter your page URL"
    )

    competitor_block = st.text_area(
        "Competitor URLs (one per line)",
        placeholder="https://competitor1.com/page\nhttps://competitor2.com/page\nhttps://competitor3.com/page",
        height=150,
        help="Optional. Add competitor URLs manually."
    )

    with st.expander("⚙️ Advanced Options"):
        auto_discover = st.checkbox("Auto-discover competitors from query", value=True)
        auto_count = st.slider("Auto competitor count", 2, 6, 4)
        use_mock = st.checkbox("Use mock competitor data (for testing)", value=False)

    if st.button("🔍 Analyze SEO Competitiveness", type="primary", use_container_width=True):
        if not query or not user_url:
            st.error("Please enter both the query and your URL.")
            return

        user_url = normalize_url(user_url)
        competitor_urls = parse_url_list(competitor_block)
        competitor_urls = [u for u in competitor_urls if u != user_url]

        if auto_discover and not use_mock:
            discovered = discover_competitors_duckduckgo(query, max_results=auto_count + 2)
            discovered = [u for u in discovered if short_domain(u) != short_domain(user_url)]
            for d in discovered:
                if d not in competitor_urls:
                    competitor_urls.append(d)
            competitor_urls = competitor_urls[:max(auto_count, len(competitor_urls))]

        if use_mock and not competitor_urls:
            competitor_urls = [
                "https://example-competitor-1.com",
                "https://example-competitor-2.com",
                "https://example-competitor-3.com",
            ]

        if not competitor_urls:
            st.error("Please provide competitor URLs or enable auto-discovery/mock mode.")
            return

        all_urls = [{"url": user_url, "label": "Your Page", "is_user_page": True}]
        for idx, cu in enumerate(competitor_urls[:6], start=1):
            all_urls.append({"url": normalize_url(cu), "label": f"Competitor {idx}", "is_user_page": False})

        progress_bar = st.progress(0)
        status_text = st.empty()

        rows = []
        total_steps = len(all_urls) + 2
        current_step = 0

        for item in all_urls:
            current_step += 1
            status_text.text(f"🔍 Extracting SEO features: {item['label']} ({current_step}/{total_steps})")
            progress_bar.progress(current_step / total_steps)

            if use_mock and not item["is_user_page"]:
                domain = short_domain(item["url"])
                features = {
                    "Relevance": float(np.random.uniform(0.60, 0.88)),
                    "Semantic_Relevance": float(np.random.uniform(0.62, 0.90)),
                    "Lexical_Relevance": float(np.random.uniform(0.50, 0.82)),
                    "Title_Score": float(np.random.uniform(0.58, 0.85)),
                    "Meta_Score": float(np.random.uniform(0.50, 0.82)),
                    "H1_Score": float(np.random.uniform(0.55, 0.84)),
                    "URL_Score": float(np.random.uniform(0.52, 0.80)),
                    "Keyword_Density_Score": float(np.random.uniform(0.48, 0.78)),
                    "Content_Quality_Score": float(np.random.uniform(0.56, 0.86)),
                    "Technical_SEO_Score": float(np.random.uniform(0.58, 0.88)),
                    "Authority_Score": float(np.random.uniform(0.52, 0.84)),
                    "CTR_Score": float(np.random.uniform(0.55, 0.83)),
                    "WC": float(np.random.randint(450, 1500)),
                    "Uniqueness_Proxy": float(np.random.uniform(0.48, 0.77)),
                    "Freshness_Score": float(np.random.uniform(0.45, 0.95)),
                    "Snippet_Readiness": float(np.random.uniform(0.45, 0.88)),
                }
                diagnostics = {
                    "url": item["url"],
                    "domain": domain,
                    "title": item["label"],
                    "meta_description": "Sample competitor description",
                    "canonical_url": item["url"],
                    "robots_meta": "",
                    "schema_present": True,
                    "og_present": True,
                    "headings_found": 5,
                    "body_word_count": int(features["WC"]),
                    "response_time_sec": None,
                    "structure_stats": {
                        "heading_count": 5,
                        "distinct_heading_types": 3,
                        "h1_count": 1,
                        "h2_count": 3,
                        "h3_count": 1,
                        "paragraph_count": 8,
                        "list_count": 2,
                        "image_count": 2,
                        "image_alt_count": 2,
                        "table_count": 0,
                        "code_block_count": 0,
                        "internal_links": 3,
                        "external_links": 2,
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
                features, fetch_status, diagnostics, error = extract_page_seo_features(
                    item["url"], query, emb_model
                )

            if error:
                st.error(f"❌ Error extracting features for {item['label']}: {error}")
                return

            rows.append({
                "url": item["url"],
                "label": item["label"],
                "is_user_page": item["is_user_page"],
                "domain": fetch_status["domain"],
                "features": features,
                "fetch_status": fetch_status,
                "diagnostics": diagnostics
            })

        current_step += 1
        status_text.text(f"📊 Computing comparative SEO features ({current_step}/{total_steps})")
        progress_bar.progress(current_step / total_steps)
        time.sleep(0.2)
        rows = compute_group_features(rows)

        current_step += 1
        status_text.text(f"✅ Analysis complete ({current_step}/{total_steps})")
        progress_bar.progress(current_step / total_steps)
        time.sleep(0.2)

        progress_bar.empty()
        status_text.empty()

        user_row = [r for r in rows if r["is_user_page"]][0]

        st.divider()
        st.header("📊 Your SEO Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            seo_score = user_row["features"]["SEO_Score"]
            if seo_score >= 0.78:
                st.markdown('<div class="success-box"><h3>✅ STRONG SEO</h3></div>', unsafe_allow_html=True)
            elif seo_score >= 0.60:
                st.markdown('<div class="warning-box"><h3>⚠️ MODERATE SEO</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="danger-box"><h3>❌ WEAK SEO</h3></div>', unsafe_allow_html=True)

        with col2:
            st.metric("SEO Score", f"{seo_score:.3f}")

        with col3:
            st.metric("Rank", f"{user_row['rank']}/{len(rows)}")

        with col4:
            st.metric("Relative WC", f"{user_row['features']['WC_rel']:.2f}")

        tier = "High" if seo_score >= 0.78 else "Medium" if seo_score >= 0.60 else "Low"
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **Interpretation**
        - **SEO Score:** Weighted overall search-optimization strength
        - **Competitive Rank:** `{user_row['rank']} / {len(rows)}`
        - **Estimated SEO Tier:** `{tier}`

        **What this means:**  
        Your page is being compared against the provided/discovered competitor set using on-page SEO, technical SEO, content quality, authority heuristics, CTR-oriented signals, freshness, snippet readiness, and hybrid semantic relevance.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        st.header("🎯 SEO Sub-Scores")

        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        with sc1:
            st.metric("Relevance", f"{user_row['features']['Relevance']:.3f}")
        with sc2:
            st.metric("Content", f"{user_row['features']['Content_Quality_Score']:.3f}")
        with sc3:
            st.metric("Technical", f"{user_row['features']['Technical_SEO_Score']:.3f}")
        with sc4:
            st.metric("Authority", f"{user_row['features']['Authority_Score']:.3f}")
        with sc5:
            st.metric("CTR", f"{user_row['features']['CTR_Score']:.3f}")

        st.divider()
        st.header("🏆 SEO Competitive Ranking")

        ranking_rows = []
        sorted_rows = sorted(rows, key=lambda r: r["rank"])
        for row in sorted_rows:
            ranking_rows.append({
                "Rank": row["rank"],
                "Page": row["label"],
                "Domain": row["domain"],
                "SEO Score": round(row["features"]["SEO_Score"], 3),
                "Relevance": round(row["features"]["Relevance"], 3),
                "Semantic Rel": round(row["features"]["Semantic_Relevance"], 3),
                "Lexical Rel": round(row["features"]["Lexical_Relevance"], 3),
                "Title": round(row["features"]["Title_Score"], 3),
                "Technical": round(row["features"]["Technical_SEO_Score"], 3),
                "Authority": round(row["features"]["Authority_Score"], 3),
                "CTR": round(row["features"]["CTR_Score"], 3),
                "Snippet": round(row["features"]["Snippet_Readiness"], 3),
                "WC": round(row["features"]["WC"], 1),
                "Type": "You" if row["is_user_page"] else "Competitor"
            })

        ranking_df = pd.DataFrame(ranking_rows)
        st.dataframe(ranking_df, use_container_width=True, height=320)

        st.divider()
        st.header("📈 Your Page vs Competitor Average")

        comp = compare_user_vs_group(user_row, rows)
        if comp:
            comp_rows = []
            for feat, vals in comp.items():
                comp_rows.append({
                    "Feature": feat,
                    "Your Value": round(vals["user"], 3),
                    "Competitor Avg": round(vals["competitor_avg"], 3),
                    "Difference": round(vals["diff"], 3),
                    "Direction": "Above Avg" if vals["diff"] > 0 else "Below Avg" if vals["diff"] < 0 else "Equal"
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, height=340)

        st.divider()
        st.header("📋 SEO Gap Analysis")

        gap_rows = build_gap_table(user_row)
        gap_df = pd.DataFrame(gap_rows)
        gap_df["Your Value"] = gap_df["Your Value"].map(lambda x: round(x, 3))
        gap_df["Target"] = gap_df["Target"].map(lambda x: round(x, 3))
        gap_df["Gap %"] = gap_df["Gap %"].map(lambda x: round(x, 1))
        st.dataframe(gap_df, use_container_width=True, height=380)

        st.divider()
        st.header("💡 SEO Recommendations")

        recommendations = generate_seo_recommendations(user_row, rows)
        for rec in recommendations:
            st.info(rec)

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
                st.write(f"**Canonical present:** {bool(row['diagnostics'].get('canonical_url', ''))}")
                st.write(f"**Robots meta:** {row['diagnostics'].get('robots_meta', '') or 'N/A'}")
                st.write(f"**Schema present:** {row['diagnostics'].get('schema_present', False)}")
                st.write(f"**Open Graph present:** {row['diagnostics'].get('og_present', False)}")
                st.write(f"**Headings found:** {row['diagnostics'].get('headings_found', 0)}")
                st.write(f"**Body word count:** {row['diagnostics'].get('body_word_count', 0)}")
                st.write(f"**Response time (sec):** {row['diagnostics'].get('response_time_sec', 'N/A')}")

                stats = row["diagnostics"].get("structure_stats", {})
                if stats:
                    diag_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in stats.items()])
                    st.dataframe(diag_df, use_container_width=True, height=280)

        st.divider()
        st.header("🧾 Your Page Feature Source Summary")

        prov_rows = []
        for feat in sorted(user_row["features"].keys()):
            prov_rows.append({
                "Feature": feat,
                "Value": round(user_row["features"][feat], 3) if isinstance(user_row["features"][feat], (float, int)) else user_row["features"][feat],
                "Source": provenance.get(feat, "Derived / runtime-computed")
            })

        prov_df = pd.DataFrame(prov_rows)
        st.dataframe(prov_df, use_container_width=True, height=360)

if __name__ == "__main__":
    main()
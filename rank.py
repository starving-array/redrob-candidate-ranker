#!/usr/bin/env python3
"""
Redrob Hackathon — Candidate Ranker
====================================
Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Or with gzipped input:
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv

Constraints respected:
  - CPU only, no GPU
  - No network calls (no OpenAI / Anthropic / any API)
  - <5 minutes wall-clock on 100K candidates
  - <16 GB RAM (streams line-by-line, never loads full dataset)
"""

import argparse
import csv
import gzip
import heapq
import json
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — weights must sum to 1.0
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "career":    0.35,   # A: Career substance (most important)
    "skills":    0.25,   # B: Skills match quality
    "experience":0.20,   # C: Years-of-experience fit
    "location":  0.10,   # D: Location & logistics
    "education": 0.10,   # E: Education tier & field
}

TOP_N = 100
TODAY = date(2026, 6, 1)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL SETS
# ─────────────────────────────────────────────────────────────────────────────

# Titles that are hard disqualifiers regardless of skills listed
DISQUALIFY_TITLES = {
    "hr manager", "human resources", "marketing manager", "content writer",
    "accountant", "sales executive", "sales manager", "civil engineer",
    "mechanical engineer", "graphic designer", "operations manager",
    "project manager", "customer support", "customer service",
    "business analyst", "financial analyst", "recruiter", "teacher",
    "lawyer", "doctor", "nurse", "architect", "interior designer",
    "supply chain", "procurement", "qa engineer", "test engineer",
    "manual tester", "product designer", "ui designer", "ux designer",
    "devops engineer", "site reliability", "network engineer",
    "frontend engineer", "front-end engineer", "front end engineer",
    "mobile developer", "ios developer", "android developer",
    "java developer", ".net developer", "full stack developer",
    "fullstack developer", "web developer", "cloud engineer",
}

# Technical titles that need career substance check (not auto-disqualify)
NEED_SUBSTANCE_TITLES = {
    "software engineer", "backend engineer", "data engineer",
    "data scientist", "research engineer", "applied scientist",
}

# Pure services firms (entire career there = disqualify)
SERVICES_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "l&t infotech", "ltimindtree",
    "kpit", "persistent systems", "niit technologies",
}

# Must-have skills — production retrieval / ML infra
MUST_HAVE_SKILLS = {
    "embeddings", "embedding", "vector search", "vector database",
    "sentence transformers", "sentence-transformers",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "elasticsearch", "opensearch", "solr",
    "information retrieval", "retrieval", "rag",
    "learning to rank", "learning-to-rank", "ltr",
    "bm25", "hybrid search", "dense retrieval",
    "recommendation system", "recommendations", "ranking",
    "ndcg", "mrr", "map", "offline evaluation",
}

# Nice-to-have skills
NICE_TO_HAVE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning llms", "fine-tuning",
    "llm", "large language model", "transformers", "hugging face",
    "huggingface", "bert", "gpt", "t5",
    "pytorch", "tensorflow", "jax",
    "mlops", "kubeflow", "mlflow", "weights & biases", "wandb",
    "xgboost", "lightgbm", "catboost",
    "python", "pyspark", "spark",
    "machine learning", "deep learning", "nlp",
    "feature engineering", "a/b testing",
    "distributed systems", "kafka", "redis",
    "scikit-learn", "sklearn",
}

# Career description keywords that signal real retrieval/ML work
CAREER_SIGNAL_KEYWORDS = [
    # Retrieval & search
    ("vector", 3), ("embedding", 3), ("retrieval", 3), ("ranking", 3),
    ("search", 2), ("recommendation", 3), ("semantic search", 4),
    ("faiss", 3), ("pinecone", 3), ("elasticsearch", 2), ("opensearch", 2),
    ("bm25", 3), ("hybrid", 2), ("dense retrieval", 4), ("rag", 3),
    ("learning to rank", 4), ("ndcg", 4), ("mrr", 3),
    # ML in production
    ("production", 2), ("deployed", 2), ("shipped", 2), ("a/b test", 3),
    ("real user", 3), ("latency", 2), ("throughput", 2), ("scale", 1),
    ("fine-tun", 2), ("lora", 2), ("qlora", 2),
    # Product company signals
    ("product", 1), ("startup", 1), ("series", 1),
    # Strong ML signals
    ("sentence transformer", 3), ("xgboost", 2), ("lightgbm", 2),
    ("feature engineering", 2), ("offline evaluation", 3),
    ("online evaluation", 3), ("eval framework", 3),
    ("model serving", 2), ("inference", 2), ("mlops", 2),
]

# Product companies (bonus in career scoring)
PRODUCT_COMPANIES_KEYWORDS = {
    "swiggy", "zomato", "flipkart", "amazon", "google", "microsoft",
    "meta", "apple", "netflix", "uber", "ola", "razorpay", "paytm",
    "phonepe", "meesho", "cred", "dream11", "byju", "unacademy",
    "freshworks", "zoho", "atlassian", "stripe", "shopify",
    "airbnb", "doordash", "lyft", "twitter", "linkedin", "salesforce",
    "databricks", "snowflake", "mongodb", "elastic", "nvidia",
    "mad street den", "sarvam", "krutrim", "cohere", "openai",
    "anthropic", "mistral", "startup", "product company",
}

# Indian cities where JD is focused (Pune/Noida preferred)
TOP_CITIES = {"pune", "noida", "gurgaon", "gurugram", "bengaluru",
              "bangalore", "hyderabad", "mumbai", "delhi", "delhi ncr",
              "ncr", "new delhi", "chennai"}

PREFERRED_CITIES = {"pune", "noida"}

# Education fields that fit the role
GOOD_FIELDS = {
    "computer science", "cs", "software engineering", "computer engineering",
    "electrical engineering", "electronics", "information technology",
    "mathematics", "statistics", "data science", "machine learning",
    "artificial intelligence", "computational science",
}


# ─────────────────────────────────────────────────────────────────────────────
# HARD FILTER — returns True if candidate should be SKIPPED
# ─────────────────────────────────────────────────────────────────────────────

def is_disqualified(c: dict) -> bool:
    profile = c["profile"]
    career = c.get("career_history", [])
    signals = c.get("redrob_signals", {})

    # 1. Too little experience
    yoe = profile.get("years_of_experience", 0)
    if yoe < 2.5:
        return True

    # 2. Current title is clearly non-technical / wrong domain
    title_lower = profile.get("current_title", "").lower()
    for bad in DISQUALIFY_TITLES:
        if bad in title_lower:
            # Only pass if title has an explicit ML/AI qualifier
            if not any(w in title_lower for w in [
                "ml", "machine learning", "ai engineer", "data scientist",
                "nlp", "research engineer", "applied scientist",
                "recommendation", "ranking", "retrieval", "search engineer",
            ]):
                return True

    # 2b. Generic technical titles (software/backend/data engineer) need ML substance
    # to pass — otherwise they're just engineers with AI keywords in skills
    needs_substance = any(nt in title_lower for nt in NEED_SUBSTANCE_TITLES)
    if needs_substance and career:
        combined_desc = " ".join(
            (r.get("description", "") + " " + r.get("title", "")).lower()
            for r in career
        )
        ml_signal_count = 0
        for kw, _ in CAREER_SIGNAL_KEYWORDS[:15]:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, combined_desc):
                ml_signal_count += 1
        if ml_signal_count < 2:
            return True

    # 3. Entire career at services firms with no product company ever
    if career:
        companies_lower = [r.get("company", "").lower() for r in career]
        all_services = all(
            any(sf in co for sf in SERVICES_FIRMS)
            for co in companies_lower
        )
        if all_services and len(career) >= 1:
            return True

    # 4. Honeypot detection: impossible experience claims
    for role in career:
        dur = role.get("duration_months", 0)
        start_str = role.get("start_date", "")
        end_str = role.get("end_date", "")
        if start_str and dur:
            try:
                start = datetime.strptime(start_str[:10], "%Y-%m-%d").date()
                if end_str:
                    end = datetime.strptime(end_str[:10], "%Y-%m-%d").date()
                    if end < start:
                        return True
                    role_months = (end - start).days / 30.44
                else:
                    role_months = (TODAY - start).days / 30.44
                
                # If stated duration is significantly larger than calendar span
                if dur > role_months + 3.0:
                    return True
            except (ValueError, TypeError):
                pass

    # 4b. Honeypot: years of experience exceeds total career span
    start_dates = []
    for role in career:
        s_str = role.get("start_date", "")
        if s_str:
            try:
                dt = datetime.strptime(s_str[:10], "%Y-%m-%d").date()
                start_dates.append(dt)
            except (ValueError, TypeError):
                pass
    if start_dates:
        earliest_start = min(start_dates)
        max_possible_yoe = (TODAY - earliest_start).days / 365.25
        if yoe > max_possible_yoe + 1.0:
            return True

    # 5. Honeypot: expert/advanced skills with zero endorsements AND zero months
    skills = c.get("skills", [])
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", 0) == 0
    )
    if expert_zero >= 3:
        return True

    # 6. Located outside India with no relocation willingness
    country = profile.get("country", "").lower()
    willing = signals.get("willing_to_relocate", False)
    if country not in ("india", "in", "") and not willing:
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# SCORING COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def score_career(c: dict) -> float:
    """
    A — Career substance (0-100).
    Reads job descriptions, company names, summary and headline for real ML/retrieval work.
    """
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    
    summary_text = (profile.get("summary", "") + " " + profile.get("headline", "")).lower()
    summary_score = 0.0
    for kw, pts in CAREER_SIGNAL_KEYWORDS:
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, summary_text):
            summary_score += pts
    summary_score = min(summary_score, 40)

    if not career:
        return min((summary_score / 40.0) * 100, 100)

    total = 0.0
    max_possible = 0.0
    recency_weights = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.05]

    for i, role in enumerate(career[:10]):
        w = recency_weights[i]
        max_possible += w * 100

        desc_lower = (role.get("description", "") + " " + role.get("title", "")).lower()
        company_lower = role.get("company", "").lower()
        role_score = 0.0

        # Keyword signals in description
        for kw, pts in CAREER_SIGNAL_KEYWORDS:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, desc_lower):
                role_score += pts

        # Current title relevance boost
        is_curr = role.get("is_current", False) or (i == 0)
        if is_curr:
            title_lower = role.get("title", "").lower()
            relevance_keywords = ["search", "retrieval", "rag", "recommendation", "recsys", "ranking", "vector", "nlp", "information retrieval"]
            if any(rk in title_lower for rk in relevance_keywords):
                role_score += 15

        # Product company bonus
        if any(pc in company_lower for pc in PRODUCT_COMPANIES_KEYWORDS):
            role_score += 8

        # Services company penalty
        if any(sf in company_lower for sf in SERVICES_FIRMS):
            role_score = role_score * 0.4

        # Company size: mid-stage startup or large tech is better
        size = role.get("company_size", "")
        size_bonus = {
            "201-500": 3, "501-1000": 4, "1001-5000": 5,
            "5001-10000": 6, "10001+": 4,
        }.get(size, 1)
        role_score += size_bonus

        # Cap per role at 60 (unnormalized)
        role_score = min(role_score, 60)
        total += w * role_score

    if max_possible == 0:
        return 0.0
        
    base_career = (total / max_possible) * 100
    final_career = base_career * 0.7 + (min(summary_score, 40) / 40.0 * 100) * 0.3
    return min(final_career, 100.0)


def score_skills(c: dict) -> float:
    """
    B — Skills match quality (0-100).
    Uses endorsements × proficiency × duration as trust signal.
    Distinguishes must-have from nice-to-have.
    """
    skills = c.get("skills", [])
    if not skills:
        return 0.0

    must_score = 0.0
    nice_score = 0.0
    proficiency_weights = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}

    for s in skills:
        name_lower = s.get("name", "").lower()
        prof = proficiency_weights.get(s.get("proficiency", "beginner"), 0.3)
        endorse = s.get("endorsements", 0)
        duration = s.get("duration_months", 0)

        # Trust score: handles zero endorsements by adding 1.0 to the log multiplier
        trust = prof * (1.0 + math.log(1 + endorse)) * math.sqrt(max(duration, 0.1))
        trust = min(trust, 20.0)

        is_must = False
        for mh in MUST_HAVE_SKILLS:
            pattern = r'\b' + re.escape(mh) + r'\b'
            if re.search(pattern, name_lower):
                is_must = True
                break

        is_nice = False
        if not is_must:
            for nh in NICE_TO_HAVE_SKILLS:
                pattern = r'\b' + re.escape(nh) + r'\b'
                if re.search(pattern, name_lower):
                    is_nice = True
                    break

        if is_must:
            must_score += trust * 2.0  # must-have skills get double weight
        elif is_nice:
            nice_score += trust

    must_score_norm = min(must_score / 120.0, 1.0) * 70   # must-have skills worth 70 pts
    nice_score_norm = min(nice_score / 200.0, 1.0) * 30   # nice-to-have worth 30 pts
    return must_score_norm + nice_score_norm


def score_experience(c: dict) -> float:
    """
    C — Years of experience fit (0-100).
    Sweet spot is 5-9 years. Soft penalties outside the band.
    """
    yoe = c["profile"].get("years_of_experience", 0)

    if 5 <= yoe <= 9:
        return 100.0
    elif 4 <= yoe < 5:
        return 80.0
    elif 9 < yoe <= 11:
        return 75.0
    elif 3 <= yoe < 4:
        return 55.0
    elif 11 < yoe <= 13:
        return 55.0
    elif 13 < yoe <= 15:
        return 40.0
    else:
        return 20.0


def score_location(c: dict) -> float:
    """
    D — Location & logistics (0-100).
    Pune/Noida best. Key Indian cities good. Notice period bonus.
    """
    profile = c["profile"]
    signals = c.get("redrob_signals", {})

    loc_lower = (profile.get("location", "") + " " + profile.get("country", "")).lower()
    willing = signals.get("willing_to_relocate", False)
    notice = signals.get("notice_period_days", 90)

    # Location score
    if any(city in loc_lower for city in PREFERRED_CITIES):
        loc_score = 100
    elif any(city in loc_lower for city in TOP_CITIES):
        loc_score = 80
    elif "india" in loc_lower or profile.get("country", "").lower() in ("india", "in"):
        loc_score = 65
    elif willing:
        loc_score = 50
    else:
        loc_score = 10

    # Notice period modifier
    if notice <= 15:
        notice_bonus = 15
    elif notice <= 30:
        notice_bonus = 10
    elif notice <= 60:
        notice_bonus = 0
    elif notice <= 90:
        notice_bonus = -5
    else:
        notice_bonus = -15

    return max(0, min(100, loc_score + notice_bonus))


def score_education(c: dict) -> float:
    """
    E — Education (0-100).
    Tier and field of study matter. This is a soft signal.
    """
    education = c.get("education", [])
    if not education:
        return 30.0  # neutral, not a dealbreaker

    best = 0.0
    tier_scores = {"tier_1": 100, "tier_2": 80, "tier_3": 55, "tier_4": 35, "unknown": 40}

    for edu in education:
        tier = edu.get("tier", "unknown")
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()

        tier_s = tier_scores.get(tier, 40)

        # Field of study bonus
        field_bonus = 0
        if any(gf in field for gf in GOOD_FIELDS):
            field_bonus = 15

        # Degree level bonus
        degree_bonus = 0
        if any(d in degree for d in ("ph.d", "phd", "m.tech", "m.s.", "m.sc", "m.e.", "mtech", "ms")):
            degree_bonus = 10
        elif "b.tech" in degree or "b.e." in degree or "b.s." in degree or "be" == degree:
            degree_bonus = 5

        score = min(100, tier_s + field_bonus + degree_bonus)
        best = max(best, score)

    return best


def behavior_multiplier(c: dict) -> float:
    """
    F — Behavioral availability multiplier (0.2 to 1.3).
    Perfect-on-paper but inactive/unresponsive candidate gets penalised.
    """
    signals = c.get("redrob_signals", {})
    mult = 1.0

    # 1. Recency of activity
    last_active_str = signals.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str[:10], "%Y-%m-%d").date()
            days_since = (TODAY - last_active).days
            if days_since <= 14:
                mult *= 1.15
            elif days_since <= 30:
                mult *= 1.05
            elif days_since <= 60:
                mult *= 1.0
            elif days_since <= 120:
                mult *= 0.85
            elif days_since <= 180:
                mult *= 0.65
            else:
                mult *= 0.40
        except (ValueError, TypeError):
            pass

    # 2. Open to work
    if signals.get("open_to_work_flag", False):
        mult *= 1.1

    # 3. Recruiter response rate
    rr = signals.get("recruiter_response_rate", 0.5)
    if rr >= 0.7:
        mult *= 1.1
    elif rr >= 0.4:
        mult *= 1.0
    elif rr >= 0.2:
        mult *= 0.85
    else:
        mult *= 0.65

    # 4. Interview completion rate
    icr = signals.get("interview_completion_rate", 0.5)
    if icr >= 0.8:
        mult *= 1.05
    elif icr < 0.3:
        mult *= 0.85

    # 5. Profile completeness
    completeness = signals.get("profile_completeness_score", 50)
    if completeness >= 85:
        mult *= 1.05
    elif completeness < 40:
        mult *= 0.90

    # 6. GitHub activity (positive signal for technical depth)
    github = signals.get("github_activity_score", -1)
    if github >= 50:
        mult *= 1.08
    elif github >= 20:
        mult *= 1.03

    # 7. Saved by recruiters
    saves = signals.get("saved_by_recruiters_30d", 0)
    if saves >= 10:
        mult *= 1.10
    elif saves >= 3:
        mult *= 1.05

    # 8. Profile views received
    views = signals.get("profile_views_received_30d", 0)
    if views >= 150:
        mult *= 1.08
    elif views >= 50:
        mult *= 1.03

    # 9. Search appearance
    searches = signals.get("search_appearance_30d", 0)
    if searches >= 600:
        mult *= 1.06
    elif searches >= 200:
        mult *= 1.02

    # Clamp
    return max(0.2, min(1.3, mult))


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_score(c: dict) -> tuple[float, dict]:
    """Returns (composite_score, component_scores_dict)."""
    components = {
        "career":     score_career(c),
        "skills":     score_skills(c),
        "experience": score_experience(c),
        "location":   score_location(c),
        "education":  score_education(c),
    }
    base = sum(WEIGHTS[k] * v for k, v in components.items())
    mult = behavior_multiplier(c)
    final = (base * mult) / 100.0  # normalise to 0-1 float
    final = max(0.0, min(1.0, final))
    return final, components


# ─────────────────────────────────────────────────────────────────────────────
# REASONING GENERATOR — uses only facts from the profile, no hallucination
# ─────────────────────────────────────────────────────────────────────────────

def generate_reasoning(c: dict, components: dict, score: float) -> str:
    profile = c["profile"]
    career = c.get("career_history", [])
    signals = c.get("redrob_signals", {})
    skills = c.get("skills", [])

    title = profile.get("current_title", "N/A")
    company = profile.get("current_company", "N/A")
    yoe = profile.get("years_of_experience", 0)
    loc = profile.get("location", "N/A")
    country = profile.get("country", "India")
    notice = signals.get("notice_period_days", 90)
    rr = signals.get("recruiter_response_rate", 0.0)
    open_work = signals.get("open_to_work_flag", False)

    # 1. Match core skills
    matched_skills = []
    for s in skills:
        name = s.get("name", "")
        name_lower = name.lower()
        for mh in MUST_HAVE_SKILLS:
            if re.search(r'\b' + re.escape(mh) + r'\b', name_lower):
                matched_skills.append(name)
                break
    
    if not matched_skills:
        for s in skills:
            name = s.get("name", "")
            name_lower = name.lower()
            for nh in NICE_TO_HAVE_SKILLS:
                if re.search(r'\b' + re.escape(nh) + r'\b', name_lower):
                    matched_skills.append(name)
                    break

    skills_str = ", ".join(matched_skills[:3]) if matched_skills else "general ML skills"

    # 2. Extract specific achievements/responsibilities from career descriptions
    achievements = []
    for role in career[:2]:
        desc = role.get("description", "")
        # Look for sentences containing key production signals
        sentences = [s.strip() for s in desc.replace("\n", " ").split(".") if len(s.strip()) > 25]
        for sent in sentences:
            sent_lower = sent.lower()
            if any(k in sent_lower for k in ["shipped", "deploy", "optimiz", "percent", "scale", "ab test", "latency", "vector", "embed"]):
                clean_sent = sent.strip().rstrip(".")
                achievements.append(clean_sent)
                break
        if len(achievements) >= 2:
            break

    # 3. Handle Logistics & Concerns
    logistics = []
    if notice <= 30:
        logistics.append(f"quick start ({notice}d notice)")
    elif notice > 60:
        logistics.append(f"long notice ({notice}d)")
        
    if open_work:
        logistics.append("actively open to work")
    if rr > 0.8:
        logistics.append("highly responsive")
    elif rr < 0.2:
        logistics.append("low responsiveness")

    logistics_str = "; ".join(logistics)

    # 4. Formulate the rank-consistent tone
    if score >= 0.80:
        tone_start = f"Excellent fit for Senior AI Engineer with {yoe:.1f} YoE."
        fit_desc = f"Currently working as {title} @ {company} in {loc}."
        exp_highlight = ""
        if achievements:
            exp_highlight = f" Proven experience: '{achievements[0]}'."
        reasoning = f"{tone_start} {fit_desc} Strong in {skills_str}. {logistics_str}.{exp_highlight}"
    elif score >= 0.60:
        tone_start = f"Strong candidate with {yoe:.1f} YoE."
        fit_desc = f"Experience as {title} at {company}."
        reasoning = f"{tone_start} {fit_desc} Has skills in {skills_str}. {logistics_str}."
        if achievements:
            reasoning += f" Relevant project: '{achievements[0]}'."
    else:
        tone_start = f"Moderate fit with {yoe:.1f} YoE."
        fit_desc = f"Background as {title}."
        reasoning = f"{tone_start} {fit_desc} Experience matches some requirements (skills: {skills_str}). {logistics_str}."
        if achievements:
            reasoning += f" Project detail: '{achievements[0]}'."

    # Clean and cap size
    reasoning = re.sub(r'\s+', ' ', reasoning).strip()
    return reasoning[:380]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RANKING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def open_candidates(path: str):
    """Open either .jsonl or .jsonl.gz and yield parsed dicts."""
    p = Path(path)
    if p.suffix == ".gz":
        f = gzip.open(path, "rt", encoding="utf-8")
    else:
        f = open(path, "r", encoding="utf-8")
    try:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
    finally:
        f.close()


def rank_candidates(candidates_path: str, out_path: str):
    heap = []  # min-heap of (score, candidate_id, candidate, components)
    total = 0
    filtered = 0

    print(f"[rank.py] Reading candidates from: {candidates_path}", file=sys.stderr)

    for c in open_candidates(candidates_path):
        total += 1
        if total % 10000 == 0:
            print(f"[rank.py]   {total:,} processed, {filtered:,} filtered out, "
                  f"{len(heap)} in top-{TOP_N} heap...", file=sys.stderr)

        if is_disqualified(c):
            filtered += 1
            continue

        score, comps = compute_score(c)
        cid = c["candidate_id"]

        entry = (score, cid, c, comps)

        if len(heap) < TOP_N:
            heapq.heappush(heap, entry)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, entry)

    print(f"[rank.py] Done. {total:,} total, {filtered:,} filtered, "
          f"{len(heap)} in final pool.", file=sys.stderr)

    # Sort by score descending (rounded to 4 decimal places), then candidate_id ascending for ties
    top100 = sorted(heap, key=lambda x: (-round(x[0], 4), x[1]))


    # Write CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_pos, (score, cid, c, comps) in enumerate(top100, start=1):
            reasoning = generate_reasoning(c, comps, score)
            writer.writerow([cid, rank_pos, f"{score:.4f}", reasoning])

    print(f"[rank.py] Submission written to: {out_path}", file=sys.stderr)
    print(f"[rank.py] Top 5 candidates:", file=sys.stderr)
    for rank_pos, (score, cid, c, comps) in enumerate(top100[:5], start=1):
        p = c["profile"]
        print(f"  #{rank_pos}  {cid}  score={score:.4f}  "
              f"{p['current_title']} @ {p['current_company']}  "
              f"YoE={p['years_of_experience']}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob candidate ranker")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", default="./ArchForge.csv",
                        help="Output CSV path (defaults to ./ArchForge.csv)")

    args = parser.parse_args()

    rank_candidates(args.candidates, args.out)


if __name__ == "__main__":
    main()

# Redrob Hackathon — Candidate Ranker

Intelligent candidate ranking system for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

## What this does

Reads `candidates.jsonl` (100,000 candidates), ranks the top 100 best-fit candidates
for the Senior AI Engineer role at Redrob AI, and outputs a CSV submission file.

**No GPU, no API calls, no external network required.** Runs in ~60–90 seconds on CPU with 16 GB RAM.

## Files

| File                         | Purpose                                                       |
| ---------------------------- | ------------------------------------------------------------- |
| `rank.py`                  | Main ranker — run this to produce your submission CSV        |
| `test_ranker.py`           | Local sanity test against sample_candidates.json              |
| `validate_submission.py`   | Official format validator (provided by organisers, unchanged) |
| `submission_metadata.yaml` | Submission metadata                                           |
| `requirements.txt`         | Dependencies (none — pure Python stdlib)                     |

## Requirements

- Python 3.8 or higher
- No external packages required

Verify: `python3 --version`

## How to run

### Step 1 — Put the data file in this directory

```bash
# If you have the gzipped file:
cp /path/to/candidates.jsonl.gz .

# Or the unzipped file:
cp /path/to/candidates.jsonl .
```

### Step 2 — Run the ranker

```bash
python rank.py --candidates ./candidates.jsonl.gz --out ./team_xxx.csv
```

Replace `team_xxx.csv` with your actual registered team ID.

Expected output (stderr):

```
[rank.py] Reading candidates from: ./candidates.jsonl.gz
[rank.py]   10,000 processed, 7,200 filtered out, 100 in top-100 heap...
[rank.py]   20,000 processed, 14,400 filtered out, 100 in top-100 heap...
...
[rank.py] Done. 100,000 total, 72,000 filtered, 100 in final pool.
[rank.py] Submission written to: ./team_xxx.csv
[rank.py] Top 5 candidates:
  #1  CAND_XXXXXXX  score=0.XXXX  ...
```

### Step 3 — Validate before submitting

```bash
python validate_submission.py team_xxx.csv
```

Expected: `Submission is valid.`

### Step 4 — Test on sample data (optional, recommended)

```bash
python test_ranker.py --sample ./sample_candidates.json
```

## Ranking approach

### Overview

Five-component weighted scoring, multiplied by a behavioral availability multiplier.

```
Final score = (A×0.35 + B×0.25 + C×0.20 + D×0.10 + E×0.10) × behavior_multiplier
```

### Stage 1 — Hard filters (applied before scoring)

Candidates are immediately disqualified if:

- Under 2.5 years experience
- Current title is clearly non-ML (HR Manager, Marketing Manager, Civil Engineer, etc.)
- Entire career history is at pure services firms (TCS, Infosys, Wipro, etc.)
- Honeypot signals: impossible date/duration math, or 5+ "expert" skills with 0 endorsements AND 0 months used
- Located outside India with no willingness to relocate

### Component A — Career substance (35%)

Reads each role's description and company name. Rewards:

- Retrieval / vector / embedding / ranking / recommendation systems work
- Production deployment signals ("shipped", "A/B test", "real users", "latency")
- Product companies (Swiggy, Zomato, Razorpay, etc.)
- Applies recency weighting (most recent role counts most)
- Penalises services firms at 40% of raw score

### Component B — Skills quality (25%)

Distinguishes real skill proficiency from keyword stuffing:

- Trust formula: `proficiency_weight × log(1 + endorsements) × sqrt(duration_months)`
- Must-have skills (FAISS, Pinecone, Embeddings, RAG, etc.) get 2× weight
- Nice-to-have skills (PyTorch, LLM fine-tuning, XGBoost, etc.) get 1× weight

### Component C — Experience fit (20%)

Sweet spot 5–9 years = 100 points. Soft penalties outside the band.

### Component D — Location & logistics (10%)

Pune/Noida = 100, other key Indian cities = 80, rest of India = 65, willing to relocate = 50, abroad + no relocate = 10. Notice period ≤30 days gets a bonus.

### Component E — Education (10%)

Tier 1 institutions (IIT, IISc, etc.) score highest. CS/EE/Math fields get a bonus.

### Behavioral multiplier (F)

Applied as a multiplier (0.2× to 1.3×) on the base score.
Incorporates: recency of last login, open-to-work flag, recruiter response rate, interview completion rate, profile completeness, GitHub activity score.

A perfect-on-paper candidate who hasn't logged in for 6+ months with a 5% response rate gets multiplied by ~0.3, effectively deprioritising them despite their skills.

### Honeypot avoidance

Our hard filters naturally reject honeypots because:

- Impossible durations (8 years at a company founded 3 years ago) are caught by date math
- "Expert" skills with 0 endorsements and 0 months used signal stuffing
- The career substance check rejects profiles with strong skills but no ML career history

### Performance

- Memory: O(1) — streams candidates one line at a time, holds only top-100 in a min-heap
- Time: ~60–90 seconds for 100K candidates on a modern CPU
- No GPU, no API calls, no network access

## Compute environment

See `submission_metadata.yaml` for exact environment details.

Single reproduce command:

```bash
python rank.py --candidates ./candidates.jsonl.gz --out ./team_xxx.csv
```

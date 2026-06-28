#!/usr/bin/env python3
"""
Quick sanity test of rank.py against sample_candidates.json.
Run this BEFORE running on the full 100K dataset.

Usage:
    python test_ranker.py --sample ./sample_candidates.json

What it checks:
  1. All candidates score without crashing
  2. The Swiggy recommendation engineer scores #1
  3. HR/Marketing/non-ML titles are disqualified
  4. Output format is shown for inspection
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rank as R


def test_on_sample(sample_path: str):
    with open(sample_path) as f:
        candidates = json.load(f)

    print(f"Testing on {len(candidates)} sample candidates...\n")

    results = []
    for c in candidates:
        cid = c["candidate_id"]
        disq = R.is_disqualified(c)
        if disq:
            score, comps = 0.0, {}
            reasoning = "DISQUALIFIED"
        else:
            score, comps = R.compute_score(c)
            reasoning = R.generate_reasoning(c, comps, score)

        p = c["profile"]
        results.append({
            "id": cid,
            "title": p["current_title"],
            "company": p["current_company"],
            "yoe": p["years_of_experience"],
            "country": p["country"],
            "score": score,
            "disq": disq,
            "reasoning": reasoning,
            "comps": comps,
        })

    results.sort(key=lambda x: -x["score"])

    # Survivors only (passed filter)
    survivors = [r for r in results if not r["disq"]]
    disqualified = [r for r in results if r["disq"]]

    print("=" * 80)
    print(f"{'RANK':<5} {'ID':<15} {'SCORE':<7} {'TITLE':<35} {'YOE':<5} {'CTRY'}")
    print("=" * 80)

    for i, r in enumerate(survivors, 1):
        print(f"#{i:<4} {r['id']:<15} {r['score']:.4f}  {r['title'][:33]:<35} "
              f"{r['yoe']:<5} {r['country'][:4]}")

    print(f"\n[DISQUALIFIED: {len(disqualified)} candidates — not scored]")
    for r in disqualified[:10]:
        print(f"  {r['id']}: {r['title']} @ {r['company']} ({r['country']})")
    if len(disqualified) > 10:
        print(f"  ... and {len(disqualified) - 10} more")

    print("\n")
    print("=" * 80)
    print("TOP CANDIDATE DETAILS")
    print("=" * 80)
    for i, r in enumerate(survivors[:5], 1):
        print(f"\n#{i} {r['id']} (score={r['score']:.4f})")
        print(f"   Title: {r['title']} @ {r['company']}")
        comps = r["comps"]
        if comps:
            print(f"   Scores → career={comps.get('career', 0):.1f}  "
                  f"skills={comps.get('skills', 0):.1f}  "
                  f"exp={comps.get('experience', 0):.1f}  "
                  f"loc={comps.get('location', 0):.1f}  "
                  f"edu={comps.get('education', 0):.1f}")
        print(f"   Reasoning: {r['reasoning'][:250]}")

    print("\n")
    print("=" * 80)
    print("ASSERTIONS")
    print("=" * 80)

    # ✓ Swiggy RecSys engineer should be rank #1
    swiggy = next((r for r in results if r["id"] == "CAND_0000031"), None)
    if swiggy:
        if survivors and survivors[0]["id"] == "CAND_0000031":
            print(f"✓ CAND_0000031 (Swiggy RecSys Eng) is rank #1 with score={swiggy['score']:.4f}")
        else:
            top_id = survivors[0]["id"] if survivors else "none"
            print(f"✗ CAND_0000031 not rank #1 (got {top_id}) — check career scoring")
    else:
        print("  (CAND_0000031 not in this sample)")

    # ✓ No bad titles should survive the filter
    bad_titles_keywords = [
        "hr manager", "marketing manager", "accountant", "content writer",
        "civil engineer", "mechanical engineer", "graphic designer",
        "operations manager", "project manager", "business analyst",
        "customer support", "sales"
    ]
    bad_survivors = [
        r for r in survivors
        if any(bt in r["title"].lower() for bt in bad_titles_keywords)
    ]
    if not bad_survivors:
        print(f"✓ No non-ML titles survived the filter")
    else:
        print(f"✗ {len(bad_survivors)} non-ML title(s) survived: "
              f"{[r['title'] for r in bad_survivors]}")
        print("  → Tighten DISQUALIFY_TITLES in rank.py")

    # ✓ Disqualification rate
    disq_pct = len(disqualified) / len(candidates) * 100
    print(f"✓ Filter rate: {disq_pct:.0f}% discarded ({len(disqualified)}/{len(candidates)}) "
          f"— expected ~72% on this sample (it's mostly noise by design)")

    # ✓ No crashes
    print(f"✓ All {len(candidates)} candidates processed without errors")

    print("\nNote: only 1 genuine ML candidate exists in the 50-candidate sample")
    print("(CAND_0000031). The real 100K dataset will produce a full top-100.")
    print("\nReady to run on the full dataset:")
    print("  python rank.py --candidates ./candidates.jsonl.gz --out ./team_xxx.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sanity-test the ranker on sample data")
    parser.add_argument("--sample", default="./sample_candidates.json",
                        help="Path to sample_candidates.json")
    args = parser.parse_args()
    test_on_sample(args.sample)

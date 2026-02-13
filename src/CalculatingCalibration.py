"""
Deterministic distance-based prediction for Tuesday 3rd February 2026.
Uses only Tuesday/Friday draws from data/22.csv.

Rules encoded:
- P1..P5 are ordered, unique, range 1..50.
- Fixed distances: P2-P1=5, P3-P2=2, P4-P3=11, P5-P4=10.
- P6..P7 are ordered, unique, range 1..12.

Outputs most/least likely positions using historical frequencies and distance patterns.
Optional LLM refinement via llm_agent.llm_refine (disabled by default).
"""

import json
from collections import Counter
from datetime import datetime

import numpy as np

from load import load_data
from features import distance_features
from llm_agent import llm_refine


TARGET_DATE = datetime(2026, 2, 3)
USE_LLM_REFINEMENT = False


DIST_P1_P2 = 5
DIST_P2_P3 = 2
DIST_P3_P4 = 11
DIST_P4_P5 = 10
SUM_DISTANCES = DIST_P1_P2 + DIST_P2_P3 + DIST_P3_P4 + DIST_P4_P5


def ordered_unique(sequence):
    return list(sorted(set(sequence)))


def calc_distances(positions):
    return [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]


def generate_p1_5_candidates():
    candidates = []
    for p1 in range(1, 51 - SUM_DISTANCES):
        p2 = p1 + DIST_P1_P2
        p3 = p2 + DIST_P2_P3
        p4 = p3 + DIST_P3_P4
        p5 = p4 + DIST_P4_P5
        candidates.append([p1, p2, p3, p4, p5])
    return candidates


def score_p1_5_candidate(candidate, freqs_by_col):
    return sum(freqs_by_col[i].get(value, 0) for i, value in enumerate(candidate))


def score_p6_7_candidate(candidate, freqs_by_col, distance_counts):
    p6, p7 = candidate
    distance_score = distance_counts.get(p7 - p6, 0)
    return freqs_by_col[0].get(p6, 0) + freqs_by_col[1].get(p7, 0) + distance_score


def main():
    df = load_data("data/22.csv")
    df = distance_features(df)

    p1_5_candidates = generate_p1_5_candidates()

    freq_p1 = Counter(df["P1"].astype(int))
    freq_p2 = Counter(df["P2"].astype(int))
    freq_p3 = Counter(df["P3"].astype(int))
    freq_p4 = Counter(df["P4"].astype(int))
    freq_p5 = Counter(df["P5"].astype(int))
    p1_5_freqs = [freq_p1, freq_p2, freq_p3, freq_p4, freq_p5]

    p1_5_scores = [score_p1_5_candidate(c, p1_5_freqs) for c in p1_5_candidates]
    best_p1_5_idx = int(np.argmax(p1_5_scores))
    worst_p1_5_idx = int(np.argmin(p1_5_scores))

    most_likely_p1_5 = p1_5_candidates[best_p1_5_idx]
    least_likely_p1_5 = p1_5_candidates[worst_p1_5_idx]

    freq_p6 = Counter(df["P6"].astype(int))
    freq_p7 = Counter(df["P7"].astype(int))
    p6_7_freqs = [freq_p6, freq_p7]

    distance_counts = Counter(df["D67"].astype(int))

    p6_7_candidates = [[p6, p7] for p6 in range(1, 12) for p7 in range(p6 + 1, 13)]
    p6_7_scores = [
        score_p6_7_candidate(c, p6_7_freqs, distance_counts) for c in p6_7_candidates
    ]

    best_p6_7_idx = int(np.argmax(p6_7_scores))
    worst_p6_7_idx = int(np.argmin(p6_7_scores))

    most_likely_p6_7 = p6_7_candidates[best_p6_7_idx]
    least_likely_p6_7 = p6_7_candidates[worst_p6_7_idx]

    most_likely = {
        "positions": {
            "P1": most_likely_p1_5[0],
            "P2": most_likely_p1_5[1],
            "P3": most_likely_p1_5[2],
            "P4": most_likely_p1_5[3],
            "P5": most_likely_p1_5[4],
            "P6": most_likely_p6_7[0],
            "P7": most_likely_p6_7[1],
        },
        "distances": {
            "P1_P2": DIST_P1_P2,
            "P2_P3": DIST_P2_P3,
            "P3_P4": DIST_P3_P4,
            "P4_P5": DIST_P4_P5,
            "P6_P7": most_likely_p6_7[1] - most_likely_p6_7[0],
        },
    }

    least_likely = {
        "positions": {
            "P1": least_likely_p1_5[0],
            "P2": least_likely_p1_5[1],
            "P3": least_likely_p1_5[2],
            "P4": least_likely_p1_5[3],
            "P5": least_likely_p1_5[4],
            "P6": least_likely_p6_7[0],
            "P7": least_likely_p6_7[1],
        },
        "distances": {
            "P1_P2": DIST_P1_P2,
            "P2_P3": DIST_P2_P3,
            "P3_P4": DIST_P3_P4,
            "P4_P5": DIST_P4_P5,
            "P6_P7": least_likely_p6_7[1] - least_likely_p6_7[0],
        },
    }

    result = {
        "target_date": TARGET_DATE.strftime("%Y-%m-%d"),
        "day_of_week": "Tuesday",
        "most_likely": most_likely,
        "least_likely": least_likely,
    }

    if USE_LLM_REFINEMENT:
        result = llm_refine(result)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

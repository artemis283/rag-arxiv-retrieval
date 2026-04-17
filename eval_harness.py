import numpy as np
import json
import requests

API_URL = "http://91.98.227.123:8000"

def precision_at_k(retrieved_ids, relevant_ids):
    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)
    if len(retrieved) == 0:
        return 0.0
    return len(retrieved & relevant) / len(retrieved)

def recall_at_k(retrieved_ids, relevant_ids):
    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)
    if len(relevant) == 0:
        return 0.0
    return len(retrieved & relevant) / len(relevant)

def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0

def evaluate_retrieval(eval_set, retrieve_fn, k=5):
    results = []
    for item in eval_set:
        retrieved = retrieve_fn(item["query"], k)
        relevant = item["relevant_chunk_ids"]
        results.append({
            "query": item["query"],
            "precision": precision_at_k(retrieved, relevant),
            "recall": recall_at_k(retrieved, relevant),
            "rr": reciprocal_rank(retrieved, relevant),
            "retrieved": retrieved,
            "relevant": relevant,
        })
    avg_precision = np.mean([r["precision"] for r in results])
    avg_recall = np.mean([r["recall"] for r in results])
    mrr = np.mean([r["rr"] for r in results])
    return {
        "per_query": results,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "mrr": mrr,
        "k": k,
    }


def print_report(results, label=""):
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    print(f"\n  k = {results['k']}")
    print(f"  Queries: {len(results['per_query'])}")
    print(f"\n  {'Query':<50} {'P@k':>6} {'R@k':>6} {'RR':>6}")
    print(f"  {'-'*68}")

    for r in results["per_query"]:
        query_short = r["query"][:48]
        print(f"  {query_short:<50} {r['precision']:>6.2f} {r['recall']:>6.2f} {r['rr']:>6.2f}")

    print(f"  {'-'*68}")
    print(f"  {'Average':<50} {results['avg_precision']:>6.2f} {results['avg_recall']:>6.2f} {results['mrr']:>6.2f}")

# --- Our adapter layer ---

def make_chunk_id(arxiv_id, section):
    return f"{arxiv_id}::{section}"

def retrieve_fn(query, k):
    resp = requests.get(f"{API_URL}/search", params={"q": query, "top_k": k})
    results = resp.json()["results"]
    return [make_chunk_id(r["arxiv_id"], r["section"]) for r in results]

def convert_eval_set(raw_eval_set):
    converted = []
    for item in raw_eval_set:
        relevant_ids = [
            make_chunk_id(c["arxiv_id"], c["section"])
            for c in item["relevant_chunks"]
        ]
        converted.append({
            "query": item["query"],
            "difficulty": item["difficulty"],
            "relevant_chunk_ids": relevant_ids,
        })
    return converted

# --- Run evaluation ---

with open("eval_set.json") as f:
    raw_eval_set = json.load(f)

eval_set = convert_eval_set(raw_eval_set)

# Baseline runs at different k values
for k in [1, 3, 5, 10, 20]:
    results = evaluate_retrieval(eval_set, retrieve_fn, k=k)
    print_report(results, f"Baseline: k={k}")

# Breakdown by difficulty at k=5
for difficulty in ["easy", "medium", "hard"]:
    subset = [e for e in eval_set if e["difficulty"] == difficulty]
    if not subset:
        continue
    results = evaluate_retrieval(subset, retrieve_fn, k=5)
    print_report(results, f"Difficulty: {difficulty.upper()} (k=5)")
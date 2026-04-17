import requests
import json

API_URL = "http://91.98.227.123:8000"
EVAL_FILE = "eval_set.json"

with open(EVAL_FILE) as f:
    eval_set = json.load(f)

print(f"Loaded {len(eval_set)} existing eval entries.\n")

queries = [
    {"query": "How does LookaheadKV decide which KV cache entries to evict?", "difficulty": "medium"},
    {"query": "What is the difference between weight-only and weight-activation quantization?", "difficulty": "medium"},
    {"query": "How does Quasar accelerate the verification phase of speculative decoding?", "difficulty": "hard"},
    {"query": "What datasets were used to train the Bielik language model?", "difficulty": "hard"},
    {"query": "How does per-channel normalization improve KV cache quantization?", "difficulty": "hard"},
]

print("=" * 70)
print("EVAL SET BUILDER - EXTRA QUERIES")
print("Type chunk numbers separated by commas (e.g. 1,3,4)")
print("Type 's' to skip, 'q' to quit.")
print("=" * 70)

for q in queries:
    if any(e["query"] == q["query"] for e in eval_set):
        print(f"\n[SKIP] Already evaluated: {q['query']}")
        continue

    print(f"\n{'=' * 70}")
    print(f"QUERY ({q['difficulty']}): {q['query']}")
    print("=" * 70)

    resp = requests.get(f"{API_URL}/search", params={"q": q["query"], "top_k": 10})
    data = resp.json()

    for i, r in enumerate(data["results"], 1):
        print(f"\n  [{i}] id={r['arxiv_id']} | §{r['section']} | sim={r['similarity']}")
        print(f"      {r['content'][:200]}...")

    print(f"\nWhich chunks are relevant? (e.g. 1,2,4 or 's' to skip, 'q' to quit)")
    choice = input("> ").strip()

    if choice.lower() == 'q':
        break
    if choice.lower() == 's':
        continue

    try:
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        relevant = []
        for idx in indices:
            r = data["results"][idx]
            relevant.append({
                "arxiv_id": r["arxiv_id"],
                "section": r["section"],
                "similarity": r["similarity"],
            })

        eval_set.append({
            "query": q["query"],
            "difficulty": q["difficulty"],
            "relevant_chunks": relevant,
            "num_relevant": len(relevant),
        })
        print(f"  ✓ Saved {len(relevant)} relevant chunks")
    except (ValueError, IndexError):
        print("  ✗ Invalid input, skipping")

with open(EVAL_FILE, "w") as f:
    json.dump(eval_set, f, indent=2)
print(f"\nSaved {len(eval_set)} eval entries to {EVAL_FILE}")
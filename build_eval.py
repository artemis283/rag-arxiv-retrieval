"""
Semi-automated eval set builder.
Asks your RAG system questions, shows you the retrieved chunks,
and lets you mark which ones are relevant.
"""
import requests
import json

API_URL = "http://91.98.227.123:8000"
EVAL_FILE = "eval_set.json"

# Load existing eval set if it exists
try:
    with open(EVAL_FILE) as f:
        eval_set = json.load(f)
    print(f"Loaded {len(eval_set)} existing eval entries.\n")
except FileNotFoundError:
    eval_set = []

queries = [
    # EASY - terminology matches closely
    {"query": "What is asymmetric quantization?", "difficulty": "easy"},
    {"query": "How does speculative decoding work?", "difficulty": "easy"},
    {"query": "What is a mixture of experts model?", "difficulty": "easy"},
    {"query": "What is KV cache eviction?", "difficulty": "easy"},
    {"query": "What is retrieval-augmented generation?", "difficulty": "easy"},
    {"query": "What is knowledge distillation?", "difficulty": "easy"},
    {"query": "What is structured pruning?", "difficulty": "easy"},

    # MEDIUM - different phrasing
    {"query": "How can you reduce memory usage during LLM inference?", "difficulty": "medium"},
    {"query": "What methods exist for compressing key-value pairs in transformers?", "difficulty": "medium"},
    {"query": "How do you make a smaller model mimic a larger one?", "difficulty": "medium"},
    {"query": "What are the tradeoffs of extreme low-bit quantization?", "difficulty": "medium"},
    {"query": "How can draft models be trained for faster decoding?", "difficulty": "medium"},
    {"query": "What benchmarks exist for evaluating RAG systems?", "difficulty": "medium"},
    {"query": "How do you decide which layers to fine-tune in a large model?", "difficulty": "medium"},

    # HARD - specific details buried in chunks
    {"query": "What optimiser was used for fine-tuning Bielik?", "difficulty": "hard"},
    {"query": "What is the effective attention dimensionality for language modeling?", "difficulty": "hard"},
    {"query": "How does InnerQ handle the prefill phase differently from decode?", "difficulty": "hard"},
    {"query": "What acceptance rate does LK loss optimise for?", "difficulty": "hard"},
    {"query": "How does EAGLE-Pangu handle tree attention on Ascend NPUs?", "difficulty": "hard"},
    {"query": "What is the role of the proximal point in oracle-robust alignment?", "difficulty": "hard"},
    {"query": "How does TaSR-RAG use taxonomy for structured reasoning?", "difficulty": "hard"},
    {"query": "What sparsity levels were tested in optimal expert-attention allocation?", "difficulty": "hard"},
]

print("=" * 70)
print("EVAL SET BUILDER")
print("For each query, review the retrieved chunks and mark relevant ones.")
print("Type chunk numbers separated by commas (e.g. 1,3,4)")
print("Type 's' to skip, 'q' to quit and save.")
print("=" * 70)

for q in queries:
    # Skip if already evaluated
    if any(e["query"] == q["query"] for e in eval_set):
        print(f"\n[SKIP] Already evaluated: {q['query']}")
        continue

    print(f"\n{'=' * 70}")
    print(f"QUERY ({q['difficulty']}): {q['query']}")
    print("=" * 70)

    # Call search endpoint with top_k=10 for more options
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

# Save
with open(EVAL_FILE, "w") as f:
    json.dump(eval_set, f, indent=2)
print(f"\nSaved {len(eval_set)} eval entries to {EVAL_FILE}")
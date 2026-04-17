import json
import numpy as np
import tiktoken
import pymupdf
from pathlib import Path
from sentence_transformers import SentenceTransformer

enc = tiktoken.get_encoding("cl100k_base")
model = SentenceTransformer("all-MiniLM-L6-v2")

PDF_DIR = Path("/Users/admin/Downloads/arxiv_corpus_2026/pdfs")
METADATA = Path("/Users/admin/Downloads/arxiv_corpus_2026/metadata.json")

def extract_pdf(pdf_path):
    doc = pymupdf.open(str(pdf_path))
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text

def chunk_text(text, paper_id, chunk_size=256, overlap_ratio=0.2):
    overlap = int(chunk_size * overlap_ratio)
    tokens = enc.encode(text)
    chunks = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append({
            "chunk_id": f"{paper_id}_{chunk_size}_{idx}",
            "paper_id": paper_id,
            "text": enc.decode(chunk_tokens),
        })
        idx += 1
        start += chunk_size - overlap
        if end >= len(tokens):
            break
    return chunks

def build_corpus(chunk_size=256):
    with open(METADATA) as f:
        papers = json.load(f)
    all_chunks = []
    for paper in papers:
        pdf_path = PDF_DIR / paper["pdf_filename"]
        if not pdf_path.exists():
            continue
        text = extract_pdf(pdf_path)
        paper_chunks = chunk_text(text, paper["arxiv_id"], chunk_size)
        for c in paper_chunks:
            c["paper_title"] = paper["title"]
        all_chunks.extend(paper_chunks)
    return all_chunks

# Sample queries to inspect results manually
SAMPLE_QUERIES = [
    "How does Quasar accelerate the verification phase of speculative decoding?",
    "How does per-channel normalization improve KV cache quantization?",
    "What is the role of the proximal point in oracle-robust alignment?",
    "What are the tradeoffs of extreme low-bit quantization?",
    "How does TaSR-RAG use taxonomy for structured reasoning?",
]

print("=" * 80)
print("EXPERIMENT 2: Varying Chunk Size")
print("=" * 80)
print(f"\n  {'Size':>6} | {'Chunks':>7} | {'Avg tokens':>10}")
print(f"  {'-'*30}")

all_results = {}

for size in [128, 256, 512, 1024]:
    chunks = build_corpus(chunk_size=size)
    texts = [c["text"] for c in chunks]

    avg_tokens = np.mean([len(enc.encode(t)) for t in texts])
    print(f"  {size:>6} | {len(chunks):>7} | {avg_tokens:>10.0f}")

    # Embed all chunks
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # Search for sample queries
    all_results[size] = {}
    for query in SAMPLE_QUERIES:
        query_vec = model.encode([query], normalize_embeddings=True)[0]
        scores = embeddings @ query_vec
        top_indices = np.argsort(scores)[::-1][:5]

        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": chunks[idx]["chunk_id"],
                "paper_id": chunks[idx]["paper_id"],
                "score": float(scores[idx]),
                "preview": chunks[idx]["text"][:150],
            })
        all_results[size][query] = results

# Print comparison for each sample query
print("\n" + "=" * 80)
print("SAMPLE QUERY COMPARISON (top 3 per chunk size)")
print("=" * 80)

for query in SAMPLE_QUERIES:
    print(f"\n  Q: {query}")
    print(f"  {'-'*75}")
    for size in [128, 256, 512, 1024]:
        print(f"\n    chunk_size={size}:")
        for r in all_results[size][query][:3]:
            print(f"      [{r['score']:.3f}] {r['paper_id']} | {r['preview'][:100]}...")
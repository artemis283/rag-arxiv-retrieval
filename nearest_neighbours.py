from sklearn.cluster import KMeans
import numpy as np
import time

def normalise(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

## Simulate collections of different sizes
for n in [1_000, 10_000, 100_000, 1_000_000]:
    vectors = normalise(np.random.randn(n, 384).astype(np.float32))
    query = np.random.randn(384).astype(np.float32)
    query = query / np.linalg.norm(query)

    start = time.perf_counter()
    scores = vectors @ query  # dot product against every vector
    top_idx = np.argpartition(-scores, 5)[:5]
    elapsed = time.perf_counter() - start

    print(f"{n:>10,} vectors: {elapsed:.4f}s")

## Create some data to work with
np.random.seed(42)
n_vectors = 5000
dim = 32
vectors = normalise(np.random.randn(n_vectors, dim).astype(np.float32))

## Step 1: Cluster the vectors
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(vectors)
centroids = kmeans.cluster_centers_  # not unit vectors, but we only use them to rank clusters
labels = kmeans.labels_

## Step 2: Build an inverted index (cluster_id -> list of vector indices)
inverted_index = {}
for idx, label in enumerate(labels):
    inverted_index.setdefault(label, []).append(idx)

print(f"Built {n_clusters} clusters")
for cluster_id in sorted(inverted_index):
    print(f"  Cluster {cluster_id}: {len(inverted_index[cluster_id])} vectors")


def ivfflat_search(query, centroids, inverted_index, vectors, nprobe=3, top_k=5):
    # Find the closest centroids by dot product (highest = closest for unit vectors)
    centroid_scores = centroids @ query
    closest_clusters = np.argsort(-centroid_scores)[:nprobe]

    # Search only those clusters
    candidates = []
    for cluster_id in closest_clusters:
        for vec_idx in inverted_index[cluster_id]:
            score = vectors[vec_idx] @ query
            candidates.append((score, vec_idx))

    candidates.sort(reverse=True)  # highest dot product first
    return candidates[:top_k]

query = normalise(np.random.randn(1, dim).astype(np.float32))[0]

## IVFFlat search
results = ivfflat_search(query, centroids, inverted_index, vectors, nprobe=3)
print("\nIVFFlat results (nprobe=3):")
for score, idx in results:
    print(f"  Vector {idx}: similarity {score:.4f}")

## Compare with brute force
all_scores = [(vectors[i] @ query, i) for i in range(len(vectors))]
all_scores.sort(reverse=True)
print("\nBrute force top 5:")
for score, idx in all_scores[:5]:
    print(f"  Vector {idx}: similarity {score:.4f}")


brute_force_top5 = set(idx for _, idx in all_scores[:5])

print("nprobe | clusters searched | recall")
print("-------+------------------+-------")
for nprobe in [1, 2, 3, 5, 10]:
    results = ivfflat_search(query, centroids, inverted_index, vectors, nprobe=nprobe)
    result_set = set(idx for _, idx in results)
    recall = len(result_set & brute_force_top5) / len(brute_force_top5)
    centroid_scores = centroids @ query
    closest = np.argsort(-centroid_scores)[:nprobe]
    n_searched = sum(len(inverted_index[c]) for c in closest)
    print(f"  {nprobe:>4}  | {n_searched:>16} | {recall:.0%}")
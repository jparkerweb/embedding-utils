# embedding-utils

**Vector math, similarity search, ANN indexing, clustering, async pipelines, evaluation metrics, and multi-provider embedding generation -- zero dependencies, full TypeScript, one import.**

<img src="https://raw.githubusercontent.com/jparkerweb/embedding-utils/refs/heads/main/embedding-utils.jpg" alt="embedding-utils" style="max-width: 885px;" />

Build semantic search, RAG pipelines, recommendation engines, duplicate detection, and document clustering without pulling in heavy ML frameworks or vector databases.

---

## Table of Contents

- [Why embedding-utils?](#why-embedding-utils)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Providers](#providers)
- [API Reference](#api-reference)
  - [Vector Math](#vector-math)
  - [Search](#search)
  - [HNSW Search](#hnsw-search-approximate-nearest-neighbor)
  - [Hybrid Search (RRF)](#hybrid-search-rrf--score-normalization)
  - [Clustering](#clustering)
  - [HDBSCAN Clustering](#hdbscan-clustering)
  - [Aggregation](#aggregation)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Async Pipeline](#async-pipeline)
  - [Quantization](#quantization)
  - [Dimensionality Reduction](#dimensionality-reduction-random-projection)
  - [Markdown Chunking](#markdown-aware-chunking)
  - [Storage](#storage)
  - [Model Management](#model-management)
  - [High-Level APIs](#high-level-apis)
- [Migration Guide (v0.2 → v0.3)](#migration-guide-v02--v03)
- [Caveats & Best Practices](#caveats--best-practices)
- [API Reference (all exports)](#api-reference-all-exports)
- [Common Patterns](#common-patterns)
- [TypeScript](#typescript)
- [Error Handling](#error-handling)
- [License](#license)

---

## Why embedding-utils?

- **Zero production dependencies** -- nothing to audit, nothing to break
- **Provider-agnostic** -- swap between local ONNX, OpenAI, Cohere, and Google Vertex with one line
- **Complete toolkit** -- math, search, clustering, quantization, caching, and serialization in a single package
- **TypeScript-first** -- strict types, full inference, no `@types` package needed
- **Production-ready** -- HNSW approximate search, async pipelines with rate limiting, and retrieval evaluation metrics
- **Tree-shakeable** -- dual ESM + CJS build; import only what you use

Use this when you need embedding operations without a full ML framework, want to switch providers without refactoring, or need vector search and clustering without a vector database.

---

## Installation

Requires **Node.js 18+**. Supports both ESM and CommonJS.

```bash
npm install embedding-utils
```

For local ONNX inference (no API key needed), also install the optional peer dependency:

```bash
npm install @huggingface/transformers
```

---

## Quick Start

### With a cloud provider

```typescript
import { createOpenAICompatibleProvider, topK } from 'embedding-utils';

// 1. Create a provider
const provider = createOpenAICompatibleProvider({
  apiKey: process.env.OPENAI_API_KEY!,
  model: 'text-embedding-3-small',
});

// 2. Embed your documents
const { embeddings } = await provider.embed([
  'The cat sat on the mat',
  'A dog played in the park',
  'Vectors are mathematical objects',
]);
// embeddings: Float32Array[] (v0.3+)

// 3. Search: find the closest matches to a query
const { embeddings: [query] } = await provider.embed('pets and animals');
const results = topK(query, embeddings, 2);
// => [{ index: 0, score: 0.87, ... }, { index: 1, score: 0.82, ... }]
```

### With local inference (no API key)

```typescript
import { createLocalProvider, topK } from 'embedding-utils';

const provider = createLocalProvider(); // uses Xenova/all-MiniLM-L12-v2 by default
const { embeddings } = await provider.embed(['hello world', 'goodbye world']);
```

---

## Providers

All providers implement the same `EmbeddingProvider` interface -- swap providers without changing application code:

```typescript
interface EmbeddingProvider {
  embed(input: string | string[], options?: EmbedOptions): Promise<EmbeddingResult>;
  readonly name: string;
  readonly dimensions: number | null;
}
```

### Provider comparison

| Provider | Factory | Auth | Batch Limit | Notes |
|----------|---------|------|:-----------:|-------|
| **Local (ONNX)** | `createLocalProvider()` | None | -- | Offline, auto-downloads models, built-in LRU cache |
| **OpenAI** | `createOpenAICompatibleProvider(config)` | API key | 2048 | Supports `dimensions` for output truncation |
| **Cohere** | `createCohereProvider(config)` | API key | 96 | Input type mapping (document/query) for best results |
| **Google Vertex** | `createGoogleVertexProvider(config)` | Access token | 5 | Token can be a string or async refresh function |
| **Voyage** | `createProvider('voyage', config)` | API key | 2048 | OpenAI-compatible wrapper |
| **Mistral** | `createProvider('mistral', config)` | API key | 2048 | OpenAI-compatible wrapper |
| **Jina** | `createProvider('jina', config)` | API key | 2048 | OpenAI-compatible wrapper |
| **OpenRouter** | `createProvider('openrouter', config)` | API key | 2048 | OpenAI-compatible wrapper |
| **Together** | `createProvider('together', config)` | API key | 2048 | OpenAI-compatible wrapper |
| **Fireworks** | `createProvider('fireworks', config)` | API key | 2048 | OpenAI-compatible wrapper |
| **Nomic** | `createProvider('nomic', config)` | API key | 2048 | OpenAI-compatible wrapper |
| **Mixedbread** | `createProvider('mixedbread', config)` | API key | 2048 | OpenAI-compatible wrapper |

Any OpenAI-compatible endpoint works via `createOpenAICompatibleProvider` with a custom `baseUrl` -- including Ollama, LM Studio, and Azure OpenAI.

### Provider examples

```typescript
import {
  createLocalProvider,
  createOpenAICompatibleProvider,
  createCohereProvider,
  createGoogleVertexProvider,
  createProvider,
} from 'embedding-utils';

// Local ONNX (requires @huggingface/transformers)
const local = createLocalProvider({ model: 'Xenova/all-MiniLM-L6-v2' });

// OpenAI
const openai = createOpenAICompatibleProvider({
  apiKey: 'sk-...',
  model: 'text-embedding-3-small',
  dimensions: 256, // optional output dimension truncation
});

// Cohere
const cohere = createCohereProvider({ apiKey: 'co-...' });
const result = await cohere.embed('hello', { inputType: 'query' }); // optimized for search

// Google Vertex AI (supports token refresh functions)
const vertex = createGoogleVertexProvider({
  projectId: 'my-project',
  accessToken: async () => getAccessToken(), // or a static 'ya29...' string
});

// Voyage, Mistral, Jina, OpenRouter via factory aliases
const voyage = createProvider('voyage', { apiKey: 'pa-...', model: 'voyage-3' });
const mistral = createProvider('mistral', { apiKey: 'sk-...', model: 'mistral-embed' });
const jina = createProvider('jina', { apiKey: 'jina-...', model: 'jina-embeddings-v3' });

// Together AI
const together = createProvider('together', {
  apiKey: 'tog-...',
  model: 'togethercomputer/m2-bert-80M-8k-retrieval',
});

// Fireworks AI
const fireworks = createProvider('fireworks', {
  apiKey: 'fw-...',
  model: 'nomic-ai/nomic-embed-text-v1.5',
});

// Nomic
const nomic = createProvider('nomic', {
  apiKey: 'nk-...',
  model: 'nomic-embed-text-v1.5',
});

// Mixedbread
const mixedbread = createProvider('mixedbread', {
  apiKey: 'mb-...',
  model: 'mixedbread-ai/mxbai-embed-large-v1',
});

// All providers return the same shape
const { embeddings, model, dimensions, usage } = await openai.embed(['hello', 'world']);
```

### Retry and batching

All cloud providers include automatic retry with exponential backoff (429 and 5xx errors) and auto-batching for large inputs. Configure via the `retry` option:

```typescript
const provider = createOpenAICompatibleProvider({
  apiKey: 'sk-...',
  model: 'text-embedding-3-small',
  retry: { maxRetries: 5, baseDelay: 500, maxDelay: 60000 },
});
```

---

## API Reference

### Vector Math

Core operations on embedding vectors. These are the low-level building blocks for comparing, combining, and transforming vectors. All functions validate inputs and throw `ValidationError` on empty vectors or dimension mismatches.

| Function | Description | Example |
|----------|-------------|---------|
| `cosineSimilarity(a, b)` | Cosine similarity (-1 to 1) | `cosineSimilarity([1, 0], [0, 1])` => `0` |
| `dotProduct(a, b)` | Dot product | `dotProduct([1, 2, 3], [4, 5, 6])` => `32` |
| `euclideanDistance(a, b)` | L2 distance | `euclideanDistance([0, 0], [3, 4])` => `5` |
| `manhattanDistance(a, b)` | L1 distance | `manhattanDistance([0, 0], [3, 4])` => `7` |
| `normalize(v)` | Unit vector | `normalize([3, 4])` => `Float32Array [0.6, 0.8]` |
| `magnitude(v)` | Vector length | `magnitude([3, 4])` => `5` |
| `add(a, b)` | Element-wise sum | `add([1, 2], [3, 4])` => `[4, 6]` |
| `subtract(a, b)` | Element-wise difference | `subtract([5, 3], [1, 2])` => `[4, 1]` |
| `scale(v, s)` | Scalar multiply | `scale([1, 2, 3], 2)` => `[2, 4, 6]` |
| `truncateDimensions(v, n)` | Matryoshka truncation + auto-normalize | `truncateDimensions([1, 2, 3, 4], 2)` => `Float32Array [0.45, 0.89]` |

`truncateDimensions` also accepts batches and preserves the input shape. In v0.3, it auto-normalizes after truncation so output vectors have L2 norm ≈ 1.0, ready for cosine similarity without manual normalization.

```typescript
// v0.3: truncateDimensions auto-normalizes for Matryoshka embeddings
const truncated = truncateDimensions(embedding1536, 256);
// truncated is Float32Array with L2 norm ≈ 1.0
// Ready for cosine similarity — no manual normalize() needed
```

#### When to use each

- **`cosineSimilarity`** -- The go-to for checking if two pieces of text are about the same thing. Use it in recommendation engines ("users who liked X also liked Y") or to verify if a generated answer is on-topic relative to the source document.
- **`dotProduct`** -- Some models (like OpenAI's `text-embedding-3-*` with `dimensions` truncation) are optimized for dot product ranking. Check your model's docs -- if it says "use dot product," use this instead of cosine.
- **`euclideanDistance`** -- Useful for anomaly detection: if a new data point's embedding is far from the cluster center, flag it as an outlier. Also commonly used in k-means style clustering.
- **`manhattanDistance`** -- Similar to euclidean but cheaper to compute and more robust to noisy dimensions. A good choice when embeddings are high-dimensional or you're running on constrained hardware.
- **`normalize`** -- Call this before storing embeddings in a database that only supports dot product search (e.g., some Postgres pgvector configs). After normalization, dot product equals cosine similarity.
- **`magnitude`** -- Sanity-check that a model is returning valid vectors. A magnitude of 0 or near-infinity means something went wrong. Also useful to verify whether embeddings are already normalized (magnitude ≈ 1.0).
- **`add` / `subtract`** -- Perform analogy-style vector arithmetic ("king − man + woman ≈ queen"). In practice, subtract the embedding of "negative sentiment" from a mixed-tone corpus to bias search toward positive content.
- **`scale`** -- Weight a particular embedding before combining it with others. For example, scale a title embedding by 2x before averaging with the body embedding, since titles carry more semantic signal.
- **`truncateDimensions`** -- OpenAI's Matryoshka-trained models let you drop from 1536 dims to 256 with minimal quality loss, saving 80%+ storage. In v0.3, truncation auto-normalizes the result so it's ready for cosine similarity without a separate `normalize()` call.

---

### Search

Find, rank, filter, and deduplicate embeddings by similarity. This is where most applications start -- "given a query, find the best matches." All search functions support four metrics via `{ metric: 'cosine' | 'dot' | 'euclidean' | 'manhattan' }` and optional `labels` for tracking source data.

```typescript
import { topK, topKMulti, aboveThreshold, deduplicate, rankBySimilarity, similarityMatrix } from 'embedding-utils';

const query = [1, 0, 0];
const corpus = [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]];
const labels = ['doc-a', 'doc-b', 'doc-c', 'doc-d'];
```

#### Top-K search

Use `topK` for **semantic search and RAG pipelines**. A user types "how do I reset my password" into your support chatbot -- embed their question, then `topK` against your knowledge base to find the most relevant help articles to feed to the LLM.

Use `topKMulti` for **batch search** -- e.g., 50 customer questions came in overnight and you want to find matching FAQ articles for all of them in one call instead of looping.

```typescript
const results = topK(query, corpus, 2, { labels });
// => [{ index: 0, score: 1.0, label: 'doc-a', embedding: [...] },
//     { index: 1, score: 0.94, label: 'doc-b', embedding: [...] }]

// Batch: search multiple queries at once
const batchResults = topKMulti([query, [0, 1, 0]], corpus, 2);
// => SearchResult[][] (one array per query)
```

#### Threshold filtering and deduplication

Use `aboveThreshold` for **intent matching** where you only want confident matches. A voice assistant matching speech to known commands should only act on matches above 0.85 -- anything below is "I don't understand." Unlike `topK`, this returns a variable number of results (could be zero).

Use `deduplicate` for **content deduplication**. You scraped 10,000 product listings and many are reposts with slightly different wording. Deduplicate at 0.95 similarity to collapse near-identical listings, keeping the first occurrence.

```typescript
// Find all embeddings above a similarity threshold
const matches = aboveThreshold(query, corpus, 0.8, { labels });
// => SearchResult[] sorted by descending score

// Remove near-duplicates (keeps first occurrence)
const unique = deduplicate(corpus, 0.95, { labels });
// => { embeddings: [...], indices: [0, 2, 3], labels: ['doc-a', 'doc-c', 'doc-d'] }
```

#### Ranking and similarity matrix

Use `rankBySimilarity` for **recommendation feeds**. A user just read an article -- rank your entire library by relevance to build a "more like this" feed. Unlike `topK`, this returns scores for *every* item.

Use `similarityMatrix` for **content audits and overlap visualization**. You have 100 support articles and want to find redundant documentation -- the NxN matrix powers a heatmap showing which articles cover the same ground.

```typescript
// Rank entire corpus by similarity
const ranked = rankBySimilarity(query, corpus);
// => SearchResult[] for all items, sorted descending

// Compute NxN pairwise similarity matrix
const matrix = similarityMatrix(corpus);
// => number[][] (symmetric, diagonal = 1.0 for cosine)
```

#### Filtered search

Pass a `filter` callback to `topK` to dynamically include/exclude items without modifying the corpus.

```typescript
// Only search documents from a specific category
const results = topK(query, corpus, 5, {
  labels: docLabels,
  filter: (index, label) => label?.startsWith('category-a') ?? false,
});
```

#### Pairwise similarity

Compare two parallel lists of embeddings element-wise. Useful for evaluating translation quality, paraphrase detection, or before/after comparison.

```typescript
import { pairwiseSimilarity } from 'embedding-utils';
const scores = pairwiseSimilarity(originalEmbeddings, translatedEmbeddings);
// => [0.95, 0.88, 0.92, ...] — one score per pair
```

#### MMR search (Maximal Marginal Relevance)

Select results that are both relevant to the query and diverse from each other. Prevents redundant results in RAG pipelines.

```typescript
import { mmrSearch } from 'embedding-utils';
const results = mmrSearch(query, corpus, 5, {
  lambda: 0.7,  // 0 = max diversity, 1 = max relevance
});
```

#### Re-ranking

Two-stage retrieval: fast initial search, then precise re-ranking with a different metric or weighted combination.

```typescript
import { rerankResults } from 'embedding-utils';
const initial = topK(query, corpus, 20);
const reranked = rerankResults(initial, query, {
  weights: { original: 0.3, rerank: 0.7 },
});
```

#### SearchIndex

Stateful in-memory search index with CRUD operations. Suitable for corpora up to ~100k embeddings.

```typescript
import { SearchIndex } from 'embedding-utils';

const index = new SearchIndex({ metric: 'cosine' });
index.add('doc-1', embedding1, { category: 'science' });
index.add('doc-2', embedding2, { category: 'history' });

const results = index.search(queryEmbedding, { topK: 5 });
index.remove('doc-1');
```

#### HNSW Search (Approximate Nearest Neighbor)

When your dataset grows beyond ~10k embeddings, brute-force search gets slow. HNSW (Hierarchical Navigable Small World) is a graph-based index that finds approximate nearest neighbors in milliseconds instead of seconds, with >95% accuracy.

**When to use HNSW vs SearchIndex:**

- `SearchIndex`: <10k embeddings, need 100% accuracy, simple setup
- `HNSWIndex`: 10k--1M+ embeddings, need fast queries, ok with ~95-99% accuracy

**Basic example** -- create, add, search:

```typescript
import { HNSWIndex } from 'embedding-utils';

// Create an index (defaults: M=16, efConstruction=200)
const index = new HNSWIndex({ metric: 'cosine' });

// Add embeddings with IDs and optional metadata
index.add('doc-1', embedding1, { title: 'Introduction to ML' });
index.add('doc-2', embedding2, { title: 'Deep Learning Basics' });
index.add('doc-3', embedding3, { title: 'Cooking with Pasta' });

// Search — returns top matches in milliseconds
const results = index.search(queryEmbedding, { topK: 2 });
// => [{ id: 'doc-1', score: 0.94, ... }, { id: 'doc-2', score: 0.89, ... }]
```

**Batch add:**

```typescript
// Add many items at once
index.addBatch([
  { id: 'doc-1', vector: emb1, metadata: { source: 'wiki' } },
  { id: 'doc-2', vector: emb2, metadata: { source: 'arxiv' } },
  // ... thousands more
]);
```

**Filtered search** -- filters run after graph traversal:

```typescript
// Only return results matching a filter
const results = index.search(query, {
  topK: 5,
  filter: (item) => item.metadata?.source === 'arxiv',
});
```

**Tuning accuracy vs speed** -- efSearch controls the trade-off:

```typescript
// Default efSearch=50 is a good balance
// Increase for higher accuracy (slower), decrease for speed (less accurate)
const precise = index.search(query, { topK: 10, efSearch: 200 }); // ~99% recall
const fast = index.search(query, { topK: 10, efSearch: 20 });    // ~90% recall, 3x faster
```

**Persistence** -- save to disk and reload:

```typescript
import { writeFileSync, readFileSync } from 'fs';

// Save index to disk
const bytes = index.serialize();
writeFileSync('my-index.hnsw', bytes);

// Load index later (even in a different process)
const loaded = HNSWIndex.deserialize(readFileSync('my-index.hnsw'));
const results = loaded.search(query, { topK: 5 }); // same results as original
```

**Configuration reference:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `M` | 16 | Connections per node. Higher = better recall, more memory |
| `efConstruction` | 200 | Build-time beam width. Higher = better graph quality, slower insert |
| `efSearch` | 50 | Query-time beam width. Higher = better recall, slower search |
| `metric` | `'cosine'` | Distance metric: `'cosine'`, `'dot'`, `'euclidean'`, `'manhattan'` |

#### Hybrid Search (RRF + Score Normalization)

Real-world search often combines multiple signals -- keyword search, semantic search, metadata filters. Reciprocal Rank Fusion (RRF) merges ranked lists from different sources into a single ranking without needing to calibrate scores across systems.

**RRF example** -- combine semantic and keyword search:

```typescript
import { fuseRankedLists } from 'embedding-utils';

// Results from semantic search (embedding similarity)
const semanticResults = [
  { id: 'doc-3', score: 0.95 },
  { id: 'doc-1', score: 0.87 },
  { id: 'doc-7', score: 0.82 },
];

// Results from keyword search (BM25 or similar)
const keywordResults = [
  { id: 'doc-1', score: 12.5 },
  { id: 'doc-3', score: 8.2 },
  { id: 'doc-5', score: 6.1 },
];

// Fuse them — scores don't need to be on the same scale!
const fused = fuseRankedLists([semanticResults, keywordResults]);
// doc-3 and doc-1 rank highest because they appear in BOTH lists

// Custom k parameter (default 60) — lower k gives more weight to top positions
const aggressive = fuseRankedLists([semanticResults, keywordResults], { k: 10 });
```

**Score normalization** -- when you need comparable scores:

```typescript
import { normalizeScores } from 'embedding-utils';

const rawScores = [0.2, 0.8, 0.5, 0.95, 0.1];

// Min-max: scale to [0, 1] range
normalizeScores(rawScores, 'min-max');

// Z-score: mean=0, std=1 (good for outlier detection)
normalizeScores(rawScores, 'z-score');

// Sigmoid: squash to (0, 1) (good for probability-like scores)
normalizeScores(rawScores, 'sigmoid');
```

**When to use which normalization:**

| Method | Output Range | Best For |
|--------|-------------|----------|
| `min-max` | [0, 1] | UI display, threshold comparison |
| `z-score` | (-inf, +inf) | Outlier detection, statistical analysis |
| `sigmoid` | (0, 1) | Probability-like confidence scores |

---

### Clustering

Group similar items together without predefined categories. Useful for discovering natural topic groupings in unlabeled data -- e.g., auto-tagging thousands of support tickets into topics like "billing issues," "shipping delays," and "app crashes" without defining those categories upfront.

Uses greedy agglomerative clustering: iterate embeddings, assign each to the most similar existing cluster (or create a new one), filter by size, merge if needed.

```typescript
import { clusterEmbeddings, getPreset, cohesionScore, silhouetteScore, assignToCluster, mergeClusters } from 'embedding-utils';
```

#### Basic clustering

```typescript
const clusters = clusterEmbeddings(embeddings, {
  similarityThreshold: 0.85,
  minClusterSize: 3,
  maxClusters: 10,
  metric: 'cosine',
});
// => [{ centroid, members, labels?, size, cohesion }]
```

#### Presets

Three built-in presets so you don't have to guess threshold values. Use `'high-precision'` for tight groups (e.g., near-duplicate detection), `'balanced'` for general topic discovery, and `'performance'` for fast broad groupings on large datasets.

| Preset | Threshold | Min Size | Max Clusters | Best for |
|--------|:---------:|:--------:|:------------:|----------|
| `high-precision` | 0.95 | 3 | 10 | Tight, highly cohesive groups |
| `balanced` | 0.85 | 5 | 5 | General-purpose clustering |
| `performance` | 0.75 | 10 | 3 | Fast, broad groupings |

```typescript
const clusters = clusterEmbeddings(embeddings, getPreset('balanced'));
```

#### Quality metrics

Use `cohesionScore` to check if an individual cluster is internally consistent -- a low score (e.g., 0.4) means the cluster is a grab-bag of unrelated items and you should tighten the threshold.

Use `silhouetteScore` to evaluate overall clustering quality. A score near 1.0 means clusters are well-separated; near 0 means items could belong to any cluster. Use this to tune your `similarityThreshold`.

```typescript
// Internal cluster quality (0-1, higher = tighter)
const cohesion = cohesionScore(clusters[0]);

// Global clustering quality (-1 to 1, higher = better separated)
const quality = silhouetteScore(clusters);
```

#### Operations

Use `assignToCluster` for **live classification** -- when a new support ticket or document arrives, classify it into an existing cluster without re-clustering everything. Returns `-1` if it doesn't fit any group (an outlier / new topic).

Use `mergeClusters` for **manual refinement** -- after auto-clustering, you notice "password reset" and "login issues" ended up as separate clusters but your team treats them as one. Merge them programmatically.

```typescript
// Classify a new embedding into existing clusters
const { clusterIndex, similarity } = assignToCluster(newEmbedding, clusters, { threshold: 0.8 });
// clusterIndex = -1 if below threshold (outlier)

// Merge two clusters
const merged = mergeClusters(clusters[0], clusters[1]);
```

#### Clustering enhancements

**`centroidCohesion`** -- Average similarity of all members to the cluster centroid. Faster than pairwise `cohesionScore` (O(n) vs O(n²)), good for per-cluster quality monitoring.

```typescript
import { centroidCohesion } from 'embedding-utils';
const quality = centroidCohesion(cluster, 'cosine'); // 0-1, higher = tighter
```

**`legacy` preset** -- Disables clustering by returning all embeddings in a single cluster. Use for backwards compatibility with pipelines that predated clustering.

| Preset | Threshold | Min Size | Max Clusters | Best for |
|--------|:---------:|:--------:|:------------:|----------|
| `high-precision` | 0.95 | 3 | 10 | Tight, highly cohesive groups |
| `balanced` | 0.85 | 5 | 5 | General-purpose clustering |
| `performance` | 0.75 | 10 | 3 | Fast, broad groupings |
| `legacy` | 0 | 1 | 1 | Clustering disabled (single cluster) |

**`assignmentStrategy`** -- Controls how embeddings are assigned during clustering. `'centroid'` (default) compares to the cluster centroid. `'average-similarity'` computes average similarity to all current members, which is more accurate but slower.

```typescript
clusterEmbeddings(embeddings, { assignmentStrategy: 'average-similarity' });
```

**`shuffle` + `shuffleSeed`** -- Greedy clustering is order-sensitive. Enable `shuffle: true` for order-independent results with a deterministic seed.

```typescript
clusterEmbeddings(embeddings, { shuffle: true, shuffleSeed: 42 });
```

**`findOptimalK` + `silhouetteByK`** -- Automatically determine the best number of clusters using silhouette analysis or the elbow method.

```typescript
import { findOptimalK, silhouetteByK } from 'embedding-utils';

const bestK = findOptimalK(embeddings, { minK: 2, maxK: 8 });
const scores = silhouetteByK(embeddings); // [{ k: 2, silhouette: 0.6 }, ...]
```

**`clusterStats` + `detectOutliers`** -- Detailed per-cluster statistics and outlier detection.

```typescript
import { clusterStats, detectOutliers } from 'embedding-utils';

const stats = clusterStats(cluster); // { minSimilarity, maxSimilarity, meanSimilarity, ... }
const outlierIndices = detectOutliers(cluster, { threshold: 2 }); // member indices > 2σ from mean
```

**`centroidDrift`** -- Measure how much a cluster centroid has shifted between two snapshots.

```typescript
import { centroidDrift } from 'embedding-utils';
const drift = centroidDrift(oldCentroid, newCentroid, 'cosine'); // 0 = no change
```

#### HDBSCAN Clustering

The existing `clusterEmbeddings` requires you to set a similarity threshold and max cluster count. HDBSCAN automatically discovers the right number of clusters AND identifies outliers -- no tuning needed. It handles clusters of different densities (e.g., a tight group of 5 similar docs alongside a loose group of 50).

**When to use HDBSCAN vs clusterEmbeddings:**

- `clusterEmbeddings`: You know roughly how many clusters to expect, want fast results, need deterministic output
- `hdbscan`: You don't know how many clusters exist, need automatic outlier detection, have varying-density data

**Basic example:**

```typescript
import { hdbscan } from 'embedding-utils';

const result = hdbscan(embeddings, { minClusterSize: 5 });

console.log(`Found ${result.clusters.length} clusters`);
console.log(`${result.noise.members.length} noise points (outliers)`);

// Each cluster has the familiar Cluster type
result.clusters.forEach((cluster, i) => {
  console.log(`  Cluster ${i}: ${cluster.size} items`);
});

// Labels array maps each input to its cluster (-1 = noise)
console.log(result.labels); // => [0, 0, 1, -1, 1, 0, -1, ...]
```

**With labels for tracking** -- trace back to source data:

```typescript
const docNames = ['intro.md', 'setup.md', 'api.md', 'faq.md', 'changelog.md'];

const result = hdbscan(embeddings, {
  minClusterSize: 2,
  labels: docNames,
});

// Labels are preserved in clusters and noise
result.clusters.forEach((cluster, i) => {
  console.log(`Cluster ${i}: ${cluster.labels?.join(', ')}`);
});
// => Cluster 0: intro.md, setup.md
// => Cluster 1: api.md, faq.md

console.log(`Outliers: ${result.noise.labels?.join(', ')}`);
// => Outliers: changelog.md
```

**Tuning** -- minClusterSize and metric:

```typescript
// Larger minClusterSize = fewer, bigger clusters (more noise points)
const broad = hdbscan(embeddings, { minClusterSize: 10 });

// Smaller minClusterSize = more, smaller clusters (fewer noise points)
const granular = hdbscan(embeddings, { minClusterSize: 3 });

// Different metric changes cluster shapes
const cosineResult = hdbscan(embeddings, { metric: 'cosine' });
const euclideanResult = hdbscan(embeddings, { metric: 'euclidean' });
```

---

### Aggregation

Combine multiple embeddings into a single representative vector. Common use case: a 20-page PDF gets split into 40 chunks, each with its own embedding. Aggregate them into one "document embedding" for coarse-grained search (find relevant documents first, then drill into chunks).

```typescript
import { averageEmbeddings, weightedAverage, incrementalAverage, centroid, maxPooling, minPooling, combineEmbeddings } from 'embedding-utils';
```

**`averageEmbeddings`** -- Simple element-wise mean. Use when all inputs carry equal weight, such as averaging chunk embeddings to represent an entire document.

```typescript
averageEmbeddings([[1, 0], [0, 1]]); // => [0.5, 0.5]
```

**`weightedAverage`** -- Give more importance to certain embeddings. Useful for hybrid representations: embed a product's title, description, and reviews separately, then weight the title 3x and description 2x to emphasize what the product *is* over what people *say*.

```typescript
weightedAverage([[1, 0], [0, 1]], [3, 1]); // => [0.75, 0.25]
```

**`incrementalAverage`** -- Streaming / running average without storing all prior embeddings. Use when processing a firehose of data (e.g., tweets, logs, sensor readings) and you need a running mean with only 1 vector in memory.

```typescript
let avg = [1, 2, 3];
avg = incrementalAverage(avg, [4, 5, 6], 1); // count = embeddings already averaged
avg = incrementalAverage(avg, [7, 8, 9], 2);
// Numerically equivalent to averageEmbeddings([[1,2,3], [4,5,6], [7,8,9]])
```

**`batchIncrementalAverage`** -- Incrementally compute the average of embedding batches without reprocessing all previous data. Use when new data arrives in batches and you want to update a running centroid without re-averaging from scratch.

```typescript
import { batchIncrementalAverage } from 'embedding-utils';

let centroidAvg = firstBatchAvg;  // average of first 10 embeddings
let count = 10;

// New batch of 5 embeddings arrives
centroidAvg = batchIncrementalAverage(centroidAvg, newBatch, count);
count += newBatch.length;
// Mathematically equivalent to averageEmbeddings([...allOldEmbeddings, ...newBatch])
```

Use `batchIncrementalAverage` over `averageEmbeddings` when you don't want to store or reprocess all historical embeddings. If you're updating one embedding at a time, use `incrementalAverage` instead.

**`maxPooling` / `minPooling`** -- Element-wise max or min across vectors. Max pooling captures the "strongest signal" across chunks -- if any chunk mentions "urgent," that signal is preserved. Min pooling captures the weakest, useful for detecting what's *missing* across a set.

```typescript
maxPooling([[1, 0, 3], [2, 5, 1]]); // => [2, 5, 3]
minPooling([[1, 0, 3], [2, 5, 1]]); // => [1, 0, 1]
```

**`combineEmbeddings`** -- Embed multiple texts and aggregate in one call. A convenient one-liner for creating a single embedding from multi-field entities (e.g., combine a user's bio, interests, and recent posts into one "user profile" vector).

```typescript
const combined = await combineEmbeddings(['hello', 'world'], provider);
const pooled = await combineEmbeddings(['hello', 'world'], provider, { aggregate: maxPooling });
```

---

### Evaluation Metrics

How do you know if your search is actually good? These metrics compare your search results against ground-truth labels to give you a number. Use them to benchmark different embedding models, tune search parameters, or set up regression tests that alert you when search quality degrades.

**Complete example** -- evaluate a search system:

```typescript
import { recallAtK, ndcg, mrr, meanAveragePrecision } from 'embedding-utils';

// Your search returned these document IDs (ranked by relevance)
const retrieved = ['doc-3', 'doc-1', 'doc-7', 'doc-4', 'doc-2'];

// Ground truth: these are the actually relevant documents
const relevant = ['doc-1', 'doc-2', 'doc-5'];

// Recall@K: what fraction of relevant docs did you find in top K?
recallAtK(retrieved, relevant, 3);  // => 0.333 (found 1 of 3 relevant in top 3)
recallAtK(retrieved, relevant, 5);  // => 0.667 (found 2 of 3 relevant in top 5)

// MRR: how high is the FIRST relevant result?
mrr(retrieved, relevant);  // => 0.5 (first relevant doc "doc-1" is at position 2 → 1/2)

// MAP: average precision across all relevant results
meanAveragePrecision(retrieved, relevant);  // => 0.417

// NDCG: normalized discounted cumulative gain (accounts for graded relevance)
const relevanceScores = { 'doc-1': 3, 'doc-2': 2, 'doc-5': 1 }; // higher = more relevant
ndcg(retrieved, relevanceScores, 5);  // => 0.63
```

**Practical use case** -- comparing two embedding models:

```typescript
// Compare OpenAI vs Cohere for your specific use case
const testQueries = [...]; // your evaluation set
const groundTruth = [...]; // manually labeled relevant docs per query

for (const model of [openaiProvider, cohereProvider]) {
  let totalRecall = 0;
  for (let i = 0; i < testQueries.length; i++) {
    const results = await searchWith(model, testQueries[i], { topK: 10 });
    totalRecall += recallAtK(results.map(r => r.id), groundTruth[i], 10);
  }
  console.log(`${model.name} avg recall@10: ${(totalRecall / testQueries.length).toFixed(3)}`);
}
```

**Metrics cheat sheet:**

| Metric | Question It Answers | Range | Use When |
|--------|-------------------|-------|----------|
| `recallAtK` | "Did I find all the relevant docs?" | 0--1 | RAG (need high recall to feed LLM) |
| `mrr` | "How quickly do I find the first relevant doc?" | 0--1 | Single-answer search (FAQ, QA) |
| `meanAveragePrecision` | "Are relevant docs ranked above irrelevant ones?" | 0--1 | General ranking quality |
| `ndcg` | "Are the MOST relevant docs ranked highest?" | 0--1 | Graded relevance (some docs more relevant than others) |

---

### Async Pipeline

Embedding thousands of documents? The async pipeline handles batching, concurrency, rate limiting, and progress tracking -- so you don't have to write retry loops and sleep timers. It works with any provider and can resume from where it left off if interrupted.

**Basic example** -- embed a large dataset:

```typescript
import { createEmbeddingPipeline, createProvider } from 'embedding-utils';

const provider = createProvider('openai', {
  apiKey: process.env.OPENAI_API_KEY!,
  model: 'text-embedding-3-small',
});

const pipeline = createEmbeddingPipeline(provider, {
  batchSize: 100,        // send 100 texts per API call
  concurrency: 3,        // up to 3 API calls in flight at once
  rateLimit: {
    requestsPerMinute: 500,
    tokensPerMinute: 1_000_000,
  },
});

const texts = loadDocuments(); // your 50,000 documents
const embeddings = await pipeline.embed(texts);
// => Float32Array[] — one embedding per input text
```

**Progress tracking:**

```typescript
const pipeline = createEmbeddingPipeline(provider, {
  batchSize: 100,
  onProgress: ({ completed, total, elapsed }) => {
    const pct = ((completed / total) * 100).toFixed(1);
    const rate = (completed / (elapsed / 1000)).toFixed(0);
    console.log(`${pct}% complete (${completed}/${total}) — ${rate} docs/sec`);
  },
});
```

**Checkpoint/resume** -- survive crashes on long jobs:

```typescript
import { writeFileSync, readFileSync, existsSync } from 'fs';

const pipeline = createEmbeddingPipeline(provider, {
  batchSize: 100,
  checkpoint: {
    interval: 5, // save progress every 5 batches
    save: async (state) => {
      writeFileSync('checkpoint.json', JSON.stringify({
        completedIds: [...state.completedIds],
        totalProcessed: state.totalProcessed,
        timestamp: state.timestamp,
      }));
    },
    load: async () => {
      if (!existsSync('checkpoint.json')) return null;
      const data = JSON.parse(readFileSync('checkpoint.json', 'utf-8'));
      return { ...data, completedIds: new Set(data.completedIds) };
    },
  },
});

// If the process crashes and restarts, it picks up where it left off
const embeddings = await pipeline.embed(texts);
```

---

### Quantization

Reduce embedding precision for storage and memory efficiency. Essential when scaling to millions of vectors -- 10M embeddings at 1536 dims takes ~57 GB as float32, but only ~14 GB as int8, fitting on a single machine with less than 1% accuracy loss.

| Type | Bits | Size vs float32 | Input Range | Precision Loss | Best for |
|------|:----:|:---------------:|:-----------:|:--------------:|----------|
| `fp16` | 16 | 50% | Any | Negligible | Default choice, wide value ranges |
| `int8` | 8 | 25% | [-1, 1] | ~0.8% | Normalized embeddings |
| `uint8` | 8 | 25% | [0, 1] | ~0.4% | Positive embeddings |
| `binary` | 1 | 3% | Any | High (sign only) | Extreme compression, candidate filtering |

**`quantize` / `dequantize`** -- Convert between full-precision and compressed representations. Use `int8` or `uint8` for general storage savings. Use `binary` for a two-stage search pipeline: quickly find the top 1,000 candidates with binary similarity, then re-rank those candidates with full-precision vectors. This is how production systems handle billion-scale data.

**`getQuantizationInfo`** -- Inspect a quantization type's range and precision before choosing. Helps you decide between `int8` (needs normalized [-1, 1] input) vs `uint8` (needs [0, 1] input).

```typescript
import { quantize, dequantize, getQuantizationInfo } from 'embedding-utils';

const embedding = [0.5, -0.3, 0.8, 0.1];

// Quantize to int8 (75% memory reduction)
const quantized = quantize(embedding, 'int8'); // => Int8Array [64, -38, 102, 13]
const restored = dequantize(quantized, 'int8'); // => [0.504, -0.299, 0.803, 0.102]

// Binary quantization (97% reduction, preserves sign only)
const binary = quantize(embedding, 'binary'); // => Uint8Array (packed bits)
const signs = dequantize(binary, 'binary');   // => [1, -1, 1, 1]

// Inspect any quantization type
getQuantizationInfo('int8');
// => { bits: 8, range: [-128, 127], description: 'Signed 8-bit integer...' }
```

#### Calibrated Quantization

Standard int8 quantization assumes embedding values fall in [-1, 1]. But many models produce values in a narrower range (e.g., [-0.3, 0.4]). Calibrated quantization learns the actual range from your data, using the full int8 resolution for better accuracy.

```typescript
import { calibrate, quantize, dequantize } from 'embedding-utils';

// Step 1: Learn the value distribution from a sample of your embeddings
const calibration = calibrate(sampleEmbeddings); // ~1000 embeddings is enough

// Step 2: Quantize with calibration (tighter mapping = less precision loss)
const quantized = quantize(embedding, 'int8', { calibration });
const restored = dequantize(quantized, 'int8', { calibration });

// Compare: calibrated quantization has ~40% less error than uncalibrated
// Uncalibrated: maps [-1, 1] → [0, 255] — wastes resolution on unused range
// Calibrated: maps [actual_min, actual_max] → [0, 255] — uses full resolution
```

#### Hamming Distance

After binary quantization (1-bit), you can compare embeddings with Hamming distance -- just count differing bits. This is ~100x faster than cosine similarity and is used for the first stage of billion-scale search: quickly narrow down candidates, then re-rank with full-precision vectors.

```typescript
import { quantize, hammingDistance, hammingSimilarity } from 'embedding-utils';

// Binary quantize: 1536 floats → 192 bytes (97% smaller)
const binaryA = quantize(embeddingA, 'binary');
const binaryB = quantize(embeddingB, 'binary');

// Fast bitwise comparison
const distance = hammingDistance(binaryA, binaryB); // number of differing bits
const similarity = hammingSimilarity(binaryA, binaryB, 1536); // 0-1 scale

// Two-stage search pattern:
// 1. Binary search (fast, approximate) to get top 1000 candidates
// 2. Full-precision re-rank on those 1000
```

---

### Dimensionality Reduction (Random Projection)

Need to reduce 1536-dim embeddings to 256 dims but your model doesn't support Matryoshka truncation? Random projection (Johnson-Lindenstrauss lemma) preserves pairwise distances while shrinking vectors. It's mathematically guaranteed to work -- no training needed.

**When to use `truncateDimensions` vs `createRandomProjection`:**

- `truncateDimensions`: Your model was trained with Matryoshka (e.g., OpenAI text-embedding-3-*) -- use this, it's free
- `createRandomProjection`: Your model wasn't trained with Matryoshka -- use this for a mathematically-sound reduction

**Basic example:**

```typescript
import { createRandomProjection } from 'embedding-utils';

// Create a projector: 1536 dims → 256 dims
const projector = createRandomProjection(1536, 256, { seed: 42 });

// Project a single vector
const reduced = projector.project(embedding); // Float32Array (256 dims)

// Project a batch (more efficient)
const allReduced = projector.projectBatch(embeddings); // Float32Array[] (256 dims each)

// Pairwise distances are approximately preserved (JL guarantee)
// cosineSimilarity(reduced_a, reduced_b) ≈ cosineSimilarity(original_a, original_b)
```

**Deterministic with seed:**

```typescript
// Same seed = same projection matrix = reproducible results
const p1 = createRandomProjection(1536, 256, { seed: 42 });
const p2 = createRandomProjection(1536, 256, { seed: 42 });
// p1.project(vec) and p2.project(vec) produce identical output
```

---

### Markdown-Aware Chunking

Standard text chunking splits blindly by token count. Markdown-aware chunking respects document structure -- it never splits a code block in half, keeps lists together, and tracks which heading each chunk belongs to. Essential for RAG on documentation, READMEs, and knowledge bases.

**Basic example:**

```typescript
import { chunkByStructure } from 'embedding-utils';

const markdown = `
# Getting Started

Install the package and set up your first project.

## Installation

\`\`\`bash
npm install my-package
\`\`\`

## Configuration

Create a config file with these settings:

| Key | Value | Required |
|-----|-------|----------|
| apiKey | string | yes |
| timeout | number | no |
`;

const chunks = chunkByStructure(markdown, { maxTokens: 200 });

chunks.forEach(chunk => {
  console.log(`[${chunk.metadata.type}] ${chunk.metadata.headings.join(' > ')}`);
  console.log(chunk.text.substring(0, 80) + '...');
  console.log();
});
// [heading] Getting Started
// # Getting Started...
//
// [paragraph] Getting Started
// Install the package and set up your first project....
//
// [code] Getting Started > Installation
// ```bash\nnpm install my-package\n```...
//
// [table] Getting Started > Configuration
// | Key | Value | Required |...
```

**Metadata tracking** -- heading breadcrumbs enable scoped search:

```typescript
// Each chunk knows its heading context — use it for metadata filtering
const chunks = chunkByStructure(docsMarkdown, { maxTokens: 512 });

for (const chunk of chunks) {
  await index.add(generateId(), chunkEmbedding, {
    headings: chunk.metadata.headings, // ['API Reference', 'Authentication']
    type: chunk.metadata.type,         // 'paragraph', 'code', 'table', etc.
    offset: chunk.metadata.offset,     // character position in original doc
  });
}

// Later: search only within a specific section
const results = index.search(query, {
  topK: 5,
  filter: (item) => item.metadata?.headings?.includes('Authentication'),
});
```

---

### Storage

Serialize embeddings for persistence and cache frequently used results. Use these to avoid re-calling expensive embedding APIs -- pre-compute once, store, and reload on startup.

#### Serialization

Three formats with different trade-offs:

| Format | Output Type | Relative Size | Use Case |
|--------|:-----------:|:-------------:|----------|
| `json` | `string` | Largest | Human-readable, debugging, APIs |
| `binary` | `Uint8Array` | Smallest | File storage, binary protocols |
| `base64` | `string` | ~33% > binary | Text-safe transmission (HTTP, databases) |

Use `binary` for writing embeddings to disk files -- smallest size, fastest I/O. Use `base64` when you need to store in a Postgres `TEXT` column or send over a JSON API. Use `json` for debugging and inspection where you want to see the actual numbers.

```typescript
import { serialize, deserialize } from 'embedding-utils';

const embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];

const json = serialize(embeddings, 'json');     // JSON string
const binary = serialize(embeddings, 'binary'); // Uint8Array
const base64 = serialize(embeddings, 'base64'); // base64 string

const restored = deserialize(json, 'json');     // number[][] (identical to original)
```

#### LRU Cache

In-memory cache with configurable size limits and TTL. Use this to avoid redundant API calls -- users in your app often search similar queries, and caching means "how to reset password" doesn't hit the OpenAI API again for 60 seconds.

Implements the `CacheProvider` interface for pluggability. The async interface means you can later swap in Redis or SQLite without changing application code.

```typescript
import { createLRUCache } from 'embedding-utils';

const cache = createLRUCache({ maxSize: 1000, ttl: 60000 }); // 60s TTL

await cache.set('doc-1', [[0.1, 0.2, 0.3]]);
const cached = await cache.get('doc-1'); // => [[0.1, 0.2, 0.3]]
await cache.has('doc-1');                // => true
await cache.delete('doc-1');
await cache.clear();
```

---

### Model Management

Manage locally-downloaded ONNX models for the local provider. Only relevant if you're using `createLocalProvider()` for offline / API-key-free embedding.

- **`downloadModel`** -- Pre-download a model so it's ready at runtime. Include this in your Docker build or CI setup to bake the model into the image and avoid a cold-start download on first request.
- **`listModels`** -- See which models are cached locally. Useful for admin dashboards showing disk usage.
- **`deleteModel`** -- Remove old model versions after upgrading to a newer one, or clean up models you no longer use.
- **`setModelPath`** -- Point the model cache to a custom directory. In a shared server environment, use a shared NFS mount so multiple workers don't each download their own copy.
- **`getModelInfo`** -- Check a model's dimensions (384 vs 768) and size (22M vs 109M) before downloading, to pick the right trade-off between inference speed and embedding quality.

```typescript
import { downloadModel, listModels, deleteModel, setModelPath, getModelInfo, MODEL_REGISTRY } from 'embedding-utils';

// Browse built-in model registry
getModelInfo('Xenova/all-MiniLM-L12-v2');
// => { id, dimensions: 384, size: '33M', description: '...' }

// Download, list, delete models
const path = await downloadModel('Xenova/all-MiniLM-L6-v2');
const models = await listModels();
await deleteModel('Xenova/all-MiniLM-L6-v2');

// Override the default cache directory
setModelPath('/path/to/models');
```

#### Built-in model registry

| Model | Dimensions | Size | Description |
|-------|:----------:|:----:|-------------|
| `Xenova/all-MiniLM-L12-v2` | 384 | 33M | All-round English embedding model |
| `Xenova/all-MiniLM-L6-v2` | 384 | 22M | Lightweight English embedding model |
| `Xenova/bge-small-en-v1.5` | 384 | 33M | BGE small English (asymmetric prefixes) |
| `Xenova/bge-base-en-v1.5` | 768 | 109M | BGE base English (asymmetric prefixes) |

---

### High-Level APIs

#### EmbeddingStore

A batteries-included store that combines a provider, optional cache, and search index. Add texts, search by text or embedding -- the store handles embedding generation and caching automatically.

```typescript
import { createEmbeddingStore, createOpenAICompatibleProvider } from 'embedding-utils';

const store = createEmbeddingStore({
  provider: createOpenAICompatibleProvider({ apiKey: 'sk-...', model: 'text-embedding-3-small' }),
  cache: { maxSize: 500 },
  metric: 'cosine',
});

await store.add('doc-1', 'The cat sat on the mat');
await store.add('doc-2', 'A dog played in the park');

const results = await store.search('animals', { topK: 2 });
// => [{ id: 'doc-1', score: 0.87, ... }, { id: 'doc-2', score: 0.82, ... }]
```

#### IncrementalClusterer

Online clustering that assigns embeddings as they arrive. No need to re-cluster the entire dataset when new data comes in.

```typescript
import { IncrementalClusterer } from 'embedding-utils';

const clusterer = new IncrementalClusterer({ similarityThreshold: 0.85 });
clusterer.addEmbedding([1, 0, 0], 'doc-a');
clusterer.addEmbedding([0.98, 0.02, 0], 'doc-b');
clusterer.addEmbedding([0, 1, 0], 'doc-c');

const clusters = clusterer.getClusters();
clusterer.rebalance(); // re-optimize clusters with full re-clustering
```

#### Text Chunking

Split text into chunks for embedding. Useful for processing long documents that exceed model token limits.

```typescript
import { chunkByTokenCount, chunkBySentence } from 'embedding-utils';

// Split by approximate token count (~1.3 tokens/word)
const chunks = chunkByTokenCount(longText, 512, { overlap: 50 });

// Split on sentence boundaries (uses Intl.Segmenter)
const sentences = chunkBySentence(longText, { maxTokens: 512 });
```

#### Getting Started (end-to-end)

```typescript
import { createEmbeddingStore, createOpenAICompatibleProvider } from 'embedding-utils';

const store = createEmbeddingStore({
  provider: createOpenAICompatibleProvider({ apiKey: 'sk-...', model: 'text-embedding-3-small' }),
});

// Add documents
await store.addBatch([
  { id: '1', text: 'Machine learning basics' },
  { id: '2', text: 'Cooking pasta recipes' },
  { id: '3', text: 'Neural network architectures' },
]);

// Search
const results = await store.search('AI and deep learning', { topK: 2 });
console.log(results.map(r => r.id)); // => ['3', '1']
```

---

## Migration Guide (v0.2 → v0.3)

### Breaking Change Summary

All vector-returning functions now return `Float32Array` instead of `number[]`. This reduces memory usage by ~50% and improves computation speed, but requires awareness of a few behavioral differences.

### What Changed

| Operation | v0.2 | v0.3 |
|-----------|------|------|
| `provider.embed(texts)` | `{ embeddings: number[][] }` | `{ embeddings: Float32Array[] }` |
| `topK(query, corpus, k)` | `results[0].embedding: number[]` | `results[0].embedding: Float32Array` |
| `normalize(v)` | `number[]` | `Float32Array` |
| `clusterEmbeddings(...)` | `cluster.centroid: number[]` | `cluster.centroid: Float32Array` |
| `deserialize(data, fmt)` | `number[][]` | `Float32Array[]` |

### Common Migration Patterns

```typescript
// Converting Float32Array back to number[] (if needed for legacy code)
const arr: number[] = [...float32Array];
// or
const arr: number[] = Array.from(float32Array);

// JSON serialization (Float32Array serializes differently!)
// v0.2: JSON.stringify([0.1, 0.2]) => '[0.1,0.2]'
// v0.3: JSON.stringify(new Float32Array([0.1, 0.2])) => '{"0":0.1,"1":0.2}'
// Fix: convert first
JSON.stringify([...embedding]); // => '[0.1,0.2]'

// Array.isArray check
// v0.2: Array.isArray(embedding) => true
// v0.3: Array.isArray(embedding) => false
// Fix: use the new Vector type guard
import { isVector } from 'embedding-utils';
isVector(embedding); // => true for both number[] and Float32Array

// Spread into arrays
// v0.2: const combined = [...embA, ...embB]  => number[]
// v0.3: const combined = [...embA, ...embB]  => number[] (spread converts Float32Array to number[])
// This still works! Spread automatically converts.

// .map() returns Float32Array, not Array
// v0.2: embedding.map(x => x * 2) => number[]
// v0.3: embedding.map(x => x * 2) => Float32Array  (because Float32Array.map returns Float32Array)
// Usually fine, but if you need Array methods like .includes(), convert first:
const arr = [...embedding].filter(x => x > 0);
```

### Input is NOT Breaking

All functions now accept BOTH `number[]` and `Float32Array` as input. Your existing `number[]` inputs still work without any changes:

```typescript
// All functions accept BOTH number[] and Float32Array as input
cosineSimilarity([1, 2, 3], [4, 5, 6]);              // still works
cosineSimilarity(new Float32Array([1, 2, 3]), [4, 5, 6]); // also works
```

---

## Caveats & Best Practices

- **Embedding dimension consistency** -- Always use the same model and dimensions for embeddings you plan to compare. Mixing embeddings from different models produces meaningless similarity scores.

- **Normalization** -- Cosine similarity assumes unit vectors for best results. Use `normalize()` if your provider doesn't auto-normalize, especially before storing in databases that only support dot product search.

- **Cache key considerations** -- Cache keys include model and dimensions. Switching models invalidates cache entries. Use a consistent provider config within a cache lifetime.

- **Clustering order sensitivity** -- Greedy agglomerative clustering is order-dependent. Use `shuffle: true` with a `shuffleSeed` for reproducible, order-independent results.

- **Text chunking accuracy** -- Token count is estimated at ~1.3 tokens/word. This is a heuristic, not a tokenizer -- for precise token budgets, use your model's actual tokenizer.

- **SearchIndex scalability** -- `SearchIndex` uses brute-force linear scan, suitable for up to ~100k embeddings. For larger corpora, use a dedicated vector database.

---

## API Reference (all exports)

### Math

| Function | Description |
|----------|-------------|
| `cosineSimilarity(a, b)` | Cosine similarity (-1 to 1) |
| `cosineDistance(a, b)` | 1 - cosineSimilarity |
| `dotProduct(a, b)` | Dot product |
| `euclideanDistance(a, b)` | L2 distance |
| `manhattanDistance(a, b)` | L1 distance |
| `normalize(v)` | Unit vector |
| `magnitude(v)` | Vector length |
| `add(a, b)` | Element-wise sum |
| `subtract(a, b)` | Element-wise difference |
| `scale(v, s)` | Scalar multiply |
| `isNormalized(v)` | Check if magnitude ≈ 1 |
| `truncateDimensions(v, n)` | Matryoshka truncation + auto-normalize |
| `validateDimensions(embeddings)` | Verify matching dimensions |
| `createRandomProjection(source, target, opts?)` | Johnson-Lindenstrauss dimensionality reduction |

### Search

| Function / Class | Description |
|------------------|-------------|
| `topK(query, corpus, k, opts?)` | Top-K similarity search |
| `topKMulti(queries, corpus, k, opts?)` | Batch top-K search |
| `aboveThreshold(query, corpus, t, opts?)` | Threshold-filtered search |
| `deduplicate(corpus, threshold, opts?)` | Remove near-duplicates |
| `rankBySimilarity(query, corpus, opts?)` | Rank entire corpus |
| `similarityMatrix(corpus, opts?)` | NxN pairwise matrix |
| `pairwiseSimilarity(a, b, metric?)` | Element-wise pair comparison |
| `rerankResults(results, query, opts?)` | Re-score search results |
| `mmrSearch(query, corpus, k, opts?)` | Diverse search (MMR) |
| `SearchIndex` | Stateful CRUD + search index |
| `HNSWIndex` | Approximate nearest neighbor index (HNSW graph) |
| `fuseRankedLists(lists, opts?)` | Reciprocal Rank Fusion |
| `normalizeScores(scores, method)` | Score normalization (min-max, z-score, sigmoid) |

### Clustering

| Function / Class | Description |
|------------------|-------------|
| `clusterEmbeddings(embeddings, config?)` | Greedy agglomerative clustering |
| `getPreset(name)` | Get preset config |
| `cohesionScore(cluster)` | Pairwise cohesion (O(n²)) |
| `centroidCohesion(cluster, metric?)` | Centroid-based cohesion (O(n)) |
| `silhouetteScore(clusters)` | Global clustering quality |
| `assignToCluster(embedding, clusters, opts?)` | Classify into existing cluster |
| `mergeClusters(a, b)` | Merge two clusters |
| `findOptimalK(embeddings, opts?)` | Find optimal cluster count |
| `silhouetteByK(embeddings, opts?)` | Silhouette scores per K |
| `clusterStats(cluster, metric?)` | Detailed cluster statistics |
| `detectOutliers(cluster, opts?)` | Find outlier members |
| `centroidDrift(old, new, metric?)` | Measure centroid shift |
| `IncrementalClusterer` | Online incremental clustering |
| `hdbscan(embeddings, opts?)` | Density-based clustering with auto cluster count |

### Aggregation

| Function | Description |
|----------|-------------|
| `averageEmbeddings(embeddings)` | Element-wise mean |
| `weightedAverage(embeddings, weights)` | Weighted mean |
| `incrementalAverage(avg, new, count)` | Streaming average |
| `batchIncrementalAverage(avg, batch, count)` | Batch streaming average |
| `centroid(embeddings)` | Alias for averageEmbeddings |
| `maxPooling(embeddings)` | Element-wise max |
| `minPooling(embeddings)` | Element-wise min |
| `combineEmbeddings(texts, provider, opts?)` | Embed + aggregate |

### Eval

| Function | Description |
|----------|-------------|
| `recallAtK(retrieved, relevant, k?)` | Recall at K |
| `ndcg(retrieved, relevance, k?)` | Normalized Discounted Cumulative Gain |
| `mrr(retrieved, relevant)` | Mean Reciprocal Rank |
| `meanAveragePrecision(retrieved, relevant)` | Mean Average Precision |

### Pipeline

| Function | Description |
|----------|-------------|
| `createEmbeddingPipeline(provider, opts?)` | Async embedding pipeline with batching and rate limiting |

### Providers

| Function | Description |
|----------|-------------|
| `createLocalProvider(config?)` | Local ONNX provider |
| `createOpenAICompatibleProvider(config)` | OpenAI / compatible APIs |
| `createCohereProvider(config)` | Cohere API |
| `createGoogleVertexProvider(config)` | Google Vertex AI |
| `createProvider(type, config)` | Factory for named providers |
| `withCache(provider, opts?)` | Add caching middleware |
| `retryWithBackoff(fn, config?)` | Retry utility |
| `autoBatch(inputs, batchSize, fn)` | Auto-batching utility |

### Storage

| Function | Description |
|----------|-------------|
| `serialize(embeddings, format)` | Serialize to JSON/binary/base64 |
| `deserialize(data, format)` | Deserialize embeddings |
| `createLRUCache(opts?)` | In-memory LRU cache |
| `warmCache(cache, entries)` | Pre-populate cache |
| `estimateMemorySavings(embeddings, type)` | Quantization savings estimate |

### Quantization

| Function | Description |
|----------|-------------|
| `quantize(embedding, type)` | Reduce precision |
| `dequantize(data, type)` | Restore precision |
| `getQuantizationInfo(type)` | Inspect type details |
| `calibrate(embeddings)` | Learn per-dimension value ranges for quantization |
| `hammingDistance(a, b)` | Hamming distance for binary vectors |
| `hammingSimilarity(a, b, dims)` | Hamming similarity (0-1 scale) |

### Models

| Function | Description |
|----------|-------------|
| `downloadModel(id)` | Download ONNX model |
| `listModels()` | List cached models |
| `deleteModel(id)` | Remove cached model |
| `setModelPath(path)` | Set cache directory |
| `getModelInfo(id)` | Get model metadata |
| `registerModel(info)` | Add custom model to registry |
| `getRecommendedModel(criteria?)` | Get model recommendation |

### Text

| Function | Description |
|----------|-------------|
| `chunkByTokenCount(text, maxTokens, opts?)` | Split by token count |
| `chunkBySentence(text, opts?)` | Split on sentence boundaries |
| `chunkByStructure(text, opts?)` | Markdown-aware chunking with heading metadata |
| `getTokenizerInfo(model)` | Get model tokenizer info |

### Store

| Function / Type | Description |
|-----------------|-------------|
| `createEmbeddingStore(config)` | Create high-level store |
| `EmbeddingStore` | Store interface type |

---

## Common Patterns

### Semantic search pipeline

```typescript
import { createOpenAICompatibleProvider, topK } from 'embedding-utils';

const provider = createOpenAICompatibleProvider({
  apiKey: process.env.OPENAI_API_KEY!,
  model: 'text-embedding-3-small',
});

// Index your documents once
const documents = ['Climate change overview', 'Quantum computing basics', 'History of jazz'];
const { embeddings: corpus } = await provider.embed(documents);

// Search at query time
const { embeddings: [query] } = await provider.embed('environmental science');
const results = topK(query, corpus, 2, { labels: documents });
console.log(results[0].label); // => 'Climate change overview'
```

### Duplicate detection

```typescript
import { deduplicate } from 'embedding-utils';

const { embeddings: unique, indices } = deduplicate(allEmbeddings, 0.95);
console.log(`Removed ${allEmbeddings.length - unique.length} duplicates`);
```

### Document clustering with quality check

```typescript
import { clusterEmbeddings, getPreset, silhouetteScore } from 'embedding-utils';

const clusters = clusterEmbeddings(embeddings, getPreset('balanced'));
const quality = silhouetteScore(clusters);
console.log(`${clusters.length} clusters, silhouette score: ${quality.toFixed(2)}`);

clusters.forEach((c, i) => console.log(`  Cluster ${i}: ${c.size} items, cohesion ${c.cohesion.toFixed(2)}`));
```

### Memory-efficient storage

```typescript
import { quantize, dequantize, serialize, deserialize } from 'embedding-utils';

// Quantize before storing (75% memory reduction)
const quantized = embeddings.map(e => quantize(e, 'int8'));

// Or serialize for disk/network
const binary = serialize(embeddings, 'binary');
// Restore later
const restored = deserialize(binary, 'binary');
```

### Streaming average (no memory accumulation)

```typescript
import { incrementalAverage } from 'embedding-utils';

let avg = firstEmbedding;
let count = 1;

for await (const embedding of embeddingStream) {
  avg = incrementalAverage(avg, embedding, count);
  count++;
}
// avg is the running mean -- only 1 vector in memory at a time
```

### Topic analysis pipeline (fast-topic-analysis pattern)

Build a complete topic detection system: embed training phrases, cluster them, then compare incoming text against cluster centroids. This is the pattern used by [fast-topic-analysis](https://github.com/jparkerweb/fast-topic-analysis).

```typescript
import {
  createLocalProvider,
  clusterEmbeddings,
  getPreset,
  cosineSimilarity,
  centroidCohesion,
  assignToCluster,
  batchIncrementalAverage,
} from 'embedding-utils';

const provider = createLocalProvider({
  model: 'Xenova/all-MiniLM-L12-v2',
  documentPrefix: '',          // set if your model needs it
  queryPrefix: '',
});

// ── 1. Generate training embeddings per topic ──
const topics = [
  { name: 'cookies', phrases: ['chocolate chip cookies', 'baking cookies at home', ...], threshold: 0.4 },
  { name: 'space',   phrases: ['NASA launches rocket', 'Mars exploration plans', ...],   threshold: 0.4 },
];

for (const topic of topics) {
  const { embeddings } = await provider.embed(topic.phrases, { inputType: 'document' });

  // ── 2. Cluster the embeddings ──
  const clusters = clusterEmbeddings(embeddings, getPreset('balanced'));

  // ── 3. Inspect cluster quality ──
  for (const cluster of clusters) {
    const quality = centroidCohesion(cluster);
    console.log(`${topic.name}: ${cluster.size} phrases, cohesion ${quality.toFixed(3)}`);
  }

  // Store clusters (centroids + metadata) for runtime matching
  topic.clusters = clusters;
}

// ── 4. Runtime: classify incoming text ──
const { embeddings: [queryEmbed] } = await provider.embed('I love baking cookies', {
  inputType: 'query',
});

for (const topic of topics) {
  for (const cluster of topic.clusters) {
    const similarity = cosineSimilarity(queryEmbed, cluster.centroid);
    if (similarity >= topic.threshold) {
      console.log(`Match: "${topic.name}" (similarity: ${similarity.toFixed(3)})`);
    }
  }
}
```

### Incremental centroid updates

When new training data arrives, update existing cluster centroids without re-processing all historical data. This is mathematically equivalent to re-computing from scratch.

```typescript
import { batchIncrementalAverage, assignToCluster } from 'embedding-utils';

// New phrases arrive for the "cookies" topic
const { embeddings: newEmbeddings } = await provider.embed(newPhrases, {
  inputType: 'document',
});

for (const newEmbed of newEmbeddings) {
  // Find the nearest cluster
  const { clusterIndex, similarity } = assignToCluster(newEmbed, clusters);

  if (clusterIndex >= 0) {
    // Update the cluster centroid with weighted average
    clusters[clusterIndex].centroid = batchIncrementalAverage(
      clusters[clusterIndex].centroid,
      [newEmbed],
      clusters[clusterIndex].size,
    );
    clusters[clusterIndex].size++;
    clusters[clusterIndex].members.push(newEmbed);
  }
}
```

### Migration from fast-topic-analysis

If you're migrating from fast-topic-analysis's built-in functions to embedding-utils, here's how each function maps:

| fast-topic-analysis | embedding-utils | Notes |
|---------------------|-----------------|-------|
| `generateEmbeddings(phrases)` | `provider.embed(phrases)` | Use `createLocalProvider()` with same model |
| `cosineSimilarity(a, b)` | `cosineSimilarity(a, b)` | Identical signature and behavior |
| `calculateAverageEmbedding(embeddings)` | `averageEmbeddings(embeddings)` | Same formula |
| `weightedAverage(existing, count, newEmbeds)` | `batchIncrementalAverage(existing, newEmbeds, count)` | Note: `count` moves to 3rd param. Do **not** confuse with embedding-utils' own `weightedAverage(embeddings, weights)` which has a different signature. |
| `combineTopicEmbeddings(existing, count, phrases)` | `batchIncrementalAverage(existing, newEmbeds, count)` | Generate embeddings separately first, then pass to `batchIncrementalAverage` |
| `clusterEmbeddings(embeddings)` | `clusterEmbeddings(embeddings, config)` | Same algorithm, typed config. FTA's `phrasesWithEmbeddings` param is not needed — track labels separately. |
| `clusteringConfig` presets | `getPreset('balanced')` | `'high-precision'`, `'balanced'`, `'performance'`, `'legacy'`. Note: threshold values differ slightly — FTA balanced=0.9, EU balanced=0.85. Adjust if you need exact FTA behavior. |
| Cohesion calculation (inline in generate.js) | `centroidCohesion(cluster)` | Both compute avg similarity of members to centroid |
| Disabled clustering (`enabled: false`) | `getPreset('legacy')` | Returns single cluster with all embeddings |
| `prefixConfig.dataPrefix` | `LocalProviderConfig.documentPrefix` | Set in provider config |
| `prefixConfig.queryPrefix` | `LocalProviderConfig.queryPrefix` | Set in provider config |

---

## TypeScript

Full strict-mode type definitions ship with the package -- no `@types` install needed.

```typescript
import type {
  Vector,
  EmbeddingProvider,
  EmbeddingResult,
  EmbedOptions,
  SearchResult,
  Cluster,
  ClusteringConfig,
  SimilarityMetric,
  ProviderType,
  CacheProvider,
  CacheOptions,
  RetryConfig,
  ModelInfo,
  LocalProviderConfig,
  OpenAICompatibleConfig,
  CohereConfig,
  GoogleVertexConfig,
  HNSWOptions,
  HNSWSearchOptions,
  HDBSCANOptions,
  HDBSCANResult,
  PipelineOptions,
  CheckpointState,
  QuantizationCalibration,
  RandomProjector,
  StructuredChunk,
  RankedItem,
  NormalizationMethod,
} from 'embedding-utils';
```

Types are fully inferred -- you rarely need explicit annotations:

```typescript
const results = topK(query, corpus, 5);
//    ^? SearchResult[] -- fully typed

results[0].score;     // number
results[0].embedding; // Float32Array
results[0].label;     // string | undefined
```

Config types catch mistakes at compile time:

```typescript
topK(query, corpus, 5, { metric: 'cosine' });   // OK
topK(query, corpus, 5, { metric: 'invalid' });   // Type error
```

---

## Error Handling

Four error classes cover all failure modes, all extending `EmbeddingError`:

| Error | Thrown When | Key Fields |
|-------|------------|------------|
| `ValidationError` | Invalid inputs (empty vectors, dimension mismatch, bad config) | `message` |
| `ProviderError` | Cloud API failures (auth, rate limits, server errors) | `provider`, `status?` |
| `ModelNotFoundError` | Local model not found or `@huggingface/transformers` not installed | `message` |
| `EmbeddingError` | Base class for all above | `message` |

```typescript
import { cosineSimilarity, ValidationError, ProviderError } from 'embedding-utils';

try {
  cosineSimilarity([1, 2], [1, 2, 3]); // dimension mismatch
} catch (error) {
  if (error instanceof ValidationError) {
    console.error(error.message); // => 'Dimension mismatch: 2 vs 3'
  }
}

try {
  await provider.embed('hello');
} catch (error) {
  if (error instanceof ProviderError) {
    console.error(`${error.provider} failed (HTTP ${error.status}): ${error.message}`);
  }
}
```

---

## License

[Apache-2.0](./LICENSE)

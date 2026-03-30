# embedding-utils

[![npm version](https://img.shields.io/npm/v/embedding-utils)](https://www.npmjs.com/package/embedding-utils)
[![license](https://img.shields.io/npm/l/embedding-utils)](./LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict-3178C6)](https://www.typescriptlang.org/)
[![zero dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)](#)
[![node](https://img.shields.io/node/v/embedding-utils)](https://nodejs.org/)

**Vector math, similarity search, clustering, and multi-provider embedding generation -- zero dependencies, full TypeScript, one import.**

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
  - [Clustering](#clustering)
  - [Aggregation](#aggregation)
  - [Quantization](#quantization)
  - [Storage](#storage)
  - [Model Management](#model-management)
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

Core operations on embedding vectors. All functions validate inputs and throw `ValidationError` on empty vectors or dimension mismatches.

| Function | Description | Example |
|----------|-------------|---------|
| `cosineSimilarity(a, b)` | Cosine similarity (-1 to 1) | `cosineSimilarity([1, 0], [0, 1])` => `0` |
| `dotProduct(a, b)` | Dot product | `dotProduct([1, 2, 3], [4, 5, 6])` => `32` |
| `euclideanDistance(a, b)` | L2 distance | `euclideanDistance([0, 0], [3, 4])` => `5` |
| `manhattanDistance(a, b)` | L1 distance | `manhattanDistance([0, 0], [3, 4])` => `7` |
| `normalize(v)` | Unit vector | `normalize([3, 4])` => `[0.6, 0.8]` |
| `magnitude(v)` | Vector length | `magnitude([3, 4])` => `5` |
| `add(a, b)` | Element-wise sum | `add([1, 2], [3, 4])` => `[4, 6]` |
| `subtract(a, b)` | Element-wise difference | `subtract([5, 3], [1, 2])` => `[4, 1]` |
| `scale(v, s)` | Scalar multiply | `scale([1, 2, 3], 2)` => `[2, 4, 6]` |
| `truncateDimensions(v, n)` | Matryoshka truncation | `truncateDimensions([1, 2, 3, 4], 2)` => `[1, 2]` |

`truncateDimensions` also accepts batches (`number[][]`) and preserves the input shape.

---

### Search

Find, rank, filter, and deduplicate embeddings by similarity. All search functions support four metrics via `{ metric: 'cosine' | 'dot' | 'euclidean' | 'manhattan' }` and optional `labels` for tracking source data.

```typescript
import { topK, topKMulti, aboveThreshold, deduplicate, rankBySimilarity, similarityMatrix } from 'embedding-utils';

const query = [1, 0, 0];
const corpus = [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]];
const labels = ['doc-a', 'doc-b', 'doc-c', 'doc-d'];
```

#### Top-K search

```typescript
const results = topK(query, corpus, 2, { labels });
// => [{ index: 0, score: 1.0, label: 'doc-a', embedding: [...] },
//     { index: 1, score: 0.94, label: 'doc-b', embedding: [...] }]

// Batch: search multiple queries at once
const batchResults = topKMulti([query, [0, 1, 0]], corpus, 2);
// => SearchResult[][] (one array per query)
```

#### Threshold filtering and deduplication

```typescript
// Find all embeddings above a similarity threshold
const matches = aboveThreshold(query, corpus, 0.8, { labels });
// => SearchResult[] sorted by descending score

// Remove near-duplicates (keeps first occurrence)
const unique = deduplicate(corpus, 0.95, { labels });
// => { embeddings: [...], indices: [0, 2, 3], labels: ['doc-a', 'doc-c', 'doc-d'] }
```

#### Ranking and similarity matrix

```typescript
// Rank entire corpus by similarity
const ranked = rankBySimilarity(query, corpus);
// => SearchResult[] for all items, sorted descending

// Compute NxN pairwise similarity matrix
const matrix = similarityMatrix(corpus);
// => number[][] (symmetric, diagonal = 1.0 for cosine)
```

---

### Clustering

Group embeddings using greedy agglomerative clustering. Iterate embeddings, assign each to the most similar existing cluster (or create a new one), filter by size, merge if needed.

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

Three built-in presets for common scenarios:

| Preset | Threshold | Min Size | Max Clusters | Best for |
|--------|:---------:|:--------:|:------------:|----------|
| `high-precision` | 0.95 | 3 | 10 | Tight, highly cohesive groups |
| `balanced` | 0.85 | 5 | 5 | General-purpose clustering |
| `performance` | 0.75 | 10 | 3 | Fast, broad groupings |

```typescript
const clusters = clusterEmbeddings(embeddings, getPreset('balanced'));
```

#### Quality metrics

```typescript
// Internal cluster quality (0-1, higher = tighter)
const cohesion = cohesionScore(clusters[0]);

// Global clustering quality (-1 to 1, higher = better separated)
const quality = silhouetteScore(clusters);
```

#### Operations

```typescript
// Classify a new embedding into existing clusters
const { clusterIndex, similarity } = assignToCluster(newEmbedding, clusters, { threshold: 0.8 });
// clusterIndex = -1 if below threshold (outlier)

// Merge two clusters
const merged = mergeClusters(clusters[0], clusters[1]);
```

---

### Aggregation

Combine multiple embeddings into a single vector.

```typescript
import { averageEmbeddings, weightedAverage, incrementalAverage, centroid, maxPooling, minPooling, combineEmbeddings } from 'embedding-utils';

// Element-wise average
averageEmbeddings([[1, 0], [0, 1]]); // => [0.5, 0.5]

// Weighted average (give more importance to certain embeddings)
weightedAverage([[1, 0], [0, 1]], [3, 1]); // => [0.75, 0.25]

// Streaming / incremental average (no need to store all prior embeddings)
let avg = [1, 2, 3];
avg = incrementalAverage(avg, [4, 5, 6], 1); // count = embeddings already averaged
avg = incrementalAverage(avg, [7, 8, 9], 2);
// Numerically equivalent to averageEmbeddings([[1,2,3], [4,5,6], [7,8,9]])

// Element-wise max/min pooling
maxPooling([[1, 0, 3], [2, 5, 1]]); // => [2, 5, 3]
minPooling([[1, 0, 3], [2, 5, 1]]); // => [1, 0, 1]

// Embed multiple texts and aggregate in one call
const combined = await combineEmbeddings(['hello', 'world'], provider);
const pooled = await combineEmbeddings(['hello', 'world'], provider, { aggregate: maxPooling });
```

---

### Quantization

Reduce embedding precision for storage and memory efficiency.

| Type | Bits | Size vs float32 | Input Range | Precision Loss | Best for |
|------|:----:|:---------------:|:-----------:|:--------------:|----------|
| `fp16` | 16 | 50% | Any | Negligible | Default choice, wide value ranges |
| `int8` | 8 | 25% | [-1, 1] | ~0.8% | Normalized embeddings |
| `uint8` | 8 | 25% | [0, 1] | ~0.4% | Positive embeddings |
| `binary` | 1 | 3% | Any | High (sign only) | Extreme compression, candidate filtering |

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

---

### Storage

Serialize embeddings for persistence and cache frequently used results.

#### Serialization

Three formats with different trade-offs:

| Format | Output Type | Relative Size | Use Case |
|--------|:-----------:|:-------------:|----------|
| `json` | `string` | Largest | Human-readable, debugging, APIs |
| `binary` | `Uint8Array` | Smallest | File storage, binary protocols |
| `base64` | `string` | ~33% > binary | Text-safe transmission (HTTP, databases) |

```typescript
import { serialize, deserialize } from 'embedding-utils';

const embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];

const json = serialize(embeddings, 'json');     // JSON string
const binary = serialize(embeddings, 'binary'); // Uint8Array
const base64 = serialize(embeddings, 'base64'); // base64 string

const restored = deserialize(json, 'json');     // number[][] (identical to original)
```

#### LRU Cache

In-memory cache with configurable size limits and TTL. Implements the `CacheProvider` interface for pluggability.

```typescript
import { createLRUCache } from 'embedding-utils';

const cache = createLRUCache({ maxSize: 1000, ttl: 60000 }); // 60s TTL

await cache.set('doc-1', [[0.1, 0.2, 0.3]]);
const cached = await cache.get('doc-1'); // => [[0.1, 0.2, 0.3]]
await cache.has('doc-1');                // => true
await cache.delete('doc-1');
await cache.clear();
```

The async interface allows swapping in custom backends (Redis, SQLite, etc.) by implementing `CacheProvider`.

---

### Model Management

Manage local ONNX models for the local provider.

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

---

## TypeScript

Full strict-mode type definitions ship with the package -- no `@types` install needed.

```typescript
import type {
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
} from 'embedding-utils';
```

Types are fully inferred -- you rarely need explicit annotations:

```typescript
const results = topK(query, corpus, 5);
//    ^? SearchResult[] -- fully typed

results[0].score;     // number
results[0].embedding; // number[]
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

# embedding-utils

[![npm version](https://img.shields.io/npm/v/embedding-utils)](https://www.npmjs.com/package/embedding-utils)
[![license](https://img.shields.io/npm/l/embedding-utils)](./LICENSE)

Lightweight, provider-agnostic embedding utilities for Node.js. Vector math, similarity search, clustering, quantization, and embedding generation with zero production dependencies.

## Features

- **Vector Math** - Cosine similarity, dot product, Euclidean/Manhattan distance, normalization, arithmetic
- **Search** - Top-K, threshold filtering, deduplication, ranking, similarity matrices
- **Clustering** - Greedy agglomerative clustering with presets and quality metrics
- **Aggregation** - Average, weighted average, pooling, incremental updates
- **Quantization** - fp16, int8, uint8, binary quantization/dequantization
- **Storage** - JSON/binary/base64 serialization, LRU cache
- **Providers** - Local (ONNX), OpenAI, Cohere, Google Vertex, Voyage, Mistral, Jina, OpenRouter
- **TypeScript** - Full type definitions included, works out of the box

## Installation

```bash
npm install embedding-utils
```

For local (ONNX) embedding generation, also install the optional peer dependency:

```bash
npm install @huggingface/transformers
```

## Quick Start

```typescript
import {
  cosineSimilarity,
  topK,
  clusterEmbeddings,
  createOpenAICompatibleProvider,
} from 'embedding-utils';

// Vector math
const similarity = cosineSimilarity([1, 0, 0], [0, 1, 0]); // 0

// Search
const results = topK(queryEmbedding, corpus, 5);

// Clustering
const clusters = clusterEmbeddings(embeddings, { similarityThreshold: 0.9 });

// Cloud provider
const provider = createOpenAICompatibleProvider({
  apiKey: process.env.OPENAI_API_KEY!,
  model: 'text-embedding-3-small',
});
const result = await provider.embed('Hello world');
```

## API Reference

### Math

```typescript
import {
  cosineSimilarity,
  dotProduct,
  euclideanDistance,
  manhattanDistance,
  normalize,
  magnitude,
  add,
  subtract,
  scale,
  truncateDimensions,
} from 'embedding-utils';

cosineSimilarity([1, 0], [0, 1]); // 0
dotProduct([1, 2, 3], [4, 5, 6]); // 32
euclideanDistance([0, 0], [3, 4]); // 5
normalize([3, 4]); // [0.6, 0.8]
truncateDimensions([1, 2, 3, 4], 2); // [1, 2]
```

### Aggregation

```typescript
import {
  averageEmbeddings,
  weightedAverage,
  incrementalAverage,
  centroid,
  maxPooling,
  minPooling,
  combineEmbeddings,
} from 'embedding-utils';

averageEmbeddings([[1, 0], [0, 1]]); // [0.5, 0.5]
weightedAverage([[1, 0], [0, 1]], [3, 1]); // [0.75, 0.25]
maxPooling([[1, 0, 3], [2, 5, 1]]); // [2, 5, 3]

// Combine text embeddings via a provider
const combined = await combineEmbeddings(['hello', 'world'], provider);
```

### Search

```typescript
import {
  topK,
  topKMulti,
  aboveThreshold,
  deduplicate,
  rankBySimilarity,
  similarityMatrix,
} from 'embedding-utils';

const results = topK(query, corpus, 10, { metric: 'cosine', labels: ['a', 'b', 'c'] });
// [{ index, score, embedding, label }]

const unique = deduplicate(embeddings, 0.95);
// { embeddings, indices }

const matrix = similarityMatrix(embeddings);
// number[][] (NxN symmetric)
```

### Clustering

```typescript
import {
  clusterEmbeddings,
  CLUSTERING_PRESETS,
  getPreset,
  cohesionScore,
  silhouetteScore,
  assignToCluster,
  mergeClusters,
} from 'embedding-utils';

const clusters = clusterEmbeddings(embeddings, getPreset('balanced'));
// [{ centroid, members, size, cohesion }]

const quality = silhouetteScore(clusters);
const assignment = assignToCluster(newEmbedding, clusters, { threshold: 0.8 });
```

### Storage

```typescript
import { serialize, deserialize, createLRUCache } from 'embedding-utils';

const json = serialize(embeddings, 'json');
const binary = serialize(embeddings, 'binary');
const base64 = serialize(embeddings, 'base64');
const restored = deserialize(json, 'json');

const cache = createLRUCache({ maxSize: 1000, ttl: 60000 });
await cache.set('key', embeddings);
const cached = await cache.get('key');
```

### Quantization

```typescript
import { quantize, dequantize, getQuantizationInfo } from 'embedding-utils';

const quantized = quantize([0.5, -0.3, 0.8], 'int8'); // Int8Array
const restored = dequantize(quantized, 'int8'); // number[]
const info = getQuantizationInfo('int8');
// { bits: 8, range: [-128, 127], description: '...' }
```

### Providers

```typescript
import {
  createLocalProvider,
  createOpenAICompatibleProvider,
  createCohereProvider,
  createGoogleVertexProvider,
  createProvider,
} from 'embedding-utils';
```

| Provider | Factory | Config |
|----------|---------|--------|
| Local (ONNX) | `createLocalProvider()` | `{ model?, precision? }` |
| OpenAI | `createOpenAICompatibleProvider(config)` | `{ apiKey, model }` |
| Voyage | `createProvider('voyage', config)` | `{ apiKey, model, baseUrl? }` |
| Mistral | `createProvider('mistral', config)` | `{ apiKey, model }` |
| Jina | `createProvider('jina', config)` | `{ apiKey, model }` |
| OpenRouter | `createProvider('openrouter', config)` | `{ apiKey, model }` |
| Cohere | `createCohereProvider(config)` | `{ apiKey, model? }` |
| Google Vertex | `createGoogleVertexProvider(config)` | `{ projectId, accessToken }` |

```typescript
// OpenAI
const openai = createOpenAICompatibleProvider({
  apiKey: 'sk-...',
  model: 'text-embedding-3-small',
  dimensions: 256,
});

// Cohere
const cohere = createCohereProvider({ apiKey: 'co-...' });

// Google Vertex AI
const vertex = createGoogleVertexProvider({
  projectId: 'my-project',
  accessToken: 'ya29...',
});

// Local ONNX (requires @huggingface/transformers)
const local = createLocalProvider({ model: 'Xenova/all-MiniLM-L6-v2' });

// All providers share the same interface
const result = await openai.embed('hello world');
// { embeddings: number[][], model: string, dimensions: number }
```

### Models

```typescript
import {
  downloadModel,
  listModels,
  deleteModel,
  setModelPath,
  getModelInfo,
  MODEL_REGISTRY,
} from 'embedding-utils';

const info = getModelInfo('Xenova/all-MiniLM-L12-v2');
const path = await downloadModel('Xenova/all-MiniLM-L6-v2');
const cached = await listModels();
await deleteModel('Xenova/all-MiniLM-L6-v2');
```

## TypeScript

Full type definitions are included. Import types directly:

```typescript
import type {
  EmbeddingProvider,
  EmbeddingResult,
  SearchResult,
  Cluster,
  ClusteringConfig,
  SimilarityMetric,
} from 'embedding-utils';
```

## License

MIT

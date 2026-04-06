# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2026-04-06

### Breaking Changes

- **Float32Array migration:** All vector-returning functions now return `Float32Array` instead of `number[]`. This affects `provider.embed()`, `normalize()`, `topK()`, `clusterEmbeddings()`, `deserialize()`, and all other functions that produce embeddings. All functions still accept both `number[]` and `Float32Array` as input. See the Migration Guide in README for common patterns and gotchas.
- **`truncateDimensions` auto-normalizes:** Output vectors are now L2-normalized after truncation, ready for cosine similarity without a separate `normalize()` call.

### New Features

#### Search
- **HNSW Index (FR-8):** `HNSWIndex` class for approximate nearest neighbor search with configurable M, efConstruction, efSearch, all four similarity metrics, string IDs with metadata, filtered search, and binary serialization/deserialization
- **Reciprocal Rank Fusion (FR-9):** `fuseRankedLists()` merges multiple ranked lists without score calibration
- **Score Normalization (FR-10):** `normalizeScores()` with min-max, z-score, and sigmoid methods

#### Clustering
- **HDBSCAN (FR-7):** `hdbscan()` density-based clustering with automatic cluster count detection, noise point identification, and configurable minClusterSize/minSamples/metric

#### Evaluation
- **Retrieval Metrics (FR-11):** `recallAtK()`, `ndcg()`, `mrr()`, `meanAveragePrecision()` for measuring search quality against ground truth

#### Pipeline
- **Async Embedding Pipeline (FR-13):** `createEmbeddingPipeline()` with configurable batch size, concurrency, token-bucket rate limiting, progress callbacks, and checkpoint/resume support

#### Math
- **Random Projection (FR-12):** `createRandomProjection()` for Johnson-Lindenstrauss dimensionality reduction with deterministic seeding and batch projection

#### Quantization
- **Calibrated Quantization (FR-15):** `calibrate()`, `calibratedQuantize()`, `calibratedDequantize()` learn per-dimension value ranges for tighter int8 mapping
- **Hamming Distance (FR-14):** `hammingDistance()` and `hammingSimilarity()` for fast binary vector comparison

#### Text
- **Markdown-Aware Chunking (FR-6):** `chunkByStructure()` splits markdown respecting code fences, lists, tables, and headings with breadcrumb metadata

#### Providers
- **New provider presets:** `together`, `fireworks`, `nomic`, `mixedbread` via `createProvider()` factory

#### Foundation
- **Float32Array migration (FR-1):** All internal vector operations use Float32Array for ~50% memory reduction and improved computation speed
- **Vector utilities:** `toFloat32()` conversion helper and `isVector()` type guard

### New Namespaces

- `Eval` — `{ recallAtK, ndcg, mrr, meanAveragePrecision }`
- `Pipeline` — `{ createEmbeddingPipeline, TokenBucketRateLimiter }`

### Updated Namespaces

- `Search` — added `HNSWIndex`, `fuseRankedLists`, `normalizeScores`
- `Clustering` — added `hdbscan`
- `Math` — added `createRandomProjection`

## [0.2.0] - 2026-04-03

### Bug Fixes

- Unified duplicated `computeScore` metric dispatch into a single internal module
- All `switch(metric)` blocks now throw on unknown values instead of silently returning undefined
- Consolidated duplicated `validateVectorPair` logic with consistent error messages
- Extracted duplicated clustering centroid computation into shared helper

### New Features

#### Performance
- `topK` uses heap-based selection (O(n log k)) for small k, significantly faster on large corpora
- Deterministic clustering via `shuffle: true, shuffleSeed: N` using seeded PRNG
- `cosineDistance(a, b)`, `isNormalized(v)`, `validateDimensions(embeddings)`

#### Type Safety & Errors
- `EmbeddingUtilsError` base class with `ValidationError`, `DimensionMismatchError`, `ProviderError`, `ModelNotFoundError`
- `SearchOptions` interface with `metric`, `labels`, and `filter` callback
- Replaced all `any` types in providers with proper types

#### Providers
- `withCache(provider, opts?)` -- Caching middleware for any provider
- `warmCache(cache, entries)` -- Pre-populate cache
- Typed provider configs: `LocalProviderConfig`, `OpenAICompatibleConfig`, `CohereConfig`, `GoogleVertexConfig`

#### Clustering
- `centroidCohesion`, `clusterStats`, `detectOutliers`, `centroidDrift`
- `findOptimalK` and `silhouetteByK` for automatic K selection
- `IncrementalClusterer` class for online clustering
- `assignmentStrategy` option (`'centroid'` or `'average-similarity'`)
- `legacy` preset (single cluster, clustering disabled)
- Cluster redistribution preserves all data points
- `clusterEmbeddings` accepts optional `labels` parameter to track source text through clustering, redistribution, and merging (each cluster's `labels` array maps 1:1 to its `members`)

#### Search
- `pairwiseSimilarity`, `mmrSearch`, `rerankResults`
- `SearchIndex` class with CRUD + brute-force search
- `filter` callback in search options

#### High-Level APIs
- `createEmbeddingStore(config)` -- Provider + cache + search in one
- `chunkByTokenCount` and `chunkBySentence` for text splitting
- `batchIncrementalAverage` for batch streaming averages

#### Models & Storage
- `registerModel`, `getRecommendedModel` for custom model registry
- `estimateMemorySavings` for quantization planning
- Browser-safe base64 serialization (no `Buffer` dependency)

## [0.1.0] - 2026-03-30

### Added

- **Math module:** `cosineSimilarity`, `dotProduct`, `euclideanDistance`, `manhattanDistance`, `normalize`, `magnitude`, `add`, `subtract`, `scale`, `truncateDimensions`
- **Aggregation module:** `averageEmbeddings`, `weightedAverage`, `incrementalAverage`, `centroid`, `maxPooling`, `minPooling`, `combineEmbeddings`
- **Search module:** `topK`, `topKMulti`, `aboveThreshold`, `deduplicate`, `rankBySimilarity`, `similarityMatrix`
- **Clustering module:** `clusterEmbeddings`, `CLUSTERING_PRESETS`, `getPreset`, `cohesionScore`, `silhouetteScore`, `assignToCluster`, `mergeClusters`
- **Storage module:** `serialize`/`deserialize` (JSON, binary, base64), `createLRUCache` with pluggable `CacheProvider` interface
- **Quantization module:** `quantize`/`dequantize` (fp16, int8, uint8, binary), `getQuantizationInfo`
- **Providers:** Local ONNX via `@huggingface/transformers`, OpenAI-compatible (OpenAI, Voyage, Mistral, Jina, OpenRouter), Cohere, Google Vertex AI
- **Provider factory:** `createProvider` with type aliases for all supported providers
- **Model management:** `downloadModel`, `listModels`, `deleteModel`, `setModelPath`, `getModelInfo`, `MODEL_REGISTRY`
- **Shared utilities:** `retryWithBackoff` (exponential backoff with jitter), `autoBatch`
- Dual ESM/CJS build with full TypeScript declarations
- Zero production dependencies
- Apache-2.0 license

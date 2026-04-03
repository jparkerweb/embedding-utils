# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-04-03

### Breaking Changes

- **Binary serialization format v2** -- New binary format is now the default. v1 data is auto-detected and read transparently by `deserialize()`, but newly serialized data uses v2.
- **`SearchOptions` replaces inline types** -- Search functions now use the `SearchOptions` interface instead of inline `{ metric?, labels? }` parameters.
- **Optional peer dependency bumped to `@huggingface/transformers ^4.0.0`** -- Users on v3 should upgrade. The library tests compatibility with both.
- **Domain error classes replace plain `Error`** -- All errors now extend `EmbeddingUtilsError` with specific subclasses (`ValidationError`, `DimensionMismatchError`, `ProviderError`, `ModelNotFoundError`). Code catching generic `Error` will still work, but you can now catch more precisely.

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

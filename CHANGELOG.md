# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Maintenance

Tooling and dependency work only — **no changes to the public API or to runtime behavior**. Consumers are unaffected.

- **TypeScript 6:** upgraded from 5.9 to 6.0.3, the highest major currently supported by `@typescript-eslint` v8 (`>=4.8.4 <6.1.0`). TypeScript 7 is deliberately deferred until typescript-eslint supports it. `tsconfig.json` now sets `"ignoreDeprecations": "6.0"`, which TS 6 requires because tsup's dts pipeline injects the now-deprecated `baseUrl`.
- **All npm audit advisories resolved (5 → 0).** Added `overrides` for `sharp` (libvips CVEs), `adm-zip`, and `esbuild`, each of which was pinned below its fixed version by a dependency with no upstream fix available. All three sat under `@huggingface/transformers`, the optional peer dependency, so consumers not using the local ONNX provider were never exposed. Every override is documented with a condition for its removal.
- **`package-lock.json` is now tracked.** It was previously gitignored, so installs were not reproducible and the `overrides` above had no recorded resolution. Use `npm ci`.
- **Fixed `npm run format`, which was a silent no-op on Windows.** Its globs were single-quoted, which `cmd`/PowerShell do not strip, so Prettier matched zero files and exited successfully. As a result 90 files had been formatted at Prettier's defaults rather than this project's `.prettierrc` (100 columns, `es5` trailing commas); the configured style is now applied throughout. Added `npm run format:check`.
- **`npm run lint` now covers `tests/` as well as `src/`.** `eslint.config.mjs` had always declared `tests/**`, but the script never passed it. Cleared the dead code this surfaced — unused imports, a vestigial `EmbeddingStore`, and an abandoned serialization buffer — and relaxed `no-explicit-any` for test files only.
- **Added CI** (`.github/workflows/ci.yml`): format, lint, typecheck, test and build across Node 20/22/24, plus an `engines-floor` job that runs the built `dist/` on Node 18 to verify the `engines.node` claim (the dev toolchain itself now requires Node >=20), and a weekly `npm audit`.
- **`scripts/` is now linted and formatted**, and `.gitattributes` pins the tree to LF line endings so files authored on Windows can no longer enter the index with CRLF.

## [0.6.0] - 2026-08-12

### New Features

- **Shared ONNX pipelines (session reuse):** `createLocalProvider` now shares one underlying `@huggingface/transformers` pipeline per unique `model`/`precision`/`device`/`modelPath`/`cacheDir`/`allowRemoteModels` combination via a bounded (LRU, max 4) process-level registry. Pipeline construction creates a native ONNX InferenceSession (~1.5s and hundreds of MB for typical models) that transformers.js does not cache — previously every `createLocalProvider(...)` call with the same config paid that cost again on first `embed()`. Pooling and prefixes are per-inference arguments and deliberately do **not** split the session, so providers differing only in `pooling`/`documentPrefix`/`queryPrefix` share one pipeline. Opt out per provider with the new `LocalProviderConfig.reuse: false` (private session). Failed constructions are evicted so the next call retries.
- **Shared tokenizers:** `createTokenizer(...).load()` similarly shares one loaded tokenizer per unique `model`/`modelPath`/`cacheDir`/`allowRemoteModels` combination (bounded registry, max 8), skipping the ~350ms tokenizer-file re-parse `AutoTokenizer.from_pretrained` performs on every call. Tokenizers are stateless, so sharing is safe. Opt out with the new `CreateTokenizerOptions.reuse: false`.
- **`disposeLocalPipelines()`:** disposes every shared pipeline (releasing its ONNX session via the pipeline's `dispose()`) and empties the registry. Call on shutdown or between tests. Providers created before the call must not be used afterwards.
- **`disposeLocalTokenizers()`:** empties the tokenizer registry (nothing native to dispose; existing tokenizers keep working).

### Notes

- **Behavior-compatible, memory-profile change:** numeric output is unchanged for every configuration. The only observable differences are (a) construction cost drops to ~0 for repeated same-config providers/tokenizers, and (b) a process that iterates over many models now holds up to the registry bounds (4 pipelines / 8 tokenizers) alive concurrently instead of one-at-a-time GC. Registry eviction drops the reference without disposing — active holders keep working and memory is reclaimed when they release it.
- The live smoke test (`npm run smoke`) now also verifies that a second same-config provider reuses the shared pipeline and produces bit-identical vectors.

## [0.5.0] - 2026-05-28

### New Features

- **Exact token counting (`createTokenizer`):** New standalone `createTokenizer(model, opts?)` factory backed by a model's `@huggingface/transformers` tokenizer. Returns a `LocalTokenizer` with an idempotent `load()`, synchronous `count(text)` / `countBatch(texts)` (matching transformers' unpadded `input_ids.size`), and `readonly maxTokens` / `modelId`. Counting is synchronous after `load()`, so it can be used inline in hot loops. Calling `count`/`countBatch` before `load()` throws `EmbeddingUtilsError`; a missing `@huggingface/transformers` peer throws `ModelNotFoundError` from `load()`.
- **`device` option for the local provider:** `LocalProviderConfig.device` selects the transformers execution provider (e.g. `'cpu'` / `'webgpu'`). It is passed through to the pipeline for non-`'webgpu'` values and intentionally omitted for `'webgpu'` (passing it breaks WebGPU initialization).
- **`q4` precision:** `LocalProviderConfig.precision` now also accepts `'q4'` (int4 quantized) in addition to `'fp32'` / `'fp16'` / `'q8'`.

### Fixed

- **Now-honored local provider config:** `modelPath`, `cacheDir`, and `allowRemoteModels` were declared on `LocalProviderConfig` but ignored. `createLocalProvider` now applies them to the transformers environment (`env.localModelPath`, `env.cacheDir`, `env.allowRemoteModels`) so callers can point at local models / custom caches and toggle remote downloads.
- **Per-input embedding cache:** The local provider's cache is now keyed per input text (`` `${model}:${dimensions}:${text}` ``) instead of per whole batch, so repeated or overlapping texts hit the cache across batches and only cache misses run the pipeline.
- **Empty-input guard:** `createLocalProvider().embed([])` now returns `{ embeddings: [], model, dimensions: 0 }` instead of throwing on an empty input array.

### Notes

- All changes are additive; the 0.4.0 public surface is unchanged.

## [0.4.0] - 2026-05-25

### New Features

- **Expanded model registry:** Added 19 ONNX embedding models to the built-in registry, all auto-downloadable from HuggingFace and under 2GB. New families include GTE (small/base/large + multilingual), E5 v2 (small/base/large), multilingual E5 (small/base/large), Jina v2 (small/base, 8K context), BGE large + BGE-M3, mxbai-embed-large-v1, Snowflake Arctic Embed (m-v1.5/l), and Qwen3-Embedding-0.6B (32K context, instruction-aware).
- **Configurable pooling:** Added a `pooling` field (`'mean' | 'cls' | 'last_token'`) to `ModelInfo` and `LocalProviderConfig`. The local provider now resolves the correct pooling method per model instead of hardcoding mean pooling, enabling CLS-pooled (BGE, mxbai, Snowflake Arctic) and last-token-pooled (Qwen3) models to produce correct embeddings.
- **Automatic registry prefixes:** The local provider now applies a model's registry `prefixes` based on `inputType` ('document'/'query'), so asymmetric models (E5, BGE, nomic) work correctly without manually copying prefixes into config. Explicit `documentPrefix`/`queryPrefix` config still takes precedence.

### Fixed

- **BGE pooling:** `bge-small-en-v1.5` and `bge-base-en-v1.5` now use CLS pooling (matching how BGE was trained) instead of mean pooling. This changes the embeddings these two models produce — re-embed and re-index any vectors generated by them with a prior version.

### Changed

- **`getRecommendedModel('quality')`** now returns `Xenova/bge-large-en-v1.5` (was `all-mpnet-base-v2`).

### Notes

- Three new models (`Qwen3-Embedding-0.6B-ONNX`, `multilingual-e5-large`, `bge-m3`) exceed 2GB at the default `fp32` precision — pass `precision: 'q8'` or `'fp16'` to `createLocalProvider` to stay under that.

## [0.3.1] - 2026-04-23

### Changed

- Bumped `@huggingface/transformers` peer/dev dependency from `^4.0.0` to `^4.2.0`.

## [0.3.0] - 2026-04-06

### Breaking Changes

- **Float32Array migration:** All vector-returning functions now return `Float32Array` instead of `number[]`. This affects `provider.embed()`, `normalize()`, `topK()`, `clusterEmbeddings()`, `deserialize()`, and all other functions that produce embeddings. All functions still accept both `number[]` and `Float32Array` as input. See the Migration Guide in README for common patterns and gotchas.
- **`truncateDimensions` auto-normalizes:** Output vectors are now L2-normalized after truncation, ready for cosine similarity without a separate `normalize()` call.

### New Features

#### Search
- **HNSW Index:** `HNSWIndex` class for approximate nearest neighbor search with configurable M, efConstruction, efSearch, all four similarity metrics, string IDs with metadata, filtered search, and binary serialization/deserialization
- **Reciprocal Rank Fusion:** `fuseRankedLists()` merges multiple ranked lists without score calibration
- **Score Normalization:** `normalizeScores()` with min-max, z-score, and sigmoid methods

#### Clustering
- **HDBSCAN:** `hdbscan()` density-based clustering with automatic cluster count detection, noise point identification, and configurable minClusterSize/minSamples/metric

#### Evaluation
- **Retrieval Metrics:** `recallAtK()`, `ndcg()`, `mrr()`, `meanAveragePrecision()` for measuring search quality against ground truth

#### Pipeline
- **Async Embedding Pipeline:** `createEmbeddingPipeline()` with configurable batch size, concurrency, token-bucket rate limiting, progress callbacks, and checkpoint/resume support

#### Math
- **Random Projection:** `createRandomProjection()` for Johnson-Lindenstrauss dimensionality reduction with deterministic seeding and batch projection

#### Quantization
- **Calibrated Quantization:** `calibrate()`, `calibratedQuantize()`, `calibratedDequantize()` learn per-dimension value ranges for tighter int8 mapping
- **Hamming Distance:** `hammingDistance()` and `hammingSimilarity()` for fast binary vector comparison

#### Text
- **Markdown-Aware Chunking:** `chunkByStructure()` splits markdown respecting code fences, lists, tables, and headings with breadcrumb metadata

#### Providers
- **New provider presets:** `together`, `fireworks`, `nomic`, `mixedbread` via `createProvider()` factory

#### Foundation
- **Float32Array migration:** All internal vector operations use Float32Array for ~50% memory reduction and improved computation speed
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

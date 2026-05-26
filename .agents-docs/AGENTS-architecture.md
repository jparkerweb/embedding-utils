# Architecture
> Part of [AGENTS.md](../AGENTS.md) — project guidance for AI coding agents.

## Big picture

The library is a collection of independent **feature modules** under `src/`. Each module is a directory containing implementation files plus an `index.ts` **barrel** that re-exports its public functions. The root `src/index.ts` then re-exports every module barrel — first as flat named exports (for backward compatibility), then bundled into grouped namespace objects (`Math`, `Search`, `Clustering`, `Eval`, `Pipeline`). When you add a public function, it must be exported from its module's `index.ts` **and** wired into `src/index.ts` (and the relevant namespace object if applicable).

All public types and error classes live in a single `src/types.ts` and are re-exported from the root. Errors form a hierarchy rooted at `EmbeddingUtilsError` (`EmbeddingError`, `ValidationError`, `DimensionMismatchError`, `ProviderError`, `ModelNotFoundError`).

## Module map (`src/`)

| Module | Responsibility |
|--------|----------------|
| `math/` | Vector ops: cosine/dot/euclidean/manhattan distance, normalize, add/subtract/scale, dimension truncation & validation, random projection. |
| `search/` | `topK`, threshold/dedup, ranking, similarity matrix, pairwise, rerank, **MMR**, `SearchIndex`, RRF fusion + score normalization, **HNSW** ANN index. |
| `clustering/` | k-means via presets (`CLUSTERING_PRESETS`, `getPreset`), `findOptimalK`/silhouette, cohesion metrics, outlier detection, centroid drift, `IncrementalClusterer`, **HDBSCAN**. |
| `aggregation/` | average / weighted / incremental / batch-incremental averaging, centroid, max/min pooling, `combineEmbeddings`. |
| `quantization/` | `quantize`/`dequantize`, calibration, hamming distance/similarity, memory-savings estimation, info. |
| `providers/` | `EmbeddingProvider` implementations: `local` (ONNX/transformers), `openai-compatible`, `cohere`, `google` (Vertex), a `factory`, plus `middleware` (`retryWithBackoff`, `autoBatch`, `withCache`). `types.ts` holds typed API response shapes. |
| `models/` | Local model registry & lifecycle: download/list/delete, `setModelPath`, `MODEL_REGISTRY`, `registerModel`, `getRecommendedModel`. |
| `pipeline/` | Async batched embedding pipeline (`createEmbeddingPipeline`), `TokenBucketRateLimiter`, checkpointing. |
| `text/` | Chunking (`chunkByTokenCount`, `chunkBySentence`, `chunkByStructure`), markdown-aware structure, tokenizer info. |
| `storage/` | Serialize/deserialize embeddings, LRU cache (`createLRUCache`, `warmCache`). |
| `store/` | `createEmbeddingStore` — higher-level combined embed+store+search store. |
| `eval/` | Retrieval metrics: `recallAtK`, `ndcg`, `mrr`, `meanAveragePrecision`. |
| `internal/` | **Non-public** shared helpers: clustering primitives, concurrency, heap, metrics, random, validation, vector-utils. Marked `@internal`; not re-exported (except a couple of low-level utils like `toFloat32`, `isVector`). |

## Provider model

Every provider implements the same `EmbeddingProvider` interface, so application code can swap providers with a single line. Middleware (`retryWithBackoff`, `autoBatch`, `withCache`) wraps providers compositionally. Cloud provider response parsing is typed at the JSON boundary via `src/providers/types.ts` rather than `any`.

## Adding models to the registry (`src/models/registry.ts`)

When adding an entry to `MODEL_REGISTRY`, set `pooling` and `prefixes` to match how the model was **trained** — the local provider (`src/providers/local.ts`) reads both from the registry (explicit `LocalProviderConfig` values override). Getting these wrong produces vectors that look valid (normalized, correct dimensions) but are silently degraded.

- **Pooling is load-bearing.** Verify against the model's authoritative `1_Pooling/config.json` on HuggingFace, **not** the Transformers.js model-card example (those sometimes show the wrong value, e.g. Xenova's BGE cards show `mean` though BGE is CLS-trained). Rule of thumb from verified configs: BGE (all, incl. M3), mxbai, Snowflake Arctic → `cls`; E5 / multilingual-E5, GTE (English), Jina v2, MiniLM/mpnet → `mean`; gte-multilingual-base → `cls`; Qwen3-Embedding → `last_token`. Supported values come from the `@huggingface/transformers` feature-extraction pipeline (`mean`/`cls`/`last_token`).
- **Prefixes** are applied automatically by `inputType` ('document'/'query'). E5-family models *require* `query: `/`passage: `; BGE/mxbai/Snowflake use a query-only retrieval instruction.
- **>2GB at fp32:** note it in the `description` and steer users to `precision: 'q8'`/`'fp16'`.
- Registry invariants are guarded in `tests/models/registry.test.ts`; add behavioral pooling/prefix assertions in `tests/providers/local.test.ts`.

## Data representation

Embeddings are represented as `Float32Array` (the `Vector` type) as of v0.3 — not `number[]`. Keep this in mind when adding math/search code or comparing against older examples.

## Design specs

`specs/v03-comprehensive-upgrade/` holds the design notes for the v0.3 upgrade. Consult it for rationale behind the v0.3 API/shape changes.

# Changelog

All notable changes to this project will be documented in this file.

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

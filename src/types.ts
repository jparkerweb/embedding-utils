// ─────────────────────────────────────────────────────────────────────────────
// Error Classes
//
// All errors in embedding-utils extend EmbeddingError, forming a hierarchy:
//   EmbeddingError
//   ├── ValidationError   — bad inputs (empty arrays, dimension mismatches)
//   ├── ProviderError     — cloud API failures (auth, rate limits, server errors)
//   └── ModelNotFoundError — local model or dependency not found
//
// Catch EmbeddingError to handle any library error; catch subclasses for
// fine-grained control.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Base error class for all embedding-utils errors.
 *
 * Every error thrown by this library is an instance of EmbeddingUtilsError or one
 * of its subclasses. Use `instanceof EmbeddingUtilsError` to catch all library
 * errors in a single handler.
 */
export class EmbeddingUtilsError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EmbeddingUtilsError';
  }
}

/**
 * @deprecated Use {@link EmbeddingUtilsError} instead. This alias will be removed in v1.0.
 */
export const EmbeddingError = EmbeddingUtilsError;

/**
 * Thrown when function inputs fail validation.
 *
 * Common causes:
 * - Empty embedding arrays passed to math, aggregation, or clustering functions
 * - Dimension mismatches between two vectors being compared
 * - Invalid configuration values (e.g., targetDims <= 0)
 * - Unknown provider type in the factory
 */
export class ValidationError extends EmbeddingUtilsError {
  constructor(message: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

/**
 * Thrown when two vectors have incompatible dimensions.
 *
 * A more specific subclass of {@link ValidationError} for dimension mismatches,
 * allowing callers to catch dimension errors separately from other validation failures.
 */
export class DimensionMismatchError extends ValidationError {
  constructor(message: string) {
    super(message);
    this.name = 'DimensionMismatchError';
  }
}

/**
 * Thrown when a cloud embedding provider returns an HTTP error.
 *
 * The `statusCode` field carries the HTTP status code (e.g., 429 for rate limiting,
 * 401 for auth failure, 500+ for server errors). The `provider` field identifies
 * which provider failed (e.g., 'openai-compatible', 'cohere', 'google-vertex').
 *
 * Errors with status 429 or 5xx are automatically retried by the built-in
 * retry logic. Other 4xx errors (400, 401, 403) fail immediately.
 */
export class ProviderError extends EmbeddingUtilsError {
  /** HTTP status code from the provider (e.g., 429, 500). Undefined if the error did not originate from an HTTP response. */
  statusCode?: number;
  /** Identifier for the provider that threw this error (e.g., 'cohere', 'google-vertex'). */
  provider: string;

  constructor(message: string, provider: string, statusCode?: number) {
    super(message);
    this.name = 'ProviderError';
    this.provider = provider;
    this.statusCode = statusCode;
  }
}

/**
 * Thrown when a local ONNX model or the @huggingface/transformers package
 * cannot be found.
 *
 * This typically means the optional peer dependency is not installed.
 * Fix: `npm install @huggingface/transformers`
 */
export class ModelNotFoundError extends EmbeddingUtilsError {
  constructor(message: string) {
    super(message);
    this.name = 'ModelNotFoundError';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Type Aliases
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A vector of numeric values, accepted by all embedding-utils functions.
 *
 * Functions that **accept** vectors take `Vector` (both `number[]` and
 * `Float32Array` are valid inputs). Functions that **return** vectors
 * always return `Float32Array` for memory efficiency and type safety.
 *
 * **Migration from v0.2:** If your code destructures or spreads returned
 * vectors, note that `[...float32Array]` produces `number[]` and
 * `JSON.stringify(float32Array)` produces `{"0": val, ...}` not `[val, ...]`.
 */
export type Vector = number[] | Float32Array;

/**
 * Supported similarity / distance metrics used across search, clustering,
 * and aggregation functions.
 *
 * - `'cosine'` — Cosine similarity (angle between vectors, range -1 to 1).
 *   The default for most operations. Best for comparing semantic meaning
 *   regardless of vector magnitude.
 * - `'dot'` — Dot product (raw scalar product). Use when your model is
 *   optimized for dot-product ranking (e.g., some OpenAI Matryoshka models).
 * - `'euclidean'` — Euclidean (L2) distance, converted to similarity as
 *   `1 / (1 + distance)`. Useful for anomaly detection and k-means-style
 *   clustering where absolute position matters.
 * - `'manhattan'` — Manhattan (L1) distance, converted to similarity as
 *   `1 / (1 + distance)`. More robust to noisy high-dimensional data and
 *   cheaper to compute than euclidean.
 */
export type SimilarityMetric = 'cosine' | 'dot' | 'euclidean' | 'manhattan';

/**
 * Supported embedding provider types for the {@link createProvider} factory.
 *
 * - `'local'` — Offline ONNX inference via @huggingface/transformers
 * - `'openai'` — OpenAI Embeddings API
 * - `'cohere'` — Cohere Embed API
 * - `'google-vertex'` — Google Vertex AI Predict API
 * - `'voyage'` — Voyage AI (OpenAI-compatible wrapper)
 * - `'mistral'` — Mistral AI (OpenAI-compatible wrapper)
 * - `'jina'` — Jina AI (OpenAI-compatible wrapper)
 * - `'openrouter'` — OpenRouter (OpenAI-compatible wrapper)
 * - `'together'` — Together AI (OpenAI-compatible wrapper)
 * - `'fireworks'` — Fireworks AI (OpenAI-compatible wrapper)
 * - `'nomic'` — Nomic AI (OpenAI-compatible wrapper)
 * - `'mixedbread'` — Mixedbread AI (OpenAI-compatible wrapper)
 */
export type ProviderType =
  | 'local'
  | 'openai'
  | 'cohere'
  | 'google-vertex'
  | 'voyage'
  | 'mistral'
  | 'jina'
  | 'openrouter'
  | 'together'
  | 'fireworks'
  | 'nomic'
  | 'mixedbread';

// ─────────────────────────────────────────────────────────────────────────────
// Interfaces
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Options passed to {@link EmbeddingProvider.embed} to control embedding
 * generation behavior.
 */
export interface EmbedOptions {
  /**
   * Hint for asymmetric embedding models (e.g., Cohere, BGE).
   * - `'document'` — Optimized for indexing/storage. Use when embedding
   *   content that will be searched against.
   * - `'query'` — Optimized for search queries. Use when embedding the
   *   user's search input.
   *
   * Models that don't support asymmetric embeddings ignore this field.
   */
  inputType?: 'document' | 'query';

  /**
   * Truncate output embeddings to this many dimensions (Matryoshka-style).
   * Only effective with models that support ordered dimension importance
   * (e.g., OpenAI text-embedding-3-*). Reduces storage and speeds up search
   * at the cost of some precision.
   */
  dimensions?: number;

  /**
   * Override the provider's default batch size. Inputs are split into batches
   * of this size and processed sequentially.
   */
  batchSize?: number;

  /**
   * AbortSignal for cancelling long-running embedding requests. The provider
   * will throw an 'Aborted' error if the signal fires mid-request.
   */
  signal?: AbortSignal;
}

/**
 * The result returned by all embedding providers.
 *
 * Contains the generated embedding vectors along with metadata about the
 * model, dimensions, and token usage.
 */
export interface EmbeddingResult {
  /** Array of embedding vectors. Each vector is a Float32Array with `dimensions` elements. */
  embeddings: Float32Array[];
  /** Model identifier that was used for generation (e.g., 'text-embedding-3-small'). */
  model: string;
  /** Number of dimensions in each embedding vector. */
  dimensions: number;
  /** Token usage information, if the provider tracks it. */
  usage?: { tokens: number };
}

/**
 * Common interface implemented by all embedding providers (local, OpenAI,
 * Cohere, Google Vertex, etc.).
 *
 * All providers accept the same input shape and return the same output shape,
 * making it easy to swap providers without changing application code. Use the
 * {@link createProvider} factory or individual `create*Provider` functions.
 */
export interface EmbeddingProvider {
  /**
   * Generates embeddings for one or more text inputs.
   *
   * @param input - A single text string or array of text strings to embed.
   * @param options - Optional parameters for input type, dimensions, batching,
   *                  and cancellation.
   * @returns Promise resolving to an EmbeddingResult with the generated vectors.
   */
  embed(input: string | string[], options?: EmbedOptions): Promise<EmbeddingResult>;

  /** Human-readable provider name (e.g., 'local', 'cohere', 'openai-compatible'). */
  readonly name: string;

  /**
   * Known output dimensions, or `null` if dimensions are determined at runtime.
   * Local providers return `null` because dimensions depend on the loaded model.
   */
  readonly dimensions: number | null;
}

/**
 * A single result from a search operation (topK, aboveThreshold, rankBySimilarity).
 *
 * Contains the matched embedding along with its position in the original corpus,
 * similarity score, and optional label for human identification.
 */
export interface SearchResult {
  /** Zero-based index of this embedding in the original corpus array. */
  index: number;
  /** Similarity score computed using the specified metric. Higher = more similar. */
  score: number;
  /** The matched embedding vector. */
  embedding: Float32Array;
  /** Optional label from the `labels` array passed to the search function. */
  label?: string;
}

/**
 * Options for search functions ({@link topK}, {@link aboveThreshold},
 * {@link similarityMatrix}, {@link rankBySimilarity}).
 *
 * All fields are optional. When omitted, defaults are:
 * - `metric`: 'cosine'
 * - `labels`: undefined
 * - `filter`: undefined
 */
export interface SearchOptions {
  /** Similarity metric to use. Default: 'cosine'. */
  metric?: SimilarityMetric;
  /** Labels to attach to results, must match corpus length. */
  labels?: string[];
  /** Filter function called for each corpus item. Return `true` to include. */
  filter?: (index: number, label?: string) => boolean;
}

/**
 * Represents a cluster of related embeddings produced by {@link clusterEmbeddings}.
 *
 * Each cluster has a centroid (mean vector), its member embeddings, a size count,
 * and a cohesion score measuring internal consistency.
 */
export interface Cluster {
  /** Mean vector of all cluster members. Updated after every assignment or merge. */
  centroid: Float32Array;
  /** All embedding vectors belonging to this cluster. */
  members: Float32Array[];
  /** Optional labels corresponding to each member (preserves order). */
  labels?: string[];
  /** Number of members in this cluster (always equals `members.length`). */
  size: number;
  /**
   * Internal cohesion score (0-1 for cosine metric). Higher values indicate
   * tighter, more homogeneous clusters. Computed as average pairwise similarity
   * between all members.
   */
  cohesion: number;
}

/**
 * Configuration for the {@link clusterEmbeddings} function.
 *
 * All fields are optional and have sensible defaults:
 * - `similarityThreshold`: 0.9
 * - `minClusterSize`: 5
 * - `maxClusters`: 5
 * - `metric`: 'cosine'
 *
 * Use {@link getPreset} for commonly used configurations.
 */
export interface ClusteringConfig {
  /**
   * Minimum similarity required to assign an embedding to an existing cluster.
   * Embeddings below this threshold start a new cluster (if maxClusters allows).
   * Range: 0-1. Higher = stricter, more clusters. Default: 0.9.
   */
  similarityThreshold?: number;

  /**
   * Clusters smaller than this are redistributed into the nearest valid cluster.
   * Prevents tiny, noisy clusters. Default: 5.
   */
  minClusterSize?: number;

  /**
   * Maximum number of clusters to return. If clustering produces more, the most
   * similar pairs are merged until this limit is reached. Default: 5.
   */
  maxClusters?: number;

  /** Similarity metric used for all distance computations. Default: 'cosine'. */
  metric?: SimilarityMetric;

  /**
   * Strategy for assigning embeddings to clusters during greedy assignment.
   * - `'centroid'` (default) — Compare embedding to cluster centroid.
   * - `'average-similarity'` — Compute average similarity of the embedding to
   *   all current members of each cluster and assign to the highest.
   */
  assignmentStrategy?: 'centroid' | 'average-similarity';

  /**
   * Whether to shuffle input embeddings before clustering for order-independent
   * results. When `true`, uses `shuffleSeed` as the PRNG seed (default: 42).
   */
  shuffle?: boolean;

  /**
   * Seed for the deterministic shuffle PRNG. Only used when `shuffle` is `true`.
   * Different seeds may produce different clustering results. Default: 42.
   */
  shuffleSeed?: number;
}

/**
 * Statistical summary of a single cluster, including similarity distribution
 * and outlier indices.
 */
export interface ClusterStats {
  /** Minimum member-to-centroid similarity. */
  minSimilarity: number;
  /** Maximum member-to-centroid similarity. */
  maxSimilarity: number;
  /** Mean member-to-centroid similarity. */
  meanSimilarity: number;
  /** Median member-to-centroid similarity. */
  medianSimilarity: number;
  /** Max distance from centroid (radius of the cluster). */
  radius: number;
  /** Indices of members with similarity < mean - 2*stddev. */
  outliers: number[];
}

/**
 * Configuration for the in-memory LRU cache created by {@link createLRUCache}.
 */
export interface CacheOptions {
  /**
   * Maximum number of entries in the cache. When exceeded, the least recently
   * used entry is evicted. Default: 1000.
   */
  maxSize?: number;

  /**
   * Time-to-live in milliseconds. Entries older than this are treated as expired
   * on access (lazy cleanup). Default: no expiration.
   */
  ttl?: number;

  /**
   * Custom hash function for cache keys. Receives the raw key string and should
   * return a hashed string. Useful for reducing memory when keys are very long.
   * Default: identity (no hashing).
   */
  hashFunction?: (key: string) => string;
}

/**
 * Callback invoked after each batch completes during batch processing.
 * @param completed - Number of batches completed so far
 * @param total - Total number of batches
 */
export type ProgressCallback = (completed: number, total: number) => void;

/**
 * Configuration for batch processing of embedding inputs.
 */
export interface BatchConfig {
  /**
   * Maximum number of batches to process concurrently. Default: 1 (sequential).
   * Set higher to enable concurrent batching via promise pool. Be aware of
   * provider rate limits when increasing this value.
   */
  maxConcurrency?: number;

  /**
   * Delay in milliseconds between sequential batches. Default: 0.
   * Only applies when `maxConcurrency` is 1 (sequential mode).
   * Useful for respecting rate limits.
   */
  delayBetweenBatches?: number;

  /**
   * Called after each batch completes with (completed, total) counts.
   */
  onProgress?: ProgressCallback;
}

/**
 * Statistics returned by {@link CacheProvider.getStats} for monitoring cache performance.
 */
export interface CacheStats {
  /** Number of successful cache retrievals. */
  hits: number;
  /** Number of cache misses (key not found or expired). */
  misses: number;
  /** Number of entries evicted due to capacity limits. */
  evictions: number;
  /** Hit rate as a ratio: hits / (hits + misses). 0 if no operations. */
  hitRate: number;
  /** Current number of entries in the cache. */
  size: number;
  /** Maximum capacity of the cache. */
  maxSize: number;
}

/**
 * Async cache interface for storing and retrieving embedding vectors.
 *
 * The default implementation is an in-memory LRU cache ({@link createLRUCache}),
 * but the async interface allows drop-in replacement with Redis, SQLite, or any
 * other backing store without changing application code.
 */
export interface CacheProvider {
  /** Retrieve cached embeddings by key. Returns undefined on cache miss or expiration. */
  get(key: string): Promise<Float32Array[] | undefined>;
  /** Store embeddings under a key. Evicts the oldest entry if the cache is full. */
  set(key: string, value: Float32Array[]): Promise<void>;
  /** Check if a key exists and has not expired. */
  has(key: string): Promise<boolean>;
  /** Remove a specific entry from the cache. */
  delete(key: string): Promise<void>;
  /** Remove all entries from the cache. */
  clear(): Promise<void>;
  /** Returns cache performance statistics. */
  getStats(): CacheStats;
}

/**
 * Configuration for the exponential backoff retry logic used by cloud providers.
 *
 * Retries are attempted only for transient errors: HTTP 429 (rate limit) and
 * 5xx (server errors). All other errors fail immediately.
 */
export interface RetryConfig {
  /** Maximum number of retry attempts after the initial failure. Default: 3. */
  maxRetries?: number;
  /** Base delay in milliseconds (doubles each attempt). Default: 1000. */
  baseDelay?: number;
  /** Maximum delay cap in milliseconds. Default: 30000. */
  maxDelay?: number;
}

/**
 * Configuration for the local ONNX embedding provider ({@link createLocalProvider}).
 *
 * The local provider runs inference on the CPU using @huggingface/transformers
 * with ONNX runtime. No API key is needed.
 */
export interface LocalProviderConfig {
  /** HuggingFace model identifier. Default: 'Xenova/all-MiniLM-L12-v2'. */
  model?: string;
  /** Model precision: 'fp32' (default), 'fp16', or 'q8' (int8 quantized). */
  precision?: 'fp32' | 'fp16' | 'q8';
  /** Custom path to a local model directory. */
  modelPath?: string;
  /** Directory for caching downloaded models. Default: ~/.cache/huggingface/hub */
  cacheDir?: string;
  /** Whether to allow downloading models from HuggingFace Hub. Default: true. */
  allowRemoteModels?: boolean;
  /**
   * LRU cache configuration for the local provider. Controls the size and TTL
   * of the internal embedding cache. Default: maxSize 1000, no TTL.
   */
  cache?: CacheOptions;
  /**
   * Text prefix prepended to inputs when `inputType` is 'document'.
   * Some models (e.g., nomic-embed, BGE) use prefixes to distinguish
   * between document and query embeddings for asymmetric search.
   */
  documentPrefix?: string;
  /**
   * Text prefix prepended to inputs when `inputType` is 'query'.
   * Example: 'Represent this sentence for searching relevant passages: '
   */
  queryPrefix?: string;
}

/**
 * Configuration for the OpenAI-compatible embedding provider.
 *
 * Works with OpenAI, Voyage AI, Mistral, Jina, OpenRouter, and any endpoint
 * that follows the OpenAI `/v1/embeddings` API format (including Ollama,
 * LM Studio, and Azure OpenAI with a custom baseUrl).
 */
export interface OpenAICompatibleConfig {
  /** API key for authentication (sent as Bearer token). */
  apiKey: string;
  /** Base URL for the API. Default: 'https://api.openai.com/v1'. */
  baseUrl?: string;
  /** Model name (e.g., 'text-embedding-3-small', 'voyage-3'). */
  model: string;
  /** Optional output dimension override for models supporting Matryoshka truncation. */
  dimensions?: number;
  /** Maximum inputs per API call. Default: 2048. Inputs exceeding this are auto-batched. */
  maxBatchSize?: number;
  /** Retry configuration for transient errors. */
  retry?: RetryConfig;
  /** Request timeout in milliseconds. Default: 30000 (30 seconds). */
  timeout?: number;
}

/**
 * Configuration for the Cohere embedding provider.
 *
 * Calls the Cohere v2 `/embed` API. Supports input type mapping
 * (document vs. query) for optimal asymmetric search performance.
 */
export interface CohereConfig {
  /** Cohere API key (typically starts with 'co-'). */
  apiKey: string;
  /** Model name. Default: 'embed-v4.0'. */
  model?: string;
  /** Optional output dimension override. */
  dimensions?: number;
  /** Retry configuration for transient errors. */
  retry?: RetryConfig;
  /** Request timeout in milliseconds. Default: 30000 (30 seconds). */
  timeout?: number;
}

/**
 * Configuration for the Google Vertex AI embedding provider.
 *
 * Calls the Vertex AI Predict API. The access token can be a static string
 * or an async function that refreshes the token on each request.
 */
export interface GoogleVertexConfig {
  /** Google Cloud project ID (e.g., 'my-project-123'). */
  projectId: string;
  /** GCP region. Default: 'us-central1'. */
  location?: string;
  /** Vertex AI model name. Default: 'text-embedding-005'. */
  model?: string;
  /**
   * Google Cloud access token for authentication. Pass a string for static
   * tokens, or an async function for automatic token refresh (e.g., from
   * Application Default Credentials).
   */
  accessToken: string | (() => Promise<string>);
  /** Retry configuration for transient errors. */
  retry?: RetryConfig;
  /** Request timeout in milliseconds. Default: 30000 (30 seconds). */
  timeout?: number;
}

/**
 * Maps each {@link ProviderType} to its corresponding configuration interface.
 *
 * Used by {@link createProvider} to enforce type-safe config for each provider type.
 * Alias providers (voyage, mistral, jina, openrouter) use {@link OpenAICompatibleConfig}
 * since they are OpenAI-compatible API wrappers with different base URLs.
 */
export interface ProviderConfigMap {
  local: LocalProviderConfig;
  openai: OpenAICompatibleConfig;
  cohere: CohereConfig;
  'google-vertex': GoogleVertexConfig;
  voyage: OpenAICompatibleConfig;
  mistral: OpenAICompatibleConfig;
  jina: OpenAICompatibleConfig;
  openrouter: OpenAICompatibleConfig;
  together: OpenAICompatibleConfig;
  fireworks: OpenAICompatibleConfig;
  nomic: OpenAICompatibleConfig;
  mixedbread: OpenAICompatibleConfig;
}

/**
 * Options for Maximal Marginal Relevance (MMR) search.
 *
 * Extends {@link SearchOptions} with MMR-specific parameters for controlling
 * the relevance-diversity tradeoff.
 */
export interface MMROptions extends SearchOptions {
  /**
   * Tradeoff between relevance and diversity. Range 0-1.
   * - `1.0` = pure relevance (equivalent to topK)
   * - `0.0` = pure diversity (maximize difference from already selected)
   * Default: 0.5.
   */
  lambda?: number;

  /**
   * Number of top candidates to consider before applying MMR.
   * Limits the candidate pool to improve performance on large corpora.
   * Default: k * 4.
   */
  fetchK?: number;
}

/**
 * An item stored in a {@link SearchIndex}.
 */
export interface StoredItem {
  /** Unique identifier for this item. */
  id: string;
  /** The embedding vector. */
  embedding: Float32Array;
  /** Optional metadata associated with this item. */
  metadata?: Record<string, unknown>;
}

/**
 * Metadata stored alongside serialized embeddings in binary v2 format.
 *
 * All fields are optional. When provided during {@link serialize}, the metadata
 * is encoded as JSON in the binary header and returned by {@link deserialize}.
 */
export interface SerializationMetadata {
  /** Model identifier used to generate the embeddings. */
  model?: string;
  /** Number of dimensions in each embedding vector. */
  dimensions?: number;
  /** Unix timestamp (ms) when the embeddings were created. */
  timestamp?: number;
  /** Provider that generated the embeddings (e.g., 'openai', 'cohere'). */
  provider?: string;
}

/**
 * Metadata about an embedding model from the built-in registry.
 *
 * Used by model management functions ({@link getModelInfo}, {@link listModels})
 * and stored in {@link MODEL_REGISTRY}.
 */
/**
 * Configuration for {@link createEmbeddingStore}.
 */
export interface EmbeddingStoreConfig {
  /** The embedding provider to use for generating embeddings. */
  provider: EmbeddingProvider;
  /** Optional cache configuration. When provided, wraps the provider with an LRU cache. */
  cache?: CacheOptions;
  /** Similarity metric for search operations. Default: 'cosine'. */
  metric?: SimilarityMetric;
}

/**
 * Information about a model's tokenizer, returned by {@link getTokenizerInfo}.
 */
export interface TokenizerInfo {
  /** Maximum input token count supported by the model. */
  maxTokens: number;
  /** Model identifier from the registry. */
  modelId: string;
}

export interface ModelInfo {
  /** Model identifier (e.g., 'Xenova/all-MiniLM-L12-v2'). */
  id: string;
  /** Output embedding dimensions (e.g., 384, 768). */
  dimensions: number;
  /** Maximum input token count supported by the model. */
  maxTokens: number;
  /** Human-readable description of the model. */
  description: string;
  /** Approximate model size (e.g., '33M', '109M'). */
  size?: string;
  /**
   * Text prefixes for asymmetric embedding models. When set, the local
   * provider automatically prepends these based on the `inputType` option.
   */
  prefixes?: {
    /** Prefix for document/indexing inputs (often empty string). */
    document: string;
    /** Prefix for query/search inputs. */
    query: string;
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Search Fusion Types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * An item in a ranked list for use with {@link fuseRankedLists}.
 */
export interface RankedItem {
  /** Unique identifier for this item. */
  id: string;
  /** Score from the original ranking source. */
  score: number;
}

/**
 * Supported score normalization methods for {@link normalizeScores}.
 *
 * - `'min-max'` — Scales scores to [0, 1] range
 * - `'z-score'` — Standardizes to mean=0, std=1
 * - `'sigmoid'` — Maps to (0, 1) via logistic function
 */
export type NormalizationMethod = 'min-max' | 'z-score' | 'sigmoid';

// ─────────────────────────────────────────────────────────────────────────────
// Text / Markdown Chunking Types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Block type classification for markdown-aware chunking.
 */
export type StructuredChunkType = 'paragraph' | 'code' | 'list' | 'table' | 'heading';

// ─────────────────────────────────────────────────────────────────────────────
// HDBSCAN Clustering Types (re-exported from clustering/hdbscan.ts)
// ─────────────────────────────────────────────────────────────────────────────

export type { HDBSCANOptions, HDBSCANResult } from './clustering/hdbscan';
export type { HNSWOptions, HNSWSearchOptions } from './search/hnsw';
export type { RandomProjector } from './math/projection';
export type { QuantizationCalibration } from './quantization/calibration';
export type { PipelineOptions, PipelineProgressInfo, EmbeddingPipeline } from './pipeline/pipeline';
export type { CheckpointAdapter, CheckpointState } from './pipeline/checkpoint';

/**
 * A chunk produced by {@link chunkByStructure} with structural metadata.
 *
 * Each chunk preserves its position in the document, the heading hierarchy
 * it falls under, and the type of markdown structure it represents.
 */
export interface StructuredChunk {
  /** The chunk text content. */
  text: string;
  /** Metadata describing the chunk's position and structure. */
  metadata: {
    /** Heading hierarchy this chunk falls under (e.g., ['Intro', 'Setup']). */
    headings: string[];
    /** Character offset of this chunk in the original input text. */
    offset: number;
    /** The type of markdown structure this chunk represents. */
    type: StructuredChunkType;
  };
}

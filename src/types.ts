// ── Error Classes ──

export class EmbeddingError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EmbeddingError';
  }
}

export class ValidationError extends EmbeddingError {
  constructor(message: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

export class ProviderError extends EmbeddingError {
  status?: number;
  provider: string;

  constructor(message: string, provider: string, status?: number) {
    super(message);
    this.name = 'ProviderError';
    this.provider = provider;
    this.status = status;
  }
}

export class ModelNotFoundError extends EmbeddingError {
  constructor(message: string) {
    super(message);
    this.name = 'ModelNotFoundError';
  }
}

// ── Types ──

export type SimilarityMetric = 'cosine' | 'dot' | 'euclidean' | 'manhattan';

export type ProviderType =
  | 'local'
  | 'openai'
  | 'cohere'
  | 'google-vertex'
  | 'voyage'
  | 'mistral'
  | 'jina'
  | 'openrouter';

// ── Interfaces ──

export interface EmbedOptions {
  inputType?: 'document' | 'query';
  dimensions?: number;
  batchSize?: number;
  signal?: AbortSignal;
}

export interface EmbeddingResult {
  embeddings: number[][];
  model: string;
  dimensions: number;
  usage?: { tokens: number };
}

export interface EmbeddingProvider {
  embed(input: string | string[], options?: EmbedOptions): Promise<EmbeddingResult>;
  readonly name: string;
  readonly dimensions: number | null;
}

export interface SearchResult {
  index: number;
  score: number;
  embedding: number[];
  label?: string;
}

export interface Cluster {
  centroid: number[];
  members: number[][];
  labels?: string[];
  size: number;
  cohesion: number;
}

export interface ClusteringConfig {
  similarityThreshold?: number;
  minClusterSize?: number;
  maxClusters?: number;
  metric?: SimilarityMetric;
}

export interface CacheOptions {
  maxSize?: number;
  ttl?: number;
}

export interface CacheProvider {
  get(key: string): Promise<number[][] | undefined>;
  set(key: string, value: number[][]): Promise<void>;
  has(key: string): Promise<boolean>;
  delete(key: string): Promise<void>;
  clear(): Promise<void>;
}

export interface RetryConfig {
  maxRetries?: number;
  baseDelay?: number;
  maxDelay?: number;
}

export interface LocalProviderConfig {
  model?: string;
  precision?: 'fp32' | 'fp16' | 'q8';
  modelPath?: string;
  cacheDir?: string;
  allowRemoteModels?: boolean;
  documentPrefix?: string;
  queryPrefix?: string;
}

export interface OpenAICompatibleConfig {
  apiKey: string;
  baseUrl?: string;
  model: string;
  dimensions?: number;
  maxBatchSize?: number;
  retry?: RetryConfig;
}

export interface CohereConfig {
  apiKey: string;
  model?: string;
  dimensions?: number;
  retry?: RetryConfig;
}

export interface GoogleVertexConfig {
  projectId: string;
  location?: string;
  model?: string;
  accessToken: string | (() => Promise<string>);
  retry?: RetryConfig;
}

export interface ModelInfo {
  id: string;
  dimensions: number;
  description: string;
  size?: string;
  prefixes?: {
    document: string;
    query: string;
  };
}

export type {
  Vector,
  EmbeddingProvider,
  EmbedOptions,
  EmbeddingResult,
  SimilarityMetric,
  SearchResult,
  SearchOptions,
  MMROptions,
  StoredItem,
  Cluster,
  ClusteringConfig,
  ClusterStats,
  CacheOptions,
  CacheProvider,
  CacheStats,
  BatchConfig,
  ProgressCallback,
  RetryConfig,
  LocalProviderConfig,
  OpenAICompatibleConfig,
  CohereConfig,
  GoogleVertexConfig,
  ModelInfo,
  ProviderType,
  ProviderConfigMap,
  SerializationMetadata,
  EmbeddingStoreConfig,
  TokenizerInfo,
  RankedItem,
  NormalizationMethod,
} from './types';

export {
  EmbeddingUtilsError,
  EmbeddingError,
  ValidationError,
  DimensionMismatchError,
  ProviderError,
  ModelNotFoundError,
} from './types';

export { toFloat32, isVector } from './internal/vector-utils';

export {
  cosineSimilarity,
  dotProduct,
  euclideanDistance,
  manhattanDistance,
  cosineDistance,
  normalize,
  magnitude,
  add,
  subtract,
  scale,
  isNormalized,
  truncateDimensions,
  validateDimensions,
} from './math/index';

export {
  averageEmbeddings,
  weightedAverage,
  incrementalAverage,
  batchIncrementalAverage,
  centroid,
  maxPooling,
  minPooling,
  combineEmbeddings,
} from './aggregation/index';

export {
  topK,
  topKMulti,
  aboveThreshold,
  deduplicate,
  rankBySimilarity,
  similarityMatrix,
  pairwiseSimilarity,
  rerankResults,
  mmrSearch,
  SearchIndex,
  fuseRankedLists,
  normalizeScores,
} from './search/index';

export { serialize, deserialize } from './storage/index';
export type { DeserializeResult } from './storage/index';
export { createLRUCache, warmCache } from './storage/index';

export { quantize, dequantize, getQuantizationInfo, estimateMemorySavings, hammingDistance, hammingSimilarity } from './quantization/index';

export {
  clusterEmbeddings,
  CLUSTERING_PRESETS,
  getPreset,
  cohesionScore,
  centroidCohesion,
  silhouetteScore,
  assignToCluster,
  mergeClusters,
  findOptimalK,
  silhouetteByK,
  clusterStats,
  detectOutliers,
  centroidDrift,
  IncrementalClusterer,
} from './clustering/index';

export {
  createLocalProvider,
  createOpenAICompatibleProvider,
  createCohereProvider,
  createGoogleVertexProvider,
  createProvider,
  retryWithBackoff,
  autoBatch,
  withCache,
} from './providers/index';

export {
  downloadModel,
  listModels,
  deleteModel,
  setModelPath,
  getModelInfo,
  MODEL_REGISTRY,
  registerModel,
  getRecommendedModel,
} from './models/index';

export { chunkByTokenCount, chunkBySentence } from './text/index';
export { getTokenizerInfo } from './text/index';
export { createEmbeddingStore } from './store/index';
export type { EmbeddingStore } from './store/embedding-store';

// ─────────────────────────────────────────────────────────────────────────────
// Namespace exports
//
// Provide grouped access to related functions alongside the flat exports above.
// Flat exports remain for backward compatibility.
// ─────────────────────────────────────────────────────────────────────────────

import {
  cosineSimilarity as _cosineSimilarity,
  dotProduct as _dotProduct,
  euclideanDistance as _euclideanDistance,
  manhattanDistance as _manhattanDistance,
  cosineDistance as _cosineDistance,
  normalize as _normalize,
  magnitude as _magnitude,
  add as _add,
  subtract as _subtract,
  scale as _scale,
  isNormalized as _isNormalized,
  truncateDimensions as _truncateDimensions,
  validateDimensions as _validateDimensions,
} from './math/index';

export const Math = {
  cosineSimilarity: _cosineSimilarity,
  dotProduct: _dotProduct,
  euclideanDistance: _euclideanDistance,
  manhattanDistance: _manhattanDistance,
  cosineDistance: _cosineDistance,
  normalize: _normalize,
  magnitude: _magnitude,
  add: _add,
  subtract: _subtract,
  scale: _scale,
  isNormalized: _isNormalized,
  truncateDimensions: _truncateDimensions,
  validateDimensions: _validateDimensions,
} as const;

import {
  topK as _topK,
  aboveThreshold as _aboveThreshold,
  rankBySimilarity as _rankBySimilarity,
  similarityMatrix as _similarityMatrix,
  pairwiseSimilarity as _pairwiseSimilarity,
  rerankResults as _rerankResults,
  mmrSearch as _mmrSearch,
  SearchIndex as _SearchIndex,
  fuseRankedLists as _fuseRankedLists,
  normalizeScores as _normalizeScores,
} from './search/index';

export const Search = {
  topK: _topK,
  aboveThreshold: _aboveThreshold,
  rankBySimilarity: _rankBySimilarity,
  similarityMatrix: _similarityMatrix,
  pairwiseSimilarity: _pairwiseSimilarity,
  rerankResults: _rerankResults,
  mmrSearch: _mmrSearch,
  SearchIndex: _SearchIndex,
  fuseRankedLists: _fuseRankedLists,
  normalizeScores: _normalizeScores,
} as const;

import {
  clusterEmbeddings as _clusterEmbeddings,
  findOptimalK as _findOptimalK,
  silhouetteByK as _silhouetteByK,
  clusterStats as _clusterStats,
  detectOutliers as _detectOutliers,
  centroidDrift as _centroidDrift,
  IncrementalClusterer as _IncrementalClusterer,
  cohesionScore as _cohesionScore,
  centroidCohesion as _centroidCohesion,
  silhouetteScore as _silhouetteScore,
  assignToCluster as _assignToCluster,
  mergeClusters as _mergeClusters,
  CLUSTERING_PRESETS as _CLUSTERING_PRESETS,
  getPreset as _getPreset,
} from './clustering/index';

export const Clustering = {
  clusterEmbeddings: _clusterEmbeddings,
  findOptimalK: _findOptimalK,
  silhouetteByK: _silhouetteByK,
  clusterStats: _clusterStats,
  detectOutliers: _detectOutliers,
  centroidDrift: _centroidDrift,
  IncrementalClusterer: _IncrementalClusterer,
  cohesionScore: _cohesionScore,
  centroidCohesion: _centroidCohesion,
  silhouetteScore: _silhouetteScore,
  assignToCluster: _assignToCluster,
  mergeClusters: _mergeClusters,
  CLUSTERING_PRESETS: _CLUSTERING_PRESETS,
  getPreset: _getPreset,
} as const;

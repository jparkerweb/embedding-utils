export type {
  EmbeddingProvider,
  EmbedOptions,
  EmbeddingResult,
  SimilarityMetric,
  SearchResult,
  Cluster,
  ClusteringConfig,
  CacheOptions,
  CacheProvider,
  RetryConfig,
  LocalProviderConfig,
  OpenAICompatibleConfig,
  CohereConfig,
  GoogleVertexConfig,
  ModelInfo,
  ProviderType,
} from './types';

export {
  EmbeddingError,
  ValidationError,
  ProviderError,
  ModelNotFoundError,
} from './types';

export {
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
} from './math/index';

export {
  averageEmbeddings,
  weightedAverage,
  incrementalAverage,
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
} from './search/index';

export { serialize, deserialize } from './storage/index';
export { createLRUCache } from './storage/index';

export { quantize, dequantize, getQuantizationInfo } from './quantization/index';

export {
  clusterEmbeddings,
  CLUSTERING_PRESETS,
  getPreset,
  cohesionScore,
  silhouetteScore,
  assignToCluster,
  mergeClusters,
} from './clustering/index';

export {
  createLocalProvider,
  createOpenAICompatibleProvider,
  createCohereProvider,
  createGoogleVertexProvider,
  createProvider,
  retryWithBackoff,
  autoBatch,
} from './providers/index';

export {
  downloadModel,
  listModels,
  deleteModel,
  setModelPath,
  getModelInfo,
  MODEL_REGISTRY,
} from './models/index';

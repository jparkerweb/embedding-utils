import type { ModelInfo } from '../types';
import { ValidationError } from '../types';

export const MODEL_REGISTRY: Record<string, ModelInfo> = {
  'Xenova/all-MiniLM-L12-v2': {
    id: 'Xenova/all-MiniLM-L12-v2',
    dimensions: 384,
    maxTokens: 256,
    description: 'All-round English embedding model, 33M parameters',
    size: '33M',
  },
  'Xenova/all-MiniLM-L6-v2': {
    id: 'Xenova/all-MiniLM-L6-v2',
    dimensions: 384,
    maxTokens: 256,
    description: 'Lightweight English embedding model, 22M parameters',
    size: '22M',
  },
  'Xenova/bge-small-en-v1.5': {
    id: 'Xenova/bge-small-en-v1.5',
    dimensions: 384,
    maxTokens: 512,
    description: 'BGE small English embedding model',
    size: '33M',
    prefixes: {
      document: '',
      query: 'Represent this sentence for searching relevant passages: ',
    },
  },
  'Xenova/bge-base-en-v1.5': {
    id: 'Xenova/bge-base-en-v1.5',
    dimensions: 768,
    maxTokens: 512,
    description: 'BGE base English embedding model',
    size: '109M',
    prefixes: {
      document: '',
      query: 'Represent this sentence for searching relevant passages: ',
    },
  },
  'Xenova/all-mpnet-base-v2': {
    id: 'Xenova/all-mpnet-base-v2',
    dimensions: 768,
    maxTokens: 384,
    description: 'High-quality English embedding model, 109M parameters',
    size: '109M',
  },
  'Xenova/multilingual-MiniLM-L12-v2': {
    id: 'Xenova/multilingual-MiniLM-L12-v2',
    dimensions: 384,
    maxTokens: 512,
    description: 'Multilingual embedding model supporting 50+ languages',
    size: '33M',
  },
  'nomic-ai/nomic-embed-text-v1.5': {
    id: 'nomic-ai/nomic-embed-text-v1.5',
    dimensions: 768,
    maxTokens: 8192,
    description: 'Long-context embedding model with Matryoshka support',
    size: '137M',
    prefixes: {
      document: 'search_document: ',
      query: 'search_query: ',
    },
  },
  'Xenova/distilroberta-base': {
    id: 'Xenova/distilroberta-base',
    dimensions: 768,
    maxTokens: 512,
    description: 'Distilled RoBERTa base model for general text embeddings',
    size: '82M',
  },
};

/**
 * Registers a custom model in the runtime registry.
 *
 * @param info - Model metadata to register
 * @param options - Optional settings; set `overwrite: true` to replace an existing entry
 * @throws {ValidationError} If `info.id` is empty, `info.dimensions` is not a positive integer, or the model ID is already registered (without `overwrite: true`)
 * @example
 * registerModel({ id: 'my-org/custom-model', dimensions: 512, maxTokens: 1024, description: 'Custom model' });
 */
export function registerModel(
  info: ModelInfo,
  options?: { overwrite?: boolean },
): void {
  if (!info.id || typeof info.id !== 'string') {
    throw new ValidationError('Model id must be a non-empty string');
  }
  if (!Number.isInteger(info.dimensions) || info.dimensions <= 0) {
    throw new ValidationError('Model dimensions must be a positive integer');
  }
  if (MODEL_REGISTRY[info.id] && !options?.overwrite) {
    throw new ValidationError(
      `Model "${info.id}" is already registered. Pass { overwrite: true } to replace it.`,
    );
  }
  MODEL_REGISTRY[info.id] = info;
}

/**
 * Returns a recommended model for a given use case.
 *
 * @param useCase - One of 'speed', 'balanced', 'quality', or 'multilingual'
 * @returns The recommended ModelInfo
 * @example
 * getRecommendedModel('speed'); // MiniLM-L6-v2
 * getRecommendedModel('multilingual'); // multilingual-MiniLM-L12-v2
 */
export function getRecommendedModel(
  useCase: 'speed' | 'balanced' | 'quality' | 'multilingual',
): ModelInfo {
  switch (useCase) {
    case 'speed':
      return MODEL_REGISTRY['Xenova/all-MiniLM-L6-v2'];
    case 'balanced':
      return MODEL_REGISTRY['Xenova/all-mpnet-base-v2'];
    case 'quality':
      return MODEL_REGISTRY['Xenova/all-mpnet-base-v2'];
    case 'multilingual':
      return MODEL_REGISTRY['Xenova/multilingual-MiniLM-L12-v2'];
  }
}

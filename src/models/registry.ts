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
    pooling: 'cls',
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
    pooling: 'cls',
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

  // ── GTE (Alibaba) — mean pooling, no prefixes ───────────────────────────────
  'Xenova/gte-small': {
    id: 'Xenova/gte-small',
    dimensions: 384,
    maxTokens: 512,
    description: 'GTE small English embedding model, strong quality for its size',
    size: '33M',
    pooling: 'mean',
  },
  'Xenova/gte-base': {
    id: 'Xenova/gte-base',
    dimensions: 768,
    maxTokens: 512,
    description: 'GTE base English embedding model',
    size: '109M',
    pooling: 'mean',
  },
  'Xenova/gte-large': {
    id: 'Xenova/gte-large',
    dimensions: 1024,
    maxTokens: 512,
    description: 'GTE large English embedding model, high quality',
    size: '335M',
    pooling: 'mean',
  },
  'onnx-community/gte-multilingual-base': {
    id: 'onnx-community/gte-multilingual-base',
    dimensions: 768,
    maxTokens: 8192,
    description: 'GTE multilingual base (70+ languages), long-context, Matryoshka-truncatable',
    size: '305M',
    pooling: 'cls',
  },

  // ── Jina v2 — long-context (8192 via ALiBi), mean pooling, no prefixes ───────
  'Xenova/jina-embeddings-v2-small-en': {
    id: 'Xenova/jina-embeddings-v2-small-en',
    dimensions: 512,
    maxTokens: 8192,
    description: 'Jina v2 small English embedding model, 8K context',
    size: '33M',
    pooling: 'mean',
  },
  'Xenova/jina-embeddings-v2-base-en': {
    id: 'Xenova/jina-embeddings-v2-base-en',
    dimensions: 768,
    maxTokens: 8192,
    description: 'Jina v2 base English embedding model, 8K context',
    size: '137M',
    pooling: 'mean',
  },

  // ── E5 (intfloat) — mean pooling, REQUIRE query:/passage: prefixes ───────────
  'Xenova/e5-small-v2': {
    id: 'Xenova/e5-small-v2',
    dimensions: 384,
    maxTokens: 512,
    description: 'E5 small v2 English embedding model',
    size: '33M',
    pooling: 'mean',
    prefixes: {
      document: 'passage: ',
      query: 'query: ',
    },
  },
  'Xenova/e5-base-v2': {
    id: 'Xenova/e5-base-v2',
    dimensions: 768,
    maxTokens: 512,
    description: 'E5 base v2 English embedding model',
    size: '109M',
    pooling: 'mean',
    prefixes: {
      document: 'passage: ',
      query: 'query: ',
    },
  },
  'Xenova/e5-large-v2': {
    id: 'Xenova/e5-large-v2',
    dimensions: 1024,
    maxTokens: 512,
    description: 'E5 large v2 English embedding model',
    size: '335M',
    pooling: 'mean',
    prefixes: {
      document: 'passage: ',
      query: 'query: ',
    },
  },

  // ── Multilingual ────────────────────────────────────────────────────────────
  'Xenova/paraphrase-multilingual-MiniLM-L12-v2': {
    id: 'Xenova/paraphrase-multilingual-MiniLM-L12-v2',
    dimensions: 384,
    maxTokens: 128,
    description: 'Paraphrase multilingual MiniLM (~50 languages)',
    size: '118M',
    pooling: 'mean',
  },
  'Xenova/multilingual-e5-small': {
    id: 'Xenova/multilingual-e5-small',
    dimensions: 384,
    maxTokens: 512,
    description: 'Multilingual E5 small (100+ languages); requires query:/passage: prefixes',
    size: '118M',
    pooling: 'mean',
    prefixes: {
      document: 'passage: ',
      query: 'query: ',
    },
  },
  'Xenova/multilingual-e5-base': {
    id: 'Xenova/multilingual-e5-base',
    dimensions: 768,
    maxTokens: 512,
    description: 'Multilingual E5 base (100+ languages); requires query:/passage: prefixes',
    size: '278M',
    pooling: 'mean',
    prefixes: {
      document: 'passage: ',
      query: 'query: ',
    },
  },
  'Xenova/multilingual-e5-large': {
    id: 'Xenova/multilingual-e5-large',
    dimensions: 1024,
    maxTokens: 512,
    description:
      'Multilingual E5 large (100+ languages); requires query:/passage: prefixes. Use precision q8/fp16 — fp32 exceeds 2GB',
    size: '560M',
    pooling: 'mean',
    prefixes: {
      document: 'passage: ',
      query: 'query: ',
    },
  },

  // ── BGE large + M3 ──────────────────────────────────────────────────────────
  'Xenova/bge-large-en-v1.5': {
    id: 'Xenova/bge-large-en-v1.5',
    dimensions: 1024,
    maxTokens: 512,
    description: 'BGE large English embedding model',
    size: '335M',
    pooling: 'cls',
    prefixes: {
      document: '',
      query: 'Represent this sentence for searching relevant passages: ',
    },
  },
  'Xenova/bge-m3': {
    id: 'Xenova/bge-m3',
    dimensions: 1024,
    maxTokens: 8192,
    description:
      'BGE-M3 multilingual long-context model (100+ languages). Use precision q8/fp16 — fp32 exceeds 2GB',
    size: '568M',
    pooling: 'cls',
  },

  // ── mxbai + Snowflake Arctic — CLS pooling ──────────────────────────────────
  'mixedbread-ai/mxbai-embed-large-v1': {
    id: 'mixedbread-ai/mxbai-embed-large-v1',
    dimensions: 1024,
    maxTokens: 512,
    description: 'mxbai-embed-large v1, top-tier English (Matryoshka-truncatable)',
    size: '335M',
    pooling: 'cls',
    prefixes: {
      document: '',
      query: 'Represent this sentence for searching relevant passages: ',
    },
  },
  'Snowflake/snowflake-arctic-embed-m-v1.5': {
    id: 'Snowflake/snowflake-arctic-embed-m-v1.5',
    dimensions: 768,
    maxTokens: 512,
    description: 'Snowflake Arctic Embed medium v1.5 (Matryoshka-truncatable to 256)',
    size: '109M',
    pooling: 'cls',
    prefixes: {
      document: '',
      query: 'Represent this sentence for searching relevant passages: ',
    },
  },
  'Snowflake/snowflake-arctic-embed-l': {
    id: 'Snowflake/snowflake-arctic-embed-l',
    dimensions: 1024,
    maxTokens: 512,
    description: 'Snowflake Arctic Embed large, strong retrieval quality',
    size: '335M',
    pooling: 'cls',
    prefixes: {
      document: '',
      query: 'Represent this sentence for searching relevant passages: ',
    },
  },

  // ── Qwen3 — last-token pooling, 32K context. fp32 > 2GB: use q8/fp16 ─────────
  'onnx-community/Qwen3-Embedding-0.6B-ONNX': {
    id: 'onnx-community/Qwen3-Embedding-0.6B-ONNX',
    dimensions: 1024,
    maxTokens: 32768,
    description:
      'Qwen3-Embedding 0.6B, instruction-aware, 32K context. Use precision q8/fp16 — fp32 exceeds 2GB',
    size: '595M',
    pooling: 'last_token',
    prefixes: {
      document: '',
      query:
        'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:',
    },
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
export function registerModel(info: ModelInfo, options?: { overwrite?: boolean }): void {
  if (!info.id || typeof info.id !== 'string') {
    throw new ValidationError('Model id must be a non-empty string');
  }
  if (!Number.isInteger(info.dimensions) || info.dimensions <= 0) {
    throw new ValidationError('Model dimensions must be a positive integer');
  }
  if (MODEL_REGISTRY[info.id] && !options?.overwrite) {
    throw new ValidationError(
      `Model "${info.id}" is already registered. Pass { overwrite: true } to replace it.`
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
  useCase: 'speed' | 'balanced' | 'quality' | 'multilingual'
): ModelInfo {
  switch (useCase) {
    case 'speed':
      return MODEL_REGISTRY['Xenova/all-MiniLM-L6-v2'];
    case 'balanced':
      return MODEL_REGISTRY['Xenova/all-mpnet-base-v2'];
    case 'quality':
      return MODEL_REGISTRY['Xenova/bge-large-en-v1.5'];
    case 'multilingual':
      return MODEL_REGISTRY['Xenova/multilingual-MiniLM-L12-v2'];
  }
}

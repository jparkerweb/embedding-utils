import type { ModelInfo } from '../types';

export const MODEL_REGISTRY: Record<string, ModelInfo> = {
  'Xenova/all-MiniLM-L12-v2': {
    id: 'Xenova/all-MiniLM-L12-v2',
    dimensions: 384,
    description: 'All-round English embedding model, 33M parameters',
    size: '33M',
  },
  'Xenova/all-MiniLM-L6-v2': {
    id: 'Xenova/all-MiniLM-L6-v2',
    dimensions: 384,
    description: 'Lightweight English embedding model, 22M parameters',
    size: '22M',
  },
  'Xenova/bge-small-en-v1.5': {
    id: 'Xenova/bge-small-en-v1.5',
    dimensions: 384,
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
    description: 'BGE base English embedding model',
    size: '109M',
    prefixes: {
      document: '',
      query: 'Represent this sentence for searching relevant passages: ',
    },
  },
};

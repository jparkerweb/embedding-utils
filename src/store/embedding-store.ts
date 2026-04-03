import type {
  EmbeddingStoreConfig,
  StoredItem,
} from '../types';
import { SearchIndex } from '../search/search-index';
import { withCache } from '../providers/middleware';

/**
 * High-level embedding store that composes a provider, optional cache, and
 * search index into a single add/search/remove API.
 *
 * This is intentionally a thin composition layer. Users who need fine-grained
 * control should use the lower-level APIs directly.
 */
export interface EmbeddingStore {
  /** Embed text and store the result with an ID and optional metadata. */
  add(id: string, text: string, metadata?: Record<string, unknown>): Promise<void>;
  /** Batch embed and store multiple items. */
  addBatch(items: Array<{ id: string; text: string; metadata?: Record<string, unknown> }>): Promise<void>;
  /** Embed query text and search for similar items. */
  search(
    query: string,
    options?: { topK?: number; threshold?: number; filter?: (item: StoredItem) => boolean },
  ): Promise<Array<StoredItem & { score: number }>>;
  /** Search by a pre-computed embedding vector. */
  searchByEmbedding(
    embedding: number[],
    options?: { topK?: number; threshold?: number; filter?: (item: StoredItem) => boolean },
  ): Array<StoredItem & { score: number }>;
  /** Remove an item by ID. */
  remove(id: string): boolean;
  /** Remove all items. */
  clear(): void;
  /** Number of items in the store. */
  readonly size: number;
}

/**
 * Creates a high-level embedding store that handles embedding generation,
 * caching, and search in a unified interface.
 *
 * @param config - Store configuration with provider, optional cache, and metric
 * @returns An EmbeddingStore instance
 *
 * @example
 * const store = createEmbeddingStore({
 *   provider: createProvider('openai', { apiKey: 'sk-...', model: 'text-embedding-3-small' }),
 *   cache: { maxSize: 500 },
 * });
 * await store.add('doc1', 'Hello world');
 * const results = await store.search('greeting', { topK: 5 });
 */
export function createEmbeddingStore(config: EmbeddingStoreConfig): EmbeddingStore {
  const provider = config.cache
    ? withCache(config.provider, config.cache)
    : config.provider;

  const index = new SearchIndex({ metric: config.metric });

  return {
    async add(id: string, text: string, metadata?: Record<string, unknown>): Promise<void> {
      const result = await provider.embed(text);
      index.add(id, result.embeddings[0], metadata);
    },

    async addBatch(
      items: Array<{ id: string; text: string; metadata?: Record<string, unknown> }>,
    ): Promise<void> {
      const texts = items.map((item) => item.text);
      const result = await provider.embed(texts);
      for (let i = 0; i < items.length; i++) {
        index.add(items[i].id, result.embeddings[i], items[i].metadata);
      }
    },

    async search(
      query: string,
      options?: { topK?: number; threshold?: number; filter?: (item: StoredItem) => boolean },
    ): Promise<Array<StoredItem & { score: number }>> {
      const result = await provider.embed(query);
      return index.search(result.embeddings[0], options);
    },

    searchByEmbedding(
      embedding: number[],
      options?: { topK?: number; threshold?: number; filter?: (item: StoredItem) => boolean },
    ): Array<StoredItem & { score: number }> {
      return index.search(embedding, options);
    },

    remove(id: string): boolean {
      return index.remove(id);
    },

    clear(): void {
      index.clear();
    },

    get size(): number {
      return index.size;
    },
  };
}

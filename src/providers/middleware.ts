import type { EmbeddingProvider, EmbeddingResult, EmbedOptions, CacheOptions } from '../types';
import { createLRUCache } from '../storage/cache';

/**
 * Wraps an embedding provider with an in-memory LRU cache.
 *
 * Cached entries are keyed by text content + model name + requested dimensions.
 * Embeddings are copied on store and on retrieval to prevent mutation of cached values.
 *
 * @param provider - The embedding provider to wrap
 * @param options - Cache configuration (maxSize, ttl, hashFunction)
 * @returns A new EmbeddingProvider that caches results from the wrapped provider
 * @example
 * const cached = withCache(createProvider('openai', config), { maxSize: 500 });
 * const r1 = await cached.embed('hello'); // calls provider
 * const r2 = await cached.embed('hello'); // returns from cache
 */
export function withCache(
  provider: EmbeddingProvider,
  options?: CacheOptions,
): EmbeddingProvider {
  const cache = createLRUCache({
    maxSize: options?.maxSize ?? 1000,
    ttl: options?.ttl,
  });
  const hashFn = options?.hashFunction;

  function buildKey(inputs: string[], dimensions?: number): string {
    const raw = provider.name + ':' + (dimensions ?? '') + ':' + JSON.stringify(inputs);
    return hashFn ? hashFn(raw) : raw;
  }

  function copyEmbeddings(embeddings: number[][]): number[][] {
    return embeddings.map((e) => e.slice());
  }

  return {
    name: provider.name,
    dimensions: provider.dimensions,

    async embed(
      input: string | string[],
      options?: EmbedOptions,
    ): Promise<EmbeddingResult> {
      const inputs = Array.isArray(input) ? input : [input];
      const key = buildKey(inputs, options?.dimensions);

      const cached = await cache.get(key);
      if (cached) {
        const embeddings = copyEmbeddings(cached);
        return {
          embeddings,
          model: provider.name,
          dimensions: embeddings[0]?.length ?? 0,
        };
      }

      const result = await provider.embed(input, options);

      // Store a copy to prevent external mutation of cached values
      await cache.set(key, copyEmbeddings(result.embeddings));

      return result;
    },
  };
}

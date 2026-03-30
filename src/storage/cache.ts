import type { CacheOptions, CacheProvider } from '../types';

interface CacheEntry {
  value: number[][];
  expiry: number | null;
}

/**
 * Creates an in-memory LRU cache for embedding vectors.
 * @param options - Optional configuration: maxSize (default 1000), ttl in ms (default none)
 * @returns A CacheProvider with get, set, has, delete, and clear methods
 * @example
 * const cache = createLRUCache({ maxSize: 500, ttl: 60000 });
 * await cache.set('key', [[1, 2, 3]]);
 * const result = await cache.get('key'); // [[1, 2, 3]]
 */
export function createLRUCache(options?: CacheOptions): CacheProvider {
  const maxSize = options?.maxSize ?? 1000;
  const ttl = options?.ttl ?? null;
  const map = new Map<string, CacheEntry>();

  function isExpired(entry: CacheEntry): boolean {
    return entry.expiry !== null && Date.now() > entry.expiry;
  }

  return {
    async get(key: string): Promise<number[][] | undefined> {
      const entry = map.get(key);
      if (!entry) return undefined;
      if (isExpired(entry)) {
        map.delete(key);
        return undefined;
      }
      // Move to end (most recent) by delete + re-insert
      map.delete(key);
      map.set(key, entry);
      return entry.value;
    },

    async set(key: string, value: number[][]): Promise<void> {
      // Delete if exists (to update insertion order)
      map.delete(key);
      const expiry = ttl !== null ? Date.now() + ttl : null;
      map.set(key, { value, expiry });

      // Evict least recently used (first key) if over capacity
      if (map.size > maxSize) {
        const firstKey = map.keys().next().value!;
        map.delete(firstKey);
      }
    },

    async has(key: string): Promise<boolean> {
      const entry = map.get(key);
      if (!entry) return false;
      if (isExpired(entry)) {
        map.delete(key);
        return false;
      }
      return true;
    },

    async delete(key: string): Promise<void> {
      map.delete(key);
    },

    async clear(): Promise<void> {
      map.clear();
    },
  };
}

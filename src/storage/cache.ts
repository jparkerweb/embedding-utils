import type { CacheOptions, CacheProvider, CacheStats } from '../types';

interface CacheEntry {
  value: number[][];
  expiry: number | null;
}

/**
 * Creates an in-memory LRU cache for embedding vectors.
 * @param options - Optional configuration: maxSize (default 1000), ttl in ms (default none)
 * @returns A CacheProvider with get, set, has, delete, clear, and getStats methods
 * @example
 * const cache = createLRUCache({ maxSize: 500, ttl: 60000 });
 * await cache.set('key', [[1, 2, 3]]);
 * const result = await cache.get('key'); // [[1, 2, 3]]
 */
export function createLRUCache(options?: CacheOptions): CacheProvider {
  const maxSize = options?.maxSize ?? 1000;
  const ttl = options?.ttl ?? null;
  const map = new Map<string, CacheEntry>();

  let hits = 0;
  let misses = 0;
  let evictions = 0;

  function isExpired(entry: CacheEntry): boolean {
    return entry.expiry !== null && Date.now() > entry.expiry;
  }

  return {
    async get(key: string): Promise<number[][] | undefined> {
      const entry = map.get(key);
      if (!entry) {
        misses++;
        return undefined;
      }
      if (isExpired(entry)) {
        map.delete(key);
        misses++;
        return undefined;
      }
      // Move to end (most recent) by delete + re-insert
      map.delete(key);
      map.set(key, entry);
      hits++;
      // Return a copy to prevent external mutation of cached values
      return entry.value.map((row) => row.slice());
    },

    async set(key: string, value: number[][]): Promise<void> {
      // Delete if exists (to update insertion order)
      map.delete(key);
      const expiry = ttl !== null ? Date.now() + ttl : null;
      // Store a copy to prevent external mutation of cached values
      const copied = value.map((row) => row.slice());
      map.set(key, { value: copied, expiry });

      // Evict least recently used (first key) if over capacity
      if (map.size > maxSize) {
        const firstKey = map.keys().next().value!;
        map.delete(firstKey);
        evictions++;
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

    getStats(): CacheStats {
      const total = hits + misses;
      return {
        hits,
        misses,
        evictions,
        hitRate: total === 0 ? 0 : hits / total,
        size: map.size,
        maxSize,
      };
    },
  };
}

/**
 * Populates a cache with pre-computed data.
 *
 * Iterates over data entries and stores each in the cache. If data exceeds
 * the cache's max size, LRU eviction naturally retains only the most recent entries.
 *
 * @param cache - A CacheProvider to populate (typically from createLRUCache)
 * @param data - Array of key-value pairs to insert into the cache
 * @example
 * const cache = createLRUCache({ maxSize: 100 });
 * await warmCache(cache, [
 *   { key: 'greeting', value: [[0.1, 0.2, 0.3]] },
 *   { key: 'farewell', value: [[0.4, 0.5, 0.6]] },
 * ]);
 */
export async function warmCache(
  cache: CacheProvider,
  data: Array<{ key: string; value: number[][] }>,
): Promise<void> {
  for (const entry of data) {
    await cache.set(entry.key, entry.value);
  }
}

import type { CacheOptions, CacheProvider, CacheStats } from '../types';

interface CacheEntry {
  value: Float32Array[];
  expiry: number | null;
}

/**
 * Creates an in-memory LRU cache for embedding vectors.
 * @param options - Optional configuration: maxSize (default 1000), ttl in ms (default none)
 * @returns A CacheProvider with get, set, has, delete, clear, and getStats methods
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
    async get(key: string): Promise<Float32Array[] | undefined> {
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
      return entry.value.map((row) => new Float32Array(row));
    },

    async set(key: string, value: Float32Array[]): Promise<void> {
      // Delete if exists (to update insertion order)
      map.delete(key);
      const expiry = ttl !== null ? Date.now() + ttl : null;
      // Store a copy to prevent external mutation of cached values
      const copied = value.map((row) => new Float32Array(row));
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
 * @param cache - A CacheProvider to populate (typically from createLRUCache)
 * @param data - Array of key-value pairs to insert into the cache
 */
export async function warmCache(
  cache: CacheProvider,
  data: Array<{ key: string; value: Float32Array[] }>,
): Promise<void> {
  for (const entry of data) {
    await cache.set(entry.key, entry.value);
  }
}

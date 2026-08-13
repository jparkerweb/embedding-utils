/**
 * @internal
 * Bounded keyed registry for expensive async-constructed instances (ONNX
 * pipelines, tokenizers). Ensures one construction per config key so callers
 * that create many providers/tokenizers with the same config share the
 * underlying native resource instead of paying the multi-second construction
 * cost (and native memory) repeatedly.
 *
 * Eviction semantics: when the bound is exceeded, the least-recently-used
 * entry is dropped from the registry WITHOUT being disposed — existing holders
 * keep a working reference and the resource is reclaimed when they release it.
 * Auto-disposing on eviction would create use-after-free bugs for any caller
 * still holding the instance. Explicit cleanup goes through {@link clear},
 * which hands the settled instances back to the caller for disposal.
 */

export interface InstanceRegistry<T> {
  /**
   * Returns the instance promise for `key`, constructing it via `factory`
   * only when absent. A rejected construction is evicted so the next call
   * can retry. Touching a key marks it most-recently-used.
   */
  getOrCreate(key: string, factory: () => Promise<T>): Promise<T>;
  /**
   * Empties the registry and resolves with every instance that had settled
   * successfully, so the caller can dispose native resources. Pending
   * constructions are dropped from the registry but not returned (their
   * callers still hold the promise).
   */
  clear(): Promise<T[]>;
  /** Number of entries currently registered (including pending). */
  readonly size: number;
}

export function createInstanceRegistry<T>(maxInstances: number): InstanceRegistry<T> {
  // Map preserves insertion order; delete + re-insert marks recency (same
  // idiom as storage/cache.ts).
  const map = new Map<string, Promise<T>>();

  return {
    getOrCreate(key: string, factory: () => Promise<T>): Promise<T> {
      const existing = map.get(key);
      if (existing) {
        map.delete(key);
        map.set(key, existing);
        return existing;
      }

      const promise = factory().catch((error: unknown) => {
        // Evict failed constructions (only if still ours) so callers can retry.
        if (map.get(key) === promise) {
          map.delete(key);
        }
        throw error;
      });
      map.set(key, promise);

      // Evict least-recently-used beyond the bound. Reference is dropped, not
      // disposed — see module doc.
      while (map.size > maxInstances) {
        const oldestKey = map.keys().next().value as string;
        map.delete(oldestKey);
      }

      return promise;
    },

    async clear(): Promise<T[]> {
      const pending = [...map.values()];
      map.clear();
      const settled = await Promise.allSettled(pending);
      const instances: T[] = [];
      for (const result of settled) {
        if (result.status === 'fulfilled') {
          instances.push(result.value as T);
        }
      }
      return instances;
    },

    get size(): number {
      return map.size;
    },
  };
}

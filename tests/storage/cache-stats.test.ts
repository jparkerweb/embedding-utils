import { describe, it, expect } from 'vitest';
import { createLRUCache, warmCache } from '../../src/storage/cache';

describe('CacheStats', () => {
  it('starts with zero stats', () => {
    const cache = createLRUCache({ maxSize: 10 });
    const stats = cache.getStats();
    expect(stats.hits).toBe(0);
    expect(stats.misses).toBe(0);
    expect(stats.evictions).toBe(0);
    expect(stats.hitRate).toBe(0);
    expect(stats.size).toBe(0);
    expect(stats.maxSize).toBe(10);
  });

  it('tracks hits', async () => {
    const cache = createLRUCache({ maxSize: 10 });
    await cache.set('a', [[1, 2, 3]]);
    await cache.get('a');
    await cache.get('a');

    const stats = cache.getStats();
    expect(stats.hits).toBe(2);
    expect(stats.misses).toBe(0);
  });

  it('tracks misses', async () => {
    const cache = createLRUCache({ maxSize: 10 });
    await cache.get('nonexistent');
    await cache.get('nope');

    const stats = cache.getStats();
    expect(stats.hits).toBe(0);
    expect(stats.misses).toBe(2);
  });

  it('tracks evictions', async () => {
    const cache = createLRUCache({ maxSize: 2 });
    await cache.set('a', [[1]]);
    await cache.set('b', [[2]]);
    await cache.set('c', [[3]]); // evicts 'a'

    const stats = cache.getStats();
    expect(stats.evictions).toBe(1);
    expect(stats.size).toBe(2);
  });

  it('computes hitRate correctly', async () => {
    const cache = createLRUCache({ maxSize: 10 });
    await cache.set('a', [[1]]);
    await cache.get('a'); // hit
    await cache.get('b'); // miss
    await cache.get('a'); // hit
    await cache.get('c'); // miss

    const stats = cache.getStats();
    expect(stats.hits).toBe(2);
    expect(stats.misses).toBe(2);
    expect(stats.hitRate).toBe(0.5);
  });

  it('hitRate is 0 when no operations', () => {
    const cache = createLRUCache({ maxSize: 10 });
    expect(cache.getStats().hitRate).toBe(0);
  });

  it('tracks size accurately', async () => {
    const cache = createLRUCache({ maxSize: 10 });
    await cache.set('a', [[1]]);
    await cache.set('b', [[2]]);
    expect(cache.getStats().size).toBe(2);

    await cache.delete('a');
    expect(cache.getStats().size).toBe(1);

    await cache.clear();
    expect(cache.getStats().size).toBe(0);
  });
});

describe('warmCache', () => {
  it('populates cache from data array', async () => {
    const cache = createLRUCache({ maxSize: 10 });
    await warmCache(cache, [
      { key: 'a', value: [[1, 2, 3]] },
      { key: 'b', value: [[4, 5, 6]] },
    ]);

    const a = await cache.get('a');
    expect(a).toEqual([[1, 2, 3]]);
    const b = await cache.get('b');
    expect(b).toEqual([[4, 5, 6]]);
  });

  it('respects cache maxSize (LRU eviction)', async () => {
    const cache = createLRUCache({ maxSize: 2 });
    await warmCache(cache, [
      { key: 'a', value: [[1]] },
      { key: 'b', value: [[2]] },
      { key: 'c', value: [[3]] },
    ]);

    // 'a' should have been evicted
    const a = await cache.get('a');
    expect(a).toBeUndefined();

    const b = await cache.get('b');
    expect(b).toEqual([[2]]);
    const c = await cache.get('c');
    expect(c).toEqual([[3]]);
  });

  it('handles empty data array', async () => {
    const cache = createLRUCache({ maxSize: 10 });
    await warmCache(cache, []);
    expect(cache.getStats().size).toBe(0);
  });
});

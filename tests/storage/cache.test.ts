import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createLRUCache } from '../../src/storage/cache';
import type { CacheProvider } from '../../src/types';

const f = (arr: number[]) => new Float32Array(arr);

describe('createLRUCache', () => {
  let cache: CacheProvider;

  beforeEach(() => {
    cache = createLRUCache();
  });

  it('implements CacheProvider interface', () => {
    expect(typeof cache.get).toBe('function');
    expect(typeof cache.set).toBe('function');
    expect(typeof cache.has).toBe('function');
    expect(typeof cache.delete).toBe('function');
    expect(typeof cache.clear).toBe('function');
  });

  it('basic get/set', async () => {
    const value = [f([1, 2, 3])];
    await cache.set('key1', value);
    const result = await cache.get('key1');
    expect(result).toBeDefined();
    expect(result![0]).toBeInstanceOf(Float32Array);
    expect(Array.from(result![0])).toEqual([1, 2, 3]);
  });

  it('get returns undefined for missing key', async () => {
    const result = await cache.get('nonexistent');
    expect(result).toBeUndefined();
  });

  it('has returns true for existing keys', async () => {
    await cache.set('key1', [f([1])]);
    expect(await cache.has('key1')).toBe(true);
    expect(await cache.has('missing')).toBe(false);
  });

  it('evicts least recently used when exceeding maxSize', async () => {
    const small = createLRUCache({ maxSize: 3 });
    await small.set('a', [f([1])]);
    await small.set('b', [f([2])]);
    await small.set('c', [f([3])]);
    await small.set('d', [f([4])]); // should evict 'a'

    expect(await small.has('a')).toBe(false);
    expect(await small.has('b')).toBe(true);
    expect(await small.has('c')).toBe(true);
    expect(await small.has('d')).toBe(true);
  });

  it('TTL expiration', async () => {
    vi.useFakeTimers();
    const ttlCache = createLRUCache({ ttl: 100 });

    await ttlCache.set('key1', [f([1])]);
    expect(await ttlCache.has('key1')).toBe(true);

    vi.advanceTimersByTime(150);
    expect(await ttlCache.get('key1')).toBeUndefined();
    expect(await ttlCache.has('key1')).toBe(false);

    vi.useRealTimers();
  });

  it('delete removes item', async () => {
    await cache.set('key1', [f([1])]);
    await cache.delete('key1');
    expect(await cache.has('key1')).toBe(false);
    expect(await cache.get('key1')).toBeUndefined();
  });

  it('clear empties cache', async () => {
    await cache.set('a', [f([1])]);
    await cache.set('b', [f([2])]);
    await cache.clear();
    expect(await cache.has('a')).toBe(false);
    expect(await cache.has('b')).toBe(false);
  });

  it('access order updated on get (recently accessed not evicted)', async () => {
    const small = createLRUCache({ maxSize: 3 });
    await small.set('a', [f([1])]);
    await small.set('b', [f([2])]);
    await small.set('c', [f([3])]);

    // Access 'a' to move it to most recent
    await small.get('a');

    // Adding 'd' should evict 'b' (least recently used), not 'a'
    await small.set('d', [f([4])]);

    expect(await small.has('a')).toBe(true);
    expect(await small.has('b')).toBe(false);
    expect(await small.has('c')).toBe(true);
    expect(await small.has('d')).toBe(true);
  });

  it('set existing key updates value and moves to front', async () => {
    const small = createLRUCache({ maxSize: 3 });
    await small.set('a', [f([1])]);
    await small.set('b', [f([2])]);
    await small.set('c', [f([3])]);

    // Update 'a' — moves to front
    await small.set('a', [f([10])]);

    // Adding 'd' should evict 'b' (now least recent)
    await small.set('d', [f([4])]);

    const a = await small.get('a');
    expect(a![0]).toBeInstanceOf(Float32Array);
    expect(Array.from(a![0])).toEqual([10]);
    expect(await small.has('b')).toBe(false);
  });

  it('default maxSize is 1000', async () => {
    const defaultCache = createLRUCache();
    // Add 1001 items — first one should be evicted
    for (let i = 0; i < 1001; i++) {
      await defaultCache.set(`key${i}`, [f([i])]);
    }
    expect(await defaultCache.has('key0')).toBe(false);
    expect(await defaultCache.has('key1')).toBe(true);
    expect(await defaultCache.has('key1000')).toBe(true);
  });
});

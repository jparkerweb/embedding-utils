import { describe, it, expect } from 'vitest';
import { createLRUCache } from '../../src/storage/cache';

describe('LRU cache mutation isolation', () => {
  it('stores a copy - mutating the original does not affect cached value', async () => {
    const cache = createLRUCache({ maxSize: 10 });
    const original = [new Float32Array([1, 2, 3])];
    await cache.set('key', original);

    // Mutate the original array
    original[0][0] = 999;

    const cached = await cache.get('key');
    expect(cached![0]).toBeInstanceOf(Float32Array);
    expect(Array.from(cached![0])).toEqual([1, 2, 3]);
  });

  it('returns a copy - mutating the retrieved value does not affect the cache', async () => {
    const cache = createLRUCache({ maxSize: 10 });
    await cache.set('key', [new Float32Array([1, 2, 3])]);

    const first = await cache.get('key');
    first![0][0] = 999;

    const second = await cache.get('key');
    expect(Array.from(second![0])).toEqual([1, 2, 3]);
  });

  it('multiple get calls return independent copies', async () => {
    const cache = createLRUCache({ maxSize: 10 });
    await cache.set('key', [new Float32Array([10, 20, 30])]);

    const a = await cache.get('key');
    const b = await cache.get('key');

    a![0][0] = -1;
    expect(b![0][0]).toBe(10);
  });
});

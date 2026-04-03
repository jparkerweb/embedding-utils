import { describe, it, expect, vi } from 'vitest';
import { withCache } from '../../src/providers/middleware';
import type { EmbeddingProvider, EmbeddingResult } from '../../src/types';

function createMockProvider(embeddings: number[][] = [[0.1, 0.2, 0.3]]): EmbeddingProvider & { embedCalls: number } {
  const mock = {
    name: 'mock',
    dimensions: 3,
    embedCalls: 0,
    async embed(input: string | string[]): Promise<EmbeddingResult> {
      mock.embedCalls++;
      return {
        embeddings,
        model: 'mock-model',
        dimensions: embeddings[0]?.length ?? 0,
      };
    },
  };
  return mock;
}

describe('withCache', () => {
  it('caches on second call with same input', async () => {
    const provider = createMockProvider();
    const cached = withCache(provider);

    await cached.embed('hello');
    await cached.embed('hello');

    expect(provider.embedCalls).toBe(1);
  });

  it('calls provider for different inputs', async () => {
    const provider = createMockProvider();
    const cached = withCache(provider);

    await cached.embed('hello');
    await cached.embed('world');

    expect(provider.embedCalls).toBe(2);
  });

  it('different dimensions produce different cache entries', async () => {
    const provider = createMockProvider();
    const cached = withCache(provider);

    await cached.embed('hello', { dimensions: 2 });
    await cached.embed('hello', { dimensions: 3 });

    expect(provider.embedCalls).toBe(2);
  });

  it('respects maxSize option', async () => {
    const provider = createMockProvider();
    const cached = withCache(provider, { maxSize: 2 });

    await cached.embed('a');
    await cached.embed('b');
    await cached.embed('c'); // evicts 'a'

    provider.embedCalls = 0;

    // 'a' was evicted, should call provider
    await cached.embed('a');
    expect(provider.embedCalls).toBe(1);

    // 'b' might be evicted too since 'c' was accessed more recently
    // 'c' should still be cached
    provider.embedCalls = 0;
    await cached.embed('c');
    expect(provider.embedCalls).toBe(0);
  });

  it('returns copies so cached data is not mutated externally', async () => {
    const provider = createMockProvider([[1, 2, 3]]);
    const cached = withCache(provider);

    const r1 = await cached.embed('test');
    r1.embeddings[0][0] = 999;

    const r2 = await cached.embed('test');
    expect(r2.embeddings[0][0]).toBe(1);
  });

  it('preserves provider name and dimensions', () => {
    const provider = createMockProvider();
    const cached = withCache(provider);
    expect(cached.name).toBe('mock');
    expect(cached.dimensions).toBe(3);
  });

  it('supports custom hashFunction', async () => {
    const provider = createMockProvider();
    const hashFn = vi.fn((key: string) => 'fixed-hash');
    const cached = withCache(provider, { hashFunction: hashFn });

    await cached.embed('hello');
    await cached.embed('world'); // same hash -> cache hit

    expect(hashFn).toHaveBeenCalledTimes(2);
    expect(provider.embedCalls).toBe(1); // only called once due to same hash
  });

  it('handles array input', async () => {
    const provider = createMockProvider([[0.1, 0.2], [0.3, 0.4]]);
    const cached = withCache(provider);

    await cached.embed(['hello', 'world']);
    await cached.embed(['hello', 'world']);

    expect(provider.embedCalls).toBe(1);
  });
});

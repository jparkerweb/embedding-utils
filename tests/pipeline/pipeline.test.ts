import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { batchIterator, createEmbeddingPipeline } from '../../src/pipeline/pipeline';
import type { EmbeddingProvider, EmbeddingResult } from '../../src/types';

// ── Task 7.3: Batch Iterator ────────────────────────────────────────────────

async function collectBatches<T>(gen: AsyncGenerator<T[]>): Promise<T[][]> {
  const batches: T[][] = [];
  for await (const batch of gen) {
    batches.push(batch);
  }
  return batches;
}

describe('batchIterator', () => {
  it('splits 10 items with batchSize 3 into [3, 3, 3, 1]', async () => {
    const items = Array.from({ length: 10 }, (_, i) => i);
    const batches = await collectBatches(batchIterator(items, 3));
    expect(batches.map((b) => b.length)).toEqual([3, 3, 3, 1]);
  });

  it('yields batches in order', async () => {
    const items = ['a', 'b', 'c', 'd', 'e'];
    const batches = await collectBatches(batchIterator(items, 2));
    expect(batches).toEqual([['a', 'b'], ['c', 'd'], ['e']]);
  });

  it('handles empty input with no batches yielded', async () => {
    const batches = await collectBatches(batchIterator([], 5));
    expect(batches).toEqual([]);
  });

  it('single item yields one batch of 1', async () => {
    const batches = await collectBatches(batchIterator(['x'], 3));
    expect(batches).toEqual([['x']]);
  });

  it('batchSize larger than input yields one batch with all items', async () => {
    const items = [1, 2, 3];
    const batches = await collectBatches(batchIterator(items, 100));
    expect(batches).toEqual([[1, 2, 3]]);
  });
});

// ── Task 7.5: Pipeline End-to-End ───────────────────────────────────────────

function createMockProvider(dims: number = 3): EmbeddingProvider & { callCount: number; concurrentCalls: number; maxConcurrent: number } {
  const provider = {
    name: 'mock',
    dimensions: dims,
    callCount: 0,
    concurrentCalls: 0,
    maxConcurrent: 0,
    async embed(input: string | string[]): Promise<EmbeddingResult> {
      const texts = Array.isArray(input) ? input : [input];
      provider.callCount++;
      provider.concurrentCalls++;
      provider.maxConcurrent = Math.max(provider.maxConcurrent, provider.concurrentCalls);

      // Small delay to allow concurrency tracking
      await new Promise((r) => setTimeout(r, 1));

      provider.concurrentCalls--;

      const embeddings = texts.map((t) => {
        const vec = new Float32Array(dims);
        // Deterministic: hash from text length
        for (let i = 0; i < dims; i++) {
          vec[i] = (t.length + i) / 10;
        }
        return vec;
      });

      return {
        embeddings,
        model: 'mock-model',
        dimensions: dims,
        usage: { tokens: texts.reduce((sum, t) => sum + t.length, 0) },
      };
    },
  };
  return provider;
}

describe('createEmbeddingPipeline', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('processes all input texts and returns Float32Array[] embeddings', async () => {
    vi.useRealTimers();
    const provider = createMockProvider();
    const pipeline = createEmbeddingPipeline(provider, { batchSize: 3 });

    const texts = ['hello', 'world', 'foo', 'bar', 'baz'];
    const result = await pipeline.embed(texts);

    expect(result).toHaveLength(5);
    result.forEach((vec) => {
      expect(vec).toBeInstanceOf(Float32Array);
      expect(vec.length).toBe(3);
    });
  });

  it('respects concurrency control', async () => {
    vi.useRealTimers();
    const provider = createMockProvider();
    const pipeline = createEmbeddingPipeline(provider, {
      batchSize: 2,
      concurrency: 2,
    });

    const texts = Array.from({ length: 10 }, (_, i) => `text-${i}`);
    await pipeline.embed(texts);

    // 10 texts / batchSize 2 = 5 batches, at most 2 concurrent
    expect(provider.callCount).toBe(5);
    expect(provider.maxConcurrent).toBeLessThanOrEqual(2);
  });

  it('fires onProgress callback after each batch', async () => {
    vi.useRealTimers();
    const provider = createMockProvider();
    const progressCalls: Array<{ completed: number; total: number; elapsed: number }> = [];
    const pipeline = createEmbeddingPipeline(provider, {
      batchSize: 3,
      onProgress(info) {
        progressCalls.push({ ...info });
      },
    });

    const texts = Array.from({ length: 7 }, (_, i) => `text-${i}`);
    await pipeline.embed(texts);

    // 7 texts / batchSize 3 → 3 batches
    expect(progressCalls).toHaveLength(3);
    expect(progressCalls[0].completed).toBe(1);
    expect(progressCalls[0].total).toBe(3);
    expect(progressCalls[2].completed).toBe(3);
    expect(progressCalls[2].total).toBe(3);
    progressCalls.forEach((p) => {
      expect(p.elapsed).toBeGreaterThanOrEqual(0);
    });
  });

  it('propagates provider errors immediately', async () => {
    vi.useRealTimers();
    const provider = createMockProvider();
    const originalEmbed = provider.embed.bind(provider);
    let callNum = 0;
    provider.embed = async (input: string | string[]) => {
      callNum++;
      if (callNum === 2) throw new Error('provider failure');
      return originalEmbed(input);
    };

    const pipeline = createEmbeddingPipeline(provider, { batchSize: 2, concurrency: 1 });
    await expect(pipeline.embed(['a', 'b', 'c', 'd'])).rejects.toThrow('provider failure');
  });

  it('integrates with rate limiter', async () => {
    vi.useRealTimers();
    const provider = createMockProvider();
    const pipeline = createEmbeddingPipeline(provider, {
      batchSize: 5,
      rateLimit: { requestsPerMinute: 1000, tokensPerMinute: 100000 },
    });

    const texts = ['hello', 'world'];
    const result = await pipeline.embed(texts);
    expect(result).toHaveLength(2);
  });
});

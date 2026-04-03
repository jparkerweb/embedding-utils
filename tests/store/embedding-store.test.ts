import { describe, it, expect, vi } from 'vitest';
import { createEmbeddingStore } from '../../src/store/embedding-store';
import type { EmbeddingProvider, EmbeddingResult } from '../../src/types';

function createMockProvider(): EmbeddingProvider & { embedCalls: number } {
  let callCount = 0;
  return {
    name: 'mock',
    dimensions: 3,
    get embedCalls() {
      return callCount;
    },
    async embed(input: string | string[]): Promise<EmbeddingResult> {
      callCount++;
      const inputs = Array.isArray(input) ? input : [input];
      // Produce deterministic embeddings based on text hash
      const embeddings = inputs.map((text) => {
        const hash = text.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);
        return [hash % 10 / 10, (hash * 7) % 10 / 10, (hash * 13) % 10 / 10];
      });
      return { embeddings, model: 'mock', dimensions: 3 };
    },
  };
}

describe('createEmbeddingStore', () => {
  it('add and search roundtrip', async () => {
    const provider = createMockProvider();
    const store = createEmbeddingStore({ provider });

    await store.add('doc1', 'hello world');
    await store.add('doc2', 'goodbye world');

    expect(store.size).toBe(2);

    const results = await store.search('hello world', { topK: 2 });
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]).toHaveProperty('score');
    expect(results[0]).toHaveProperty('id');
  });

  it('addBatch stores multiple items', async () => {
    const provider = createMockProvider();
    const store = createEmbeddingStore({ provider });

    await store.addBatch([
      { id: 'a', text: 'alpha' },
      { id: 'b', text: 'beta' },
      { id: 'c', text: 'gamma' },
    ]);

    expect(store.size).toBe(3);
  });

  it('remove deletes an item', async () => {
    const provider = createMockProvider();
    const store = createEmbeddingStore({ provider });

    await store.add('doc1', 'hello');
    expect(store.size).toBe(1);
    expect(store.remove('doc1')).toBe(true);
    expect(store.size).toBe(0);
    expect(store.remove('doc1')).toBe(false);
  });

  it('clear removes all items', async () => {
    const provider = createMockProvider();
    const store = createEmbeddingStore({ provider });

    await store.add('doc1', 'hello');
    await store.add('doc2', 'world');
    expect(store.size).toBe(2);
    store.clear();
    expect(store.size).toBe(0);
  });

  it('searchByEmbedding works with pre-computed vectors', async () => {
    const provider = createMockProvider();
    const store = createEmbeddingStore({ provider });

    await store.add('doc1', 'hello');
    const results = store.searchByEmbedding([0.5, 0.5, 0.5], { topK: 1 });
    expect(results.length).toBe(1);
    expect(results[0].id).toBe('doc1');
  });

  it('cache integration: provider called once for duplicate adds', async () => {
    const provider = createMockProvider();
    const store = createEmbeddingStore({
      provider,
      cache: { maxSize: 100 },
    });

    await store.add('doc1', 'same text');
    await store.add('doc2', 'same text');

    // With cache: first call misses, second call hits cache
    // Provider should be called once for the text, second time from cache
    expect(provider.embedCalls).toBe(1);
  });

  it('supports metadata on add', async () => {
    const provider = createMockProvider();
    const store = createEmbeddingStore({ provider });

    await store.add('doc1', 'hello', { title: 'Greeting' });
    const results = await store.search('hello', { topK: 1 });
    expect(results[0].metadata).toEqual({ title: 'Greeting' });
  });
});

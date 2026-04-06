import { describe, it, expect } from 'vitest';
import { HNSWIndex } from '../../src/search/hnsw';
import { SearchIndex } from '../../src/search/search-index';
import type { SimilarityMetric } from '../../src/types';

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function randomVector(dims: number): Float32Array {
  const v = new Float32Array(dims);
  for (let i = 0; i < dims; i++) v[i] = Math.random() * 2 - 1;
  return v;
}

function generateVectors(count: number, dims: number): Float32Array[] {
  return Array.from({ length: count }, () => randomVector(dims));
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 4.1: Construction and basic operations
// ─────────────────────────────────────────────────────────────────────────────

describe('HNSWIndex — construction and basic operations', () => {
  it('creates empty index with size === 0', () => {
    const index = new HNSWIndex({ metric: 'cosine' });
    expect(index.size).toBe(0);
  });

  it('add increases size by 1', () => {
    const index = new HNSWIndex();
    index.add('doc1', [1, 0, 0], { title: 'Hello' });
    expect(index.size).toBe(1);
  });

  it('addBatch adds multiple items', () => {
    const index = new HNSWIndex();
    index.addBatch([
      { id: 'a', vector: [1, 0, 0] },
      { id: 'b', vector: [0, 1, 0] },
      { id: 'c', vector: [0, 0, 1] },
    ]);
    expect(index.size).toBe(3);
  });

  it('get returns stored item with correct vector and metadata', () => {
    const index = new HNSWIndex();
    index.add('doc1', [1, 2, 3], { title: 'Test' });
    const item = index.get('doc1');
    expect(item).toBeDefined();
    expect(item!.id).toBe('doc1');
    expect(item!.embedding).toBeInstanceOf(Float32Array);
    expect(Array.from(item!.embedding)).toEqual([1, 2, 3]);
    expect(item!.metadata).toEqual({ title: 'Test' });
  });

  it('get without metadata returns undefined metadata', () => {
    const index = new HNSWIndex();
    index.add('doc1', [1, 0, 0]);
    const item = index.get('doc1');
    expect(item!.metadata).toBeUndefined();
  });

  it('clear resets size to 0', () => {
    const index = new HNSWIndex();
    index.addBatch([
      { id: 'a', vector: [1, 0] },
      { id: 'b', vector: [0, 1] },
    ]);
    expect(index.size).toBe(2);
    index.clear();
    expect(index.size).toBe(0);
    expect(index.get('a')).toBeUndefined();
  });

  it('remove decreases size, item no longer found by search', () => {
    const index = new HNSWIndex();
    index.add('doc1', [1, 0, 0]);
    index.add('doc2', [0, 1, 0]);
    expect(index.size).toBe(2);

    const removed = index.remove('doc1');
    expect(removed).toBe(true);
    expect(index.size).toBe(1);
    expect(index.get('doc1')).toBeUndefined();

    const results = index.search([1, 0, 0], { topK: 10 });
    expect(results.every((r) => r.id !== 'doc1')).toBe(true);
  });

  it('remove returns false for nonexistent ID', () => {
    const index = new HNSWIndex();
    expect(index.remove('nonexistent')).toBe(false);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Task 4.2: Search accuracy
// ─────────────────────────────────────────────────────────────────────────────

describe('HNSWIndex — search accuracy', () => {
  const DIMS = 128;
  const COUNT = 1000;

  it('search returns topK results ordered by score descending', () => {
    const index = new HNSWIndex({ metric: 'cosine' });
    const vectors = generateVectors(50, 32);
    vectors.forEach((v, i) => index.add(`v${i}`, v));

    const query = randomVector(32);
    const results = index.search(query, { topK: 10 });

    expect(results).toHaveLength(10);
    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score);
    }
  });

  it('achieves >95% recall@10 vs brute-force (cosine)', () => {
    const index = new HNSWIndex({ metric: 'cosine', efConstruction: 200 });
    const bruteForce = new SearchIndex({ metric: 'cosine' });
    const vectors = generateVectors(COUNT, DIMS);

    vectors.forEach((v, i) => {
      index.add(`v${i}`, v);
      bruteForce.add(`v${i}`, v);
    });

    let totalOverlap = 0;
    const numQueries = 50;

    for (let q = 0; q < numQueries; q++) {
      const query = randomVector(DIMS);
      const hnswResults = index.search(query, { topK: 10, efSearch: 100 });
      const bfResults = bruteForce.search(query, { topK: 10 });

      const hnswIds = new Set(hnswResults.map((r) => r.id));
      const bfIds = new Set(bfResults.map((r) => r.id));

      let overlap = 0;
      for (const id of bfIds) {
        if (hnswIds.has(id)) overlap++;
      }
      totalOverlap += overlap / 10;
    }

    const avgRecall = totalOverlap / numQueries;
    expect(avgRecall).toBeGreaterThan(0.95);
  });

  it.each(['euclidean', 'dot', 'manhattan'] as SimilarityMetric[])(
    'search works with %s metric',
    (metric) => {
      const index = new HNSWIndex({ metric });
      const bruteForce = new SearchIndex({ metric });
      const vectors = generateVectors(200, 32);

      vectors.forEach((v, i) => {
        index.add(`v${i}`, v);
        bruteForce.add(`v${i}`, v);
      });

      const query = randomVector(32);
      const hnswResults = index.search(query, { topK: 5, efSearch: 100 });
      const bfResults = bruteForce.search(query, { topK: 5 });

      expect(hnswResults.length).toBeGreaterThan(0);
      // At least 3 of top-5 should overlap
      const hnswIds = new Set(hnswResults.map((r) => r.id));
      const bfIds = bfResults.map((r) => r.id);
      const overlap = bfIds.filter((id) => hnswIds.has(id)).length;
      expect(overlap).toBeGreaterThanOrEqual(3);
    },
  );

  it('filter option restricts results', () => {
    const index = new HNSWIndex({ metric: 'cosine' });
    index.add('cat-1', [1, 0, 0], { type: 'cat' });
    index.add('dog-1', [0.9, 0.1, 0], { type: 'dog' });
    index.add('cat-2', [0.8, 0.2, 0], { type: 'cat' });

    const results = index.search([1, 0, 0], {
      topK: 10,
      filter: (item) => (item.metadata as { type: string })?.type === 'dog',
    });

    expect(results).toHaveLength(1);
    expect(results[0].id).toBe('dog-1');
  });

  it('higher efSearch improves recall', () => {
    const index = new HNSWIndex({ metric: 'cosine', M: 8, efConstruction: 100 });
    const bruteForce = new SearchIndex({ metric: 'cosine' });
    const vectors = generateVectors(500, 64);

    vectors.forEach((v, i) => {
      index.add(`v${i}`, v);
      bruteForce.add(`v${i}`, v);
    });

    const query = randomVector(64);
    const bfResults = bruteForce.search(query, { topK: 10 });
    const bfIds = new Set(bfResults.map((r) => r.id));

    const lowEf = index.search(query, { topK: 10, efSearch: 10 });
    const highEf = index.search(query, { topK: 10, efSearch: 200 });

    const lowOverlap = lowEf.filter((r) => bfIds.has(r.id)).length;
    const highOverlap = highEf.filter((r) => bfIds.has(r.id)).length;

    expect(highOverlap).toBeGreaterThanOrEqual(lowOverlap);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Task 4.3: Edge cases
// ─────────────────────────────────────────────────────────────────────────────

describe('HNSWIndex — edge cases', () => {
  it('search on empty index returns empty array', () => {
    const index = new HNSWIndex();
    const results = index.search([1, 0, 0]);
    expect(results).toEqual([]);
  });

  it('adding item with duplicate ID overwrites existing', () => {
    const index = new HNSWIndex();
    index.add('doc1', [1, 0, 0], { v: 1 });
    index.add('doc1', [0, 1, 0], { v: 2 });

    expect(index.size).toBe(1);
    const item = index.get('doc1');
    expect(Array.from(item!.embedding)).toEqual([0, 1, 0]);
    expect(item!.metadata).toEqual({ v: 2 });
  });

  it('remove then search does not return removed item', () => {
    const index = new HNSWIndex();
    index.add('a', [1, 0, 0]);
    index.add('b', [0.9, 0.1, 0]);
    index.add('c', [0.8, 0.2, 0]);

    index.remove('a');

    const results = index.search([1, 0, 0], { topK: 10 });
    expect(results.every((r) => r.id !== 'a')).toBe(true);
    expect(results.length).toBe(2);
  });

  it('single item in index — search returns that item', () => {
    const index = new HNSWIndex();
    index.add('only', [1, 0, 0]);

    const results = index.search([1, 0, 0], { topK: 5 });
    expect(results).toHaveLength(1);
    expect(results[0].id).toBe('only');
  });

  it('topK larger than index size returns all items', () => {
    const index = new HNSWIndex();
    index.add('a', [1, 0, 0]);
    index.add('b', [0, 1, 0]);

    const results = index.search([1, 0, 0], { topK: 100 });
    expect(results).toHaveLength(2);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Task 4.4: Serialization
// ─────────────────────────────────────────────────────────────────────────────

describe('HNSWIndex — serialization', () => {
  it('serialize/deserialize roundtrip preserves search results', () => {
    const index = new HNSWIndex({ metric: 'cosine' });
    const vectors = generateVectors(500, 64);
    vectors.forEach((v, i) => index.add(`v${i}`, v, { idx: i }));

    const query = randomVector(64);
    const originalResults = index.search(query, { topK: 10 });

    const data = index.serialize();
    const restored = HNSWIndex.deserialize(data);
    const restoredResults = restored.search(query, { topK: 10 });

    expect(restoredResults).toHaveLength(originalResults.length);
    expect(restoredResults.map((r) => r.id)).toEqual(originalResults.map((r) => r.id));
    // Scores should match
    for (let i = 0; i < originalResults.length; i++) {
      expect(restoredResults[i].score).toBeCloseTo(originalResults[i].score, 5);
    }
  });

  it('serialized data starts with version byte', () => {
    const index = new HNSWIndex();
    index.add('a', [1, 0, 0]);
    const data = index.serialize();
    expect(data[0]).toBe(1); // HNSW_FORMAT_VERSION = 1
  });

  it('deserializing corrupted/truncated data throws descriptive error', () => {
    expect(() => HNSWIndex.deserialize(new Uint8Array([1]))).toThrow(/Corrupted HNSW data/);
    expect(() => HNSWIndex.deserialize(new Uint8Array([1, 0, 0]))).toThrow(/Corrupted HNSW data/);
  });

  it('deserializing data with unknown version throws descriptive error', () => {
    const data = new Uint8Array([99, 0, 0, 0, 2, 123, 125]); // version 99
    expect(() => HNSWIndex.deserialize(data)).toThrow(/Unsupported HNSW format version: 99/);
  });

  it('deserialized index preserves metadata', () => {
    const index = new HNSWIndex();
    index.add('doc1', [1, 0, 0], { title: 'Hello', count: 42 });

    const data = index.serialize();
    const restored = HNSWIndex.deserialize(data);

    const item = restored.get('doc1');
    expect(item).toBeDefined();
    expect(item!.metadata).toEqual({ title: 'Hello', count: 42 });
  });

  it('deserialized index has correct size', () => {
    const index = new HNSWIndex();
    for (let i = 0; i < 100; i++) {
      index.add(`v${i}`, randomVector(16));
    }

    const data = index.serialize();
    const restored = HNSWIndex.deserialize(data);
    expect(restored.size).toBe(100);
  });
});

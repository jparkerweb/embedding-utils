import { describe, it, expect } from 'vitest';
import { SearchIndex } from '../../src/search/search-index';

describe('SearchIndex', () => {
  it('add and get', () => {
    const index = new SearchIndex();
    index.add('doc1', [1, 0, 0], { title: 'Hello' });
    const item = index.get('doc1');
    expect(item).toBeDefined();
    expect(item!.id).toBe('doc1');
    expect(item!.embedding).toBeInstanceOf(Float32Array);
    expect(Array.from(item!.embedding)).toEqual([1, 0, 0]);
    expect(item!.metadata).toEqual({ title: 'Hello' });
  });

  it('add without metadata', () => {
    const index = new SearchIndex();
    index.add('doc1', [1, 0, 0]);
    const item = index.get('doc1');
    expect(item!.metadata).toBeUndefined();
  });

  it('duplicate ID overwrites', () => {
    const index = new SearchIndex();
    index.add('doc1', [1, 0, 0]);
    index.add('doc1', [0, 1, 0], { updated: true });
    expect(index.size).toBe(1);
    expect(Array.from(index.get('doc1')!.embedding)).toEqual([0, 1, 0]);
    expect(index.get('doc1')!.metadata).toEqual({ updated: true });
  });

  it('addBatch adds multiple items', () => {
    const index = new SearchIndex();
    index.addBatch([
      { id: 'a', embedding: [1, 0, 0] },
      { id: 'b', embedding: [0, 1, 0] },
      { id: 'c', embedding: [0, 0, 1] },
    ]);
    expect(index.size).toBe(3);
  });

  it('remove returns true for existing, false for missing', () => {
    const index = new SearchIndex();
    index.add('doc1', [1, 0, 0]);
    expect(index.remove('doc1')).toBe(true);
    expect(index.remove('doc1')).toBe(false);
    expect(index.remove('nonexistent')).toBe(false);
    expect(index.size).toBe(0);
  });

  it('search returns results sorted by score', () => {
    const index = new SearchIndex();
    index.add('exact', [1, 0, 0]);
    index.add('similar', [0.9, 0.1, 0]);
    index.add('orthogonal', [0, 1, 0]);

    const results = index.search([1, 0, 0], { topK: 3 });
    expect(results).toHaveLength(3);
    expect(results[0].id).toBe('exact');
    expect(results[0].score).toBeGreaterThan(results[1].score);
    expect(results[1].score).toBeGreaterThan(results[2].score);
  });

  it('search with threshold', () => {
    const index = new SearchIndex();
    index.add('exact', [1, 0, 0]);
    index.add('orthogonal', [0, 1, 0]);

    const results = index.search([1, 0, 0], { threshold: 0.9 });
    expect(results).toHaveLength(1);
    expect(results[0].id).toBe('exact');
  });

  it('search with filter', () => {
    const index = new SearchIndex();
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

  it('get returns undefined for missing ID', () => {
    const index = new SearchIndex();
    expect(index.get('missing')).toBeUndefined();
  });

  it('size reflects current count', () => {
    const index = new SearchIndex();
    expect(index.size).toBe(0);
    index.add('a', [1, 0]);
    expect(index.size).toBe(1);
    index.add('b', [0, 1]);
    expect(index.size).toBe(2);
    index.remove('a');
    expect(index.size).toBe(1);
  });

  it('clear removes all items', () => {
    const index = new SearchIndex();
    index.addBatch([
      { id: 'a', embedding: [1, 0] },
      { id: 'b', embedding: [0, 1] },
    ]);
    expect(index.size).toBe(2);
    index.clear();
    expect(index.size).toBe(0);
    expect(index.get('a')).toBeUndefined();
  });

  it('supports custom metric', () => {
    const index = new SearchIndex({ metric: 'euclidean' });
    index.add('near', [1.1, 0, 0]);
    index.add('far', [0, 1, 0]);

    const results = index.search([1, 0, 0], { topK: 2 });
    expect(results[0].id).toBe('near');
  });
});

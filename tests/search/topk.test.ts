import { describe, it, expect } from 'vitest';
import { topK, topKMulti } from '../../src/search/topk';

describe('topK', () => {
  const query = [1, 0, 0];
  const corpus = [
    [1, 0, 0],    // identical to query
    [0, 1, 0],    // orthogonal
    [0.9, 0.1, 0], // very similar
    [-1, 0, 0],   // opposite
  ];

  it('returns k most similar embeddings in descending order', () => {
    const results = topK(query, corpus, 2);
    expect(results).toHaveLength(2);
    expect(results[0].index).toBe(0);
    expect(results[1].index).toBe(2);
    expect(results[0].score).toBeGreaterThan(results[1].score);
  });

  it('returns all when k > corpus size', () => {
    const results = topK(query, corpus, 10);
    expect(results).toHaveLength(corpus.length);
  });

  it('returns empty when k = 0', () => {
    expect(topK(query, corpus, 0)).toEqual([]);
  });

  it('supports dot product metric', () => {
    const results = topK(query, corpus, 2, { metric: 'dot' });
    expect(results).toHaveLength(2);
    expect(results[0].index).toBe(0);
  });

  it('supports euclidean metric', () => {
    const results = topK(query, corpus, 2, { metric: 'euclidean' });
    expect(results).toHaveLength(2);
    // Closest by euclidean distance → highest similarity score
    expect(results[0].index).toBe(0);
  });

  it('attaches labels when provided', () => {
    const labels = ['a', 'b', 'c', 'd'];
    const results = topK(query, corpus, 2, { labels });
    expect(results[0].label).toBe('a');
    expect(results[1].label).toBe('c');
  });

  it('includes embedding in results', () => {
    const results = topK(query, corpus, 1);
    expect(results[0].embedding).toEqual([1, 0, 0]);
  });
});

describe('topKMulti', () => {
  const corpus = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];

  it('returns independent topK results for each query', () => {
    const queries = [
      [1, 0, 0],
      [0, 1, 0],
    ];
    const results = topKMulti(queries, corpus, 1);
    expect(results).toHaveLength(2);
    expect(results[0][0].index).toBe(0);
    expect(results[1][0].index).toBe(1);
  });

  it('passes options through to topK', () => {
    const queries = [[1, 0, 0]];
    const labels = ['x', 'y', 'z'];
    const results = topKMulti(queries, corpus, 1, { labels });
    expect(results[0][0].label).toBe('x');
  });
});

import { describe, it, expect } from 'vitest';
import { aboveThreshold, deduplicate } from '../../src/search/threshold';

describe('aboveThreshold', () => {
  const query = [1, 0, 0];
  const corpus = [
    [1, 0, 0],     // similarity = 1.0
    [0.9, 0.1, 0], // high similarity
    [0, 1, 0],     // similarity = 0.0
    [-1, 0, 0],    // similarity = -1.0
  ];

  it('returns embeddings with similarity >= threshold', () => {
    const results = aboveThreshold(query, corpus, 0.9);
    expect(results.length).toBeGreaterThanOrEqual(1);
    for (const r of results) {
      expect(r.score).toBeGreaterThanOrEqual(0.9);
    }
  });

  it('returns empty when none qualify', () => {
    const results = aboveThreshold(query, corpus, 1.1);
    expect(results).toHaveLength(0);
  });

  it('returns all for threshold=0 with cosine (non-negative scores)', () => {
    // Only entries with cosine >= 0
    const results = aboveThreshold(query, corpus, 0);
    // [1,0,0] = 1.0, [0.9,0.1,0] > 0, [0,1,0] = 0.0 — all >= 0
    // [-1,0,0] = -1.0 — excluded
    expect(results).toHaveLength(3);
  });

  it('results are sorted descending by score', () => {
    const results = aboveThreshold(query, corpus, 0);
    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score);
    }
  });

  it('attaches labels when provided', () => {
    const labels = ['a', 'b', 'c', 'd'];
    const results = aboveThreshold(query, corpus, 0.9, { labels });
    for (const r of results) {
      expect(r.label).toBeDefined();
    }
  });
});

describe('deduplicate', () => {
  it('removes near-duplicate embeddings (keeps first)', () => {
    const embeddings = [
      [1, 0, 0],
      [1, 0, 0],   // exact duplicate
      [0, 1, 0],   // unique
    ];
    const result = deduplicate(embeddings, 0.99);
    expect(result.embeddings).toHaveLength(2);
    expect(result.indices).toEqual([0, 2]);
  });

  it('returns all when all are unique', () => {
    const embeddings = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    const result = deduplicate(embeddings, 0.99);
    expect(result.embeddings).toHaveLength(3);
  });

  it('returns one when all are identical', () => {
    const embeddings = [
      [1, 0, 0],
      [1, 0, 0],
      [1, 0, 0],
    ];
    const result = deduplicate(embeddings, 0.99);
    expect(result.embeddings).toHaveLength(1);
    expect(result.indices).toEqual([0]);
  });

  it('preserves labels on kept items', () => {
    const embeddings = [
      [1, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
    ];
    const result = deduplicate(embeddings, 0.99, { labels: ['a', 'b', 'c'] });
    expect(result.labels).toEqual(['a', 'c']);
  });

  it('handles empty input', () => {
    const result = deduplicate([], 0.99);
    expect(result.embeddings).toHaveLength(0);
    expect(result.indices).toEqual([]);
  });
});

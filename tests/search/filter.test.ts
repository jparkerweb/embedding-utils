import { describe, it, expect } from 'vitest';
import { topK } from '../../src/search/topk';
import { aboveThreshold } from '../../src/search/threshold';

describe('topK with filter', () => {
  const query = [1, 0, 0];
  const corpus = [
    [1, 0, 0],     // index 0 (even)
    [0.9, 0.1, 0], // index 1 (odd)
    [0.8, 0.2, 0], // index 2 (even)
    [0.7, 0.3, 0], // index 3 (odd)
    [0.6, 0.4, 0], // index 4 (even)
    [0.5, 0.5, 0], // index 5 (odd)
  ];

  it('filters out even indices, returns only odd-index results', () => {
    const results = topK(query, corpus, 10, {
      filter: (index) => index % 2 !== 0,
    });
    for (const r of results) {
      expect(r.index % 2).toBe(1);
    }
    expect(results).toHaveLength(3);
  });

  it('filter receives label when labels are provided', () => {
    const labels = ['cat-a', 'dog-b', 'cat-c', 'dog-d', 'cat-e', 'dog-f'];
    const results = topK(query, corpus, 10, {
      labels,
      filter: (_index, label) => label!.startsWith('dog'),
    });
    expect(results).toHaveLength(3);
    for (const r of results) {
      expect(r.label).toMatch(/^dog/);
    }
  });

  it('heap-based path respects filter', () => {
    // k=1 < corpus.length/2=3, so heap path is used
    const results = topK(query, corpus, 1, {
      filter: (index) => index % 2 !== 0,
    });
    expect(results).toHaveLength(1);
    expect(results[0].index).toBe(1);
  });
});

describe('aboveThreshold with filter', () => {
  const query = [1, 0, 0];
  const corpus = [
    [1, 0, 0],     // index 0, very high score
    [0.95, 0.05, 0], // index 1, high score
    [0.9, 0.1, 0],   // index 2, high score
    [0, 1, 0],       // index 3, low score
  ];

  it('filters out even indices', () => {
    const results = aboveThreshold(query, corpus, 0.0, {
      filter: (index) => index % 2 !== 0,
    });
    for (const r of results) {
      expect(r.index % 2).toBe(1);
    }
    expect(results).toHaveLength(2);
  });

  it('combines filter with threshold', () => {
    const results = aboveThreshold(query, corpus, 0.9, {
      filter: (index) => index !== 0,
    });
    // index 0 filtered out, index 1 and 2 pass threshold, index 3 below threshold
    expect(results.length).toBeGreaterThanOrEqual(1);
    for (const r of results) {
      expect(r.index).not.toBe(0);
      expect(r.score).toBeGreaterThanOrEqual(0.9);
    }
  });
});

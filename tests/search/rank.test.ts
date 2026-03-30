import { describe, it, expect } from 'vitest';
import { rankBySimilarity } from '../../src/search/rank';

describe('rankBySimilarity', () => {
  const query = [1, 0, 0];
  const corpus = [
    [0, 1, 0],     // orthogonal
    [1, 0, 0],     // identical
    [0.9, 0.1, 0], // very similar
    [-1, 0, 0],    // opposite
  ];

  it('ranks corpus by similarity descending', () => {
    const results = rankBySimilarity(query, corpus);
    expect(results).toHaveLength(4);
    expect(results[0].index).toBe(1); // identical
    expect(results[1].index).toBe(2); // very similar
    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score);
    }
  });

  it('supports custom metric', () => {
    const results = rankBySimilarity(query, corpus, { metric: 'dot' });
    expect(results[0].index).toBe(1);
  });

  it('returns empty for empty corpus', () => {
    expect(rankBySimilarity(query, [])).toEqual([]);
  });

  it('preserves labels', () => {
    const labels = ['a', 'b', 'c', 'd'];
    const results = rankBySimilarity(query, corpus, { labels });
    expect(results[0].label).toBe('b');
  });

  it('handles ties consistently', () => {
    const tiedCorpus = [
      [1, 0, 0],
      [1, 0, 0],
    ];
    const results = rankBySimilarity(query, tiedCorpus);
    expect(results).toHaveLength(2);
    expect(results[0].score).toBe(results[1].score);
  });
});

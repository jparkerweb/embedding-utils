import { describe, it, expect } from 'vitest';
import { mmrSearch } from '../../src/search/mmr';

describe('mmrSearch', () => {
  const query = [1, 0, 0];
  const corpus = [
    [1, 0, 0],       // index 0: identical to query
    [0.99, 0.01, 0], // index 1: very similar to query and index 0
    [0, 1, 0],       // index 2: orthogonal/diverse
    [0, 0, 1],       // index 3: orthogonal/diverse
    [-1, 0, 0],      // index 4: opposite
  ];

  it('with lambda=1.0 matches topK ordering (pure relevance)', () => {
    const results = mmrSearch(query, corpus, 3, { lambda: 1.0 });
    expect(results).toHaveLength(3);
    // Pure relevance: should pick the most similar first
    expect(results[0].index).toBe(0);
    expect(results[1].index).toBe(1);
  });

  it('with lambda=0.0 maximizes diversity', () => {
    const results = mmrSearch(query, corpus, 3, { lambda: 0.0 });
    expect(results).toHaveLength(3);
    // First pick is still based on initial candidate pool
    // Subsequent picks should be diverse (not index 0 and 1 together early)
    const indices = results.map((r) => r.index);
    // With pure diversity, after first pick, it should avoid near-duplicates
    expect(new Set(indices).size).toBe(3); // all unique
  });

  it('results are unique', () => {
    const results = mmrSearch(query, corpus, 4, { lambda: 0.5 });
    const indices = results.map((r) => r.index);
    expect(new Set(indices).size).toBe(indices.length);
  });

  it('returns fewer results if k > corpus size', () => {
    const results = mmrSearch(query, corpus, 100);
    expect(results.length).toBeLessThanOrEqual(corpus.length);
  });

  it('returns empty for k <= 0', () => {
    expect(mmrSearch(query, corpus, 0)).toEqual([]);
  });

  it('returns empty for empty corpus', () => {
    expect(mmrSearch(query, [], 3)).toEqual([]);
  });

  it('respects fetchK to limit candidate pool', () => {
    const results = mmrSearch(query, corpus, 2, { fetchK: 3 });
    expect(results).toHaveLength(2);
    // Only the top 3 by query similarity are candidates
  });
});

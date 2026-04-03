import { describe, it, expect } from 'vitest';
import { rerankResults } from '../../src/search/rerank';

describe('rerankResults', () => {
  const query = [1, 0, 0];

  it('reranking can change result order', () => {
    // Result at index 1 has high original score but low actual similarity
    // Result at index 0 has low original score but high actual similarity
    const results = [
      { index: 0, score: 0.1, embedding: [1, 0, 0] },    // low original, high cosine sim (1.0)
      { index: 1, score: 0.9, embedding: [0, 1, 0] },    // high original, low cosine sim (0.0)
    ];

    const reranked = rerankResults(results, query);
    // With default weights (0.5/0.5): index 0 = 0.5*0.1 + 0.5*1.0 = 0.55
    //                                  index 1 = 0.5*0.9 + 0.5*0.0 = 0.45
    expect(reranked[0].index).toBe(0);
    expect(reranked[1].index).toBe(1);
  });

  it('custom weights favor original scores', () => {
    const results = [
      { index: 0, score: 0.1, embedding: [1, 0, 0] },
      { index: 1, score: 0.9, embedding: [0, 1, 0] },
    ];

    const reranked = rerankResults(results, query, {
      weights: { original: 0.9, rerank: 0.1 },
    });
    // index 0 = 0.9*0.1 + 0.1*1.0 = 0.19
    // index 1 = 0.9*0.9 + 0.1*0.0 = 0.81
    expect(reranked[0].index).toBe(1);
  });

  it('default weights are 0.5/0.5', () => {
    const results = [
      { index: 0, score: 0.6, embedding: [1, 0, 0] },
    ];
    const reranked = rerankResults(results, query);
    // 0.5 * 0.6 + 0.5 * 1.0 = 0.8
    expect(reranked[0].score).toBeCloseTo(0.8);
  });

  it('returns results sorted by combined score descending', () => {
    const results = [
      { index: 0, score: 0.5, embedding: [0.5, 0.5, 0] },
      { index: 1, score: 0.5, embedding: [1, 0, 0] },
      { index: 2, score: 0.5, embedding: [0, 1, 0] },
    ];
    const reranked = rerankResults(results, query);
    for (let i = 1; i < reranked.length; i++) {
      expect(reranked[i - 1].score).toBeGreaterThanOrEqual(reranked[i].score);
    }
  });

  it('handles empty results', () => {
    const reranked = rerankResults([], query);
    expect(reranked).toEqual([]);
  });
});

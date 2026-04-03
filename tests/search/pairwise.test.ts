import { describe, it, expect } from 'vitest';
import { pairwiseSimilarity } from '../../src/search/pairwise';
import { ValidationError, DimensionMismatchError } from '../../src/types';

describe('pairwiseSimilarity', () => {
  it('returns all 1.0 for identical lists (cosine)', () => {
    const list = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    const scores = pairwiseSimilarity(list, list);
    expect(scores).toHaveLength(3);
    for (const s of scores) {
      expect(s).toBeCloseTo(1.0);
    }
  });

  it('computes element-wise similarity', () => {
    const listA = [[1, 0, 0], [0, 1, 0]];
    const listB = [[0, 1, 0], [0, 1, 0]];
    const scores = pairwiseSimilarity(listA, listB);
    expect(scores).toHaveLength(2);
    expect(scores[0]).toBeCloseTo(0); // orthogonal
    expect(scores[1]).toBeCloseTo(1); // identical
  });

  it('throws ValidationError for different lengths', () => {
    const listA = [[1, 0], [0, 1]];
    const listB = [[1, 0]];
    expect(() => pairwiseSimilarity(listA, listB)).toThrow(ValidationError);
  });

  it('throws DimensionMismatchError for mismatched dimensions', () => {
    const listA = [[1, 0, 0]];
    const listB = [[1, 0]];
    expect(() => pairwiseSimilarity(listA, listB)).toThrow(DimensionMismatchError);
  });

  it('supports dot product metric', () => {
    const listA = [[2, 3]];
    const listB = [[4, 5]];
    const scores = pairwiseSimilarity(listA, listB, 'dot');
    expect(scores[0]).toBeCloseTo(23); // 2*4 + 3*5
  });

  it('handles empty lists', () => {
    const scores = pairwiseSimilarity([], []);
    expect(scores).toEqual([]);
  });
});

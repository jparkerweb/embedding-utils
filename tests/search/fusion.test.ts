import { describe, it, expect } from 'vitest';
import { fuseRankedLists, normalizeScores } from '../../src/search/fusion';
import { ValidationError } from '../../src/types';

describe('fuseRankedLists', () => {
  it('merges two ranked lists with overlapping items using RRF formula', () => {
    const list1 = [
      { id: 'a', score: 0.9 },
      { id: 'b', score: 0.8 },
      { id: 'c', score: 0.7 },
    ];
    const list2 = [
      { id: 'b', score: 0.95 },
      { id: 'c', score: 0.85 },
      { id: 'd', score: 0.75 },
    ];
    const result = fuseRankedLists([list1, list2]);
    // Default k=60. 'b' appears at rank 2 in list1 and rank 1 in list2
    // b: 1/(60+2) + 1/(60+1) = 1/62 + 1/61
    // a: 1/(60+1) = 1/61
    // c: 1/(60+3) + 1/(60+2) = 1/63 + 1/62
    // d: 1/(60+3) = 1/63

    expect(result[0].id).toBe('b');
    expect(result[1].id).toBe('c');

    const bScore = 1 / 62 + 1 / 61;
    expect(result[0].score).toBeCloseTo(bScore, 10);
  });

  it('returns empty results for empty list input', () => {
    const result = fuseRankedLists([]);
    expect(result).toEqual([]);
  });

  it('returns items with RRF scores for a single list', () => {
    const list = [
      { id: 'x', score: 1.0 },
      { id: 'y', score: 0.5 },
    ];
    const result = fuseRankedLists([list]);
    expect(result).toHaveLength(2);
    expect(result[0].id).toBe('x');
    expect(result[0].score).toBeCloseTo(1 / 61, 10);
    expect(result[1].id).toBe('y');
    expect(result[1].score).toBeCloseTo(1 / 62, 10);
  });

  it('maintains stable order for items with equal fused scores', () => {
    // Two lists where items appear symmetrically
    const list1 = [{ id: 'a', score: 1.0 }];
    const list2 = [{ id: 'b', score: 1.0 }];
    const result = fuseRankedLists([list1, list2]);
    // Both have score 1/(60+1) = 1/61, should maintain stable order
    expect(result).toHaveLength(2);
    expect(result[0].score).toBeCloseTo(result[1].score, 10);
  });

  it('supports custom k parameter', () => {
    const list = [
      { id: 'a', score: 1.0 },
      { id: 'b', score: 0.5 },
    ];
    const result = fuseRankedLists([list], { k: 10 });
    expect(result[0].score).toBeCloseTo(1 / 11, 10); // 1/(10+1)
    expect(result[1].score).toBeCloseTo(1 / 12, 10); // 1/(10+2)
  });

  it('throws ValidationError for non-array elements', () => {
    expect(() => fuseRankedLists(['not-an-array' as any])).toThrow(ValidationError);
  });
});

describe('normalizeScores', () => {
  it('min-max: output values in [0, 1]', () => {
    const scores = [10, 20, 30, 40, 50];
    const result = normalizeScores(scores, 'min-max');
    expect(result[0]).toBeCloseTo(0, 10);
    expect(result[4]).toBeCloseTo(1, 10);
    expect(result[2]).toBeCloseTo(0.5, 10);
    for (const v of result) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it('z-score: output mean approximately 0, std approximately 1', () => {
    const scores = [10, 20, 30, 40, 50];
    const result = normalizeScores(scores, 'z-score');
    const mean = result.reduce((a, b) => a + b, 0) / result.length;
    expect(mean).toBeCloseTo(0, 5);
    const variance = result.reduce((a, b) => a + (b - mean) ** 2, 0) / result.length;
    expect(Math.sqrt(variance)).toBeCloseTo(1, 5);
  });

  it('sigmoid: output values in (0, 1), monotonically increasing', () => {
    const scores = [-2, -1, 0, 1, 2];
    const result = normalizeScores(scores, 'sigmoid');
    for (const v of result) {
      expect(v).toBeGreaterThan(0);
      expect(v).toBeLessThan(1);
    }
    for (let i = 1; i < result.length; i++) {
      expect(result[i]).toBeGreaterThan(result[i - 1]);
    }
  });

  it('min-max: single element returns [1]', () => {
    const result = normalizeScores([42], 'min-max');
    expect(result).toEqual([1]);
  });

  it('z-score: zero std returns all 0', () => {
    const result = normalizeScores([5, 5, 5], 'z-score');
    expect(result).toEqual([0, 0, 0]);
  });

  it('returns empty for empty input', () => {
    expect(normalizeScores([], 'min-max')).toEqual([]);
    expect(normalizeScores([], 'z-score')).toEqual([]);
    expect(normalizeScores([], 'sigmoid')).toEqual([]);
  });
});

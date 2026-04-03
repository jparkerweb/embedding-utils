import { describe, it, expect } from 'vitest';
import { computeCentroid, computePairwiseCohesion } from '../../src/internal/clustering';

describe('computeCentroid', () => {
  it('computes mean of known vectors', () => {
    const centroid = computeCentroid([[1, 0], [0, 1]]);
    expect(centroid).toEqual([0.5, 0.5]);
  });

  it('returns the vector itself for single-member', () => {
    const centroid = computeCentroid([[3, 4, 5]]);
    expect(centroid).toEqual([3, 4, 5]);
  });

  it('computes centroid of three vectors', () => {
    const centroid = computeCentroid([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    expect(centroid).toEqual([4, 5, 6]);
  });
});

describe('computePairwiseCohesion', () => {
  it('returns 1.0 for identical vectors with cosine metric', () => {
    const cohesion = computePairwiseCohesion([[1, 0], [1, 0], [1, 0]], 'cosine');
    expect(cohesion).toBeCloseTo(1.0);
  });

  it('returns 0.0 for orthogonal vectors with cosine metric', () => {
    const cohesion = computePairwiseCohesion([[1, 0], [0, 1]], 'cosine');
    expect(cohesion).toBeCloseTo(0.0);
  });

  it('returns 1.0 for single-member', () => {
    expect(computePairwiseCohesion([[1, 2, 3]], 'cosine')).toBe(1.0);
  });
});

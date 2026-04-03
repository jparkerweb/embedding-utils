import { describe, it, expect } from 'vitest';
import { computeScore, computeDistance } from '../../src/internal/metrics';
import { ValidationError } from '../../src/types';

describe('computeScore', () => {
  const a = [1, 0, 0];
  const b = [0, 1, 0];
  const same = [1, 0, 0];

  it('computes cosine similarity', () => {
    expect(computeScore(a, same, 'cosine')).toBeCloseTo(1.0);
    expect(computeScore(a, b, 'cosine')).toBeCloseTo(0.0);
  });

  it('computes dot product', () => {
    expect(computeScore([1, 2, 3], [4, 5, 6], 'dot')).toBe(32);
  });

  it('computes euclidean similarity', () => {
    const score = computeScore(a, same, 'euclidean');
    expect(score).toBeCloseTo(1.0); // 1 / (1 + 0) = 1
    const score2 = computeScore([0, 0], [3, 4], 'euclidean');
    expect(score2).toBeCloseTo(1 / (1 + 5)); // distance = 5
  });

  it('computes manhattan similarity', () => {
    const score = computeScore([0, 0], [3, 4], 'manhattan');
    expect(score).toBeCloseTo(1 / (1 + 7)); // distance = 7
  });

  it('throws ValidationError for unknown metric', () => {
    expect(() => computeScore(a, b, 'invalid' as any)).toThrow(ValidationError);
    expect(() => computeScore(a, b, 'invalid' as any)).toThrow('Unknown similarity metric');
  });
});

describe('computeDistance', () => {
  const a = [1, 0, 0];
  const b = [0, 1, 0];

  it('computes cosine distance', () => {
    expect(computeDistance(a, a, 'cosine')).toBeCloseTo(0.0);
    expect(computeDistance(a, b, 'cosine')).toBeCloseTo(1.0);
  });

  it('computes dot distance', () => {
    expect(computeDistance([1, 0], [1, 0], 'dot')).toBeCloseTo(0.0);
  });

  it('computes euclidean distance', () => {
    expect(computeDistance([0, 0], [3, 4], 'euclidean')).toBeCloseTo(5.0);
  });

  it('computes manhattan distance', () => {
    expect(computeDistance([0, 0], [3, 4], 'manhattan')).toBe(7);
  });

  it('throws ValidationError for unknown metric', () => {
    expect(() => computeDistance(a, b, 'invalid' as any)).toThrow(ValidationError);
  });
});

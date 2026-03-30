import { describe, it, expect } from 'vitest';
import { euclideanDistance, manhattanDistance } from '../../src/math/distance';
import { ValidationError } from '../../src/types';

describe('euclideanDistance', () => {
  it('returns 0 for identical vectors', () => {
    expect(euclideanDistance([1, 2, 3], [1, 2, 3])).toBe(0);
  });

  it('computes known numeric value', () => {
    // sqrt((4-1)^2 + (6-2)^2) = sqrt(9+16) = 5
    expect(euclideanDistance([1, 2], [4, 6])).toBeCloseTo(5, 10);
  });

  it('works with single-element vectors', () => {
    expect(euclideanDistance([3], [7])).toBeCloseTo(4, 10);
  });

  it('throws ValidationError for empty vectors', () => {
    expect(() => euclideanDistance([], [])).toThrow(ValidationError);
  });

  it('throws ValidationError for mismatched dimensions', () => {
    expect(() => euclideanDistance([1, 2], [1])).toThrow(ValidationError);
  });
});

describe('manhattanDistance', () => {
  it('returns 0 for identical vectors', () => {
    expect(manhattanDistance([1, 2, 3], [1, 2, 3])).toBe(0);
  });

  it('computes known numeric value', () => {
    // |4-1| + |6-2| = 3 + 4 = 7
    expect(manhattanDistance([1, 2], [4, 6])).toBe(7);
  });

  it('works with single-element vectors', () => {
    expect(manhattanDistance([3], [7])).toBe(4);
  });

  it('throws ValidationError for empty vectors', () => {
    expect(() => manhattanDistance([], [])).toThrow(ValidationError);
  });

  it('throws ValidationError for mismatched dimensions', () => {
    expect(() => manhattanDistance([1, 2], [1])).toThrow(ValidationError);
  });
});

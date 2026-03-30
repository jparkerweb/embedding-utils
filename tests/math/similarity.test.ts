import { describe, it, expect } from 'vitest';
import { cosineSimilarity, dotProduct } from '../../src/math/similarity';
import { ValidationError } from '../../src/types';

describe('cosineSimilarity', () => {
  it('returns 1 for identical vectors', () => {
    expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBeCloseTo(1, 10);
  });

  it('returns 0 for orthogonal vectors', () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0, 10);
  });

  it('returns -1 for opposite vectors', () => {
    expect(cosineSimilarity([1, 2, 3], [-1, -2, -3])).toBeCloseTo(-1, 10);
  });

  it('computes known numeric value', () => {
    // a = [1, 2, 3], b = [4, 5, 6]
    // dot = 4+10+18 = 32
    // |a| = sqrt(14), |b| = sqrt(77)
    // cos = 32 / sqrt(14*77) = 32 / sqrt(1078)
    const expected = 32 / Math.sqrt(1078);
    expect(cosineSimilarity([1, 2, 3], [4, 5, 6])).toBeCloseTo(expected, 10);
  });

  it('throws ValidationError for empty vectors', () => {
    expect(() => cosineSimilarity([], [])).toThrow(ValidationError);
  });

  it('throws ValidationError for mismatched dimensions', () => {
    expect(() => cosineSimilarity([1, 2], [1, 2, 3])).toThrow(ValidationError);
  });
});

describe('dotProduct', () => {
  it('computes known values', () => {
    // 1*4 + 2*5 + 3*6 = 32
    expect(dotProduct([1, 2, 3], [4, 5, 6])).toBe(32);
  });

  it('returns 0 for zero vector', () => {
    expect(dotProduct([0, 0, 0], [1, 2, 3])).toBe(0);
  });

  it('works with single-element vectors', () => {
    expect(dotProduct([3], [7])).toBe(21);
  });

  it('throws ValidationError for empty vectors', () => {
    expect(() => dotProduct([], [])).toThrow(ValidationError);
  });

  it('throws ValidationError for mismatched dimensions', () => {
    expect(() => dotProduct([1], [1, 2])).toThrow(ValidationError);
  });
});

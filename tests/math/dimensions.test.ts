import { describe, it, expect } from 'vitest';
import { truncateDimensions, validateDimensions } from '../../src/math/dimensions';
import { ValidationError } from '../../src/types';

describe('truncateDimensions (single vector)', () => {
  it('truncates to smaller size', () => {
    expect(truncateDimensions([1, 2, 3, 4, 5], 3)).toEqual([1, 2, 3]);
  });

  it('returns same vector when target equals current size', () => {
    expect(truncateDimensions([1, 2, 3], 3)).toEqual([1, 2, 3]);
  });

  it('throws ValidationError when target > current length', () => {
    expect(() => truncateDimensions([1, 2], 5)).toThrow(ValidationError);
  });

  it('works with single-element vector', () => {
    expect(truncateDimensions([42], 1)).toEqual([42]);
  });

  it('throws ValidationError for targetDims <= 0', () => {
    expect(() => truncateDimensions([1, 2, 3], 0)).toThrow(ValidationError);
  });
});

describe('truncateDimensions (batch)', () => {
  it('truncates all vectors in batch', () => {
    const batch = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ];
    const result = truncateDimensions(batch, 2);
    expect(result).toEqual([
      [1, 2],
      [5, 6],
    ]);
  });

  it('returns same vectors when target equals current size', () => {
    const batch = [
      [1, 2],
      [3, 4],
    ];
    expect(truncateDimensions(batch, 2)).toEqual(batch);
  });
});

describe('validateDimensions', () => {
  it('returns valid for consistent dimensions', () => {
    const result = validateDimensions([[1, 2], [3, 4], [5, 6]]);
    expect(result.valid).toBe(true);
    expect(result.dimension).toBe(2);
    expect(result.mismatches).toEqual([]);
  });

  it('returns invalid with correct mismatch indices', () => {
    const result = validateDimensions([[1, 2], [3, 4, 5], [6, 7]]);
    expect(result.valid).toBe(false);
    expect(result.dimension).toBe(2);
    expect(result.mismatches).toEqual([1]);
  });

  it('validates against expectedDim', () => {
    const result = validateDimensions([[1, 2], [3, 4]], 3);
    expect(result.valid).toBe(false);
    expect(result.dimension).toBe(3);
    expect(result.mismatches).toEqual([0, 1]);
  });

  it('returns valid for empty array', () => {
    const result = validateDimensions([]);
    expect(result.valid).toBe(true);
    expect(result.mismatches).toEqual([]);
  });

  it('uses expectedDim for empty array when provided', () => {
    const result = validateDimensions([], 5);
    expect(result.dimension).toBe(5);
  });

  it('detects multiple mismatches', () => {
    const result = validateDimensions([[1, 2], [3], [4, 5], [6]]);
    expect(result.valid).toBe(false);
    expect(result.mismatches).toEqual([1, 3]);
  });
});

import { describe, it, expect } from 'vitest';
import { truncateDimensions, validateDimensions } from '../../src/math/dimensions';
import { ValidationError } from '../../src/types';

describe('truncateDimensions (single vector)', () => {
  it('truncates to smaller size', () => {
    const result = truncateDimensions([1, 2, 3, 4, 5], 3);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([1, 2, 3]);
  });

  it('returns Float32Array when target equals current size', () => {
    const result = truncateDimensions([1, 2, 3], 3);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([1, 2, 3]);
  });

  it('throws ValidationError when target > current length', () => {
    expect(() => truncateDimensions([1, 2], 5)).toThrow(ValidationError);
  });

  it('works with single-element vector', () => {
    const result = truncateDimensions([42], 1);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([42]);
  });

  it('throws ValidationError for targetDims <= 0', () => {
    expect(() => truncateDimensions([1, 2, 3], 0)).toThrow(ValidationError);
  });

  it('accepts Float32Array input', () => {
    const result = truncateDimensions(new Float32Array([1, 2, 3, 4]), 2);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([1, 2]);
  });

  it('returns same Float32Array reference when target equals length', () => {
    const input = new Float32Array([1, 2, 3]);
    const result = truncateDimensions(input, 3);
    expect(result).toBe(input);
  });
});

describe('truncateDimensions (batch)', () => {
  it('truncates all vectors in batch', () => {
    const batch = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ];
    const result = truncateDimensions(batch, 2);
    expect(result).toHaveLength(2);
    expect(result[0]).toBeInstanceOf(Float32Array);
    expect(result[1]).toBeInstanceOf(Float32Array);
    expect(Array.from(result[0])).toEqual([1, 2]);
    expect(Array.from(result[1])).toEqual([5, 6]);
  });

  it('returns Float32Arrays when target equals current size', () => {
    const batch = [
      [1, 2],
      [3, 4],
    ];
    const result = truncateDimensions(batch, 2);
    expect(result[0]).toBeInstanceOf(Float32Array);
    expect(Array.from(result[0])).toEqual([1, 2]);
    expect(Array.from(result[1])).toEqual([3, 4]);
  });

  it('accepts Float32Array batch', () => {
    const batch = [
      new Float32Array([1, 2, 3]),
      new Float32Array([4, 5, 6]),
    ];
    const result = truncateDimensions(batch, 2);
    expect(result).toHaveLength(2);
    expect(Array.from(result[0])).toEqual([1, 2]);
    expect(Array.from(result[1])).toEqual([4, 5]);
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

  it('accepts Float32Array inputs', () => {
    const result = validateDimensions([new Float32Array([1, 2]), new Float32Array([3, 4])]);
    expect(result.valid).toBe(true);
    expect(result.dimension).toBe(2);
  });
});

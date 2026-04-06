import { describe, it, expect } from 'vitest';
import { truncateDimensions, validateDimensions } from '../../src/math/dimensions';
import { magnitude } from '../../src/math/vector';
import { ValidationError } from '../../src/types';

describe('truncateDimensions (single vector)', () => {
  it('truncates to smaller size and normalizes', () => {
    const result = truncateDimensions([1, 2, 3, 4, 5], 3);
    expect(result).toBeInstanceOf(Float32Array);
    expect(result).toHaveLength(3);
    expect(magnitude(result)).toBeCloseTo(1.0, 5);
  });

  it('returns normalized Float32Array when target equals current size', () => {
    const result = truncateDimensions([1, 2, 3], 3);
    expect(result).toBeInstanceOf(Float32Array);
    expect(result).toHaveLength(3);
    expect(magnitude(result)).toBeCloseTo(1.0, 5);
  });

  it('throws ValidationError when target > current length', () => {
    expect(() => truncateDimensions([1, 2], 5)).toThrow(ValidationError);
  });

  it('works with single-element vector', () => {
    const result = truncateDimensions([42], 1);
    expect(result).toBeInstanceOf(Float32Array);
    expect(magnitude(result)).toBeCloseTo(1.0, 5);
  });

  it('throws ValidationError for targetDims <= 0', () => {
    expect(() => truncateDimensions([1, 2, 3], 0)).toThrow(ValidationError);
  });

  it('accepts Float32Array input', () => {
    const result = truncateDimensions(new Float32Array([1, 2, 3, 4]), 2);
    expect(result).toBeInstanceOf(Float32Array);
    expect(result).toHaveLength(2);
    expect(magnitude(result)).toBeCloseTo(1.0, 5);
  });

  it('returns normalized Float32Array when target equals length', () => {
    const input = new Float32Array([3, 4]);
    const result = truncateDimensions(input, 2);
    expect(result).toBeInstanceOf(Float32Array);
    expect(magnitude(result)).toBeCloseTo(1.0, 5);
  });
});

describe('truncateDimensions (batch)', () => {
  it('truncates and normalizes all vectors in batch', () => {
    const batch = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ];
    const result = truncateDimensions(batch, 2);
    expect(result).toHaveLength(2);
    expect(result[0]).toBeInstanceOf(Float32Array);
    expect(result[1]).toBeInstanceOf(Float32Array);
    expect(result[0]).toHaveLength(2);
    expect(result[1]).toHaveLength(2);
    expect(magnitude(result[0])).toBeCloseTo(1.0, 5);
    expect(magnitude(result[1])).toBeCloseTo(1.0, 5);
  });

  it('returns normalized Float32Arrays when target equals current size', () => {
    const batch = [
      [3, 4],
      [5, 12],
    ];
    const result = truncateDimensions(batch, 2);
    expect(result[0]).toBeInstanceOf(Float32Array);
    expect(magnitude(result[0])).toBeCloseTo(1.0, 5);
    expect(magnitude(result[1])).toBeCloseTo(1.0, 5);
  });

  it('accepts Float32Array batch', () => {
    const batch = [
      new Float32Array([1, 2, 3]),
      new Float32Array([4, 5, 6]),
    ];
    const result = truncateDimensions(batch, 2);
    expect(result).toHaveLength(2);
    expect(result[0]).toHaveLength(2);
    expect(result[1]).toHaveLength(2);
    expect(magnitude(result[0])).toBeCloseTo(1.0, 5);
    expect(magnitude(result[1])).toBeCloseTo(1.0, 5);
  });
});

describe('truncateDimensions auto-normalize', () => {
  it('output has L2 norm approximately 1.0', () => {
    const result = truncateDimensions([3, 4, 5, 6, 7], 3);
    expect(magnitude(result)).toBeCloseTo(1.0, 5);
  });

  it('normalizes vectors of various magnitudes', () => {
    const vectors = [
      [100, 200, 300, 400],
      [0.001, 0.002, 0.003, 0.004],
      [1, 1, 1, 1],
    ];
    for (const v of vectors) {
      const result = truncateDimensions(v, 2);
      expect(magnitude(result)).toBeCloseTo(1.0, 5);
    }
  });

  it('normalizes Float32Array input', () => {
    const result = truncateDimensions(new Float32Array([10, 20, 30, 40]), 3);
    expect(result).toBeInstanceOf(Float32Array);
    expect(magnitude(result)).toBeCloseTo(1.0, 5);
  });

  it('normalizes batch of vectors', () => {
    const batch = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ];
    const results = truncateDimensions(batch, 2);
    for (const r of results) {
      expect(magnitude(r)).toBeCloseTo(1.0, 5);
    }
  });

  it('produces valid cosine similarity results after truncation', () => {
    // Two similar vectors should remain similar after truncate+normalize
    const a = [1, 2, 3, 0.1, 0.2];
    const b = [1.1, 2.1, 3.1, 0.5, 0.6];
    const truncA = truncateDimensions(a, 3);
    const truncB = truncateDimensions(b, 3);
    // Both normalized, so dot product = cosine similarity
    let dot = 0;
    for (let i = 0; i < truncA.length; i++) dot += truncA[i] * truncB[i];
    expect(dot).toBeGreaterThan(0.99); // very similar vectors
    expect(dot).toBeLessThanOrEqual(1.0 + 1e-6);
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

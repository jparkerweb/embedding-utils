import { describe, it, expect } from 'vitest';
import { normalize, magnitude, add, subtract, scale, isNormalized } from '../../src/math/vector';
import { ValidationError } from '../../src/types';

describe('magnitude', () => {
  it('computes known value', () => {
    // sqrt(9+16) = 5
    expect(magnitude([3, 4])).toBeCloseTo(5, 10);
  });

  it('returns 0 for zero vector', () => {
    expect(magnitude([0, 0, 0])).toBe(0);
  });

  it('throws ValidationError for empty vector', () => {
    expect(() => magnitude([])).toThrow(ValidationError);
  });

  it('accepts Float32Array input', () => {
    expect(magnitude(new Float32Array([3, 4]))).toBeCloseTo(5, 5);
  });
});

describe('normalize', () => {
  it('produces unit vector (magnitude ~1)', () => {
    const result = normalize([3, 4]);
    expect(result).toBeInstanceOf(Float32Array);
    const mag = Math.sqrt(result[0] ** 2 + result[1] ** 2);
    expect(mag).toBeCloseTo(1, 5);
  });

  it('normalizes known vector correctly', () => {
    const result = normalize([3, 4]);
    expect(result[0]).toBeCloseTo(3 / 5, 5);
    expect(result[1]).toBeCloseTo(4 / 5, 5);
  });

  it('returns zero Float32Array for zero input', () => {
    const result = normalize([0, 0, 0]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([0, 0, 0]);
  });

  it('throws ValidationError for empty vector', () => {
    expect(() => normalize([])).toThrow(ValidationError);
  });

  it('accepts Float32Array input', () => {
    const result = normalize(new Float32Array([3, 4]));
    expect(result).toBeInstanceOf(Float32Array);
    expect(result[0]).toBeCloseTo(3 / 5, 5);
  });
});

describe('add', () => {
  it('adds element-wise correctly', () => {
    const result = add([1, 2, 3], [4, 5, 6]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([5, 7, 9]);
  });

  it('throws ValidationError for dimension mismatch', () => {
    expect(() => add([1, 2], [1])).toThrow(ValidationError);
  });

  it('throws ValidationError for empty vectors', () => {
    expect(() => add([], [])).toThrow(ValidationError);
  });

  it('accepts mixed Vector inputs', () => {
    const result = add(new Float32Array([1, 2]), [3, 4]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([4, 6]);
  });
});

describe('subtract', () => {
  it('subtracts element-wise correctly', () => {
    const result = subtract([4, 5, 6], [1, 2, 3]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([3, 3, 3]);
  });

  it('throws ValidationError for dimension mismatch', () => {
    expect(() => subtract([1, 2], [1])).toThrow(ValidationError);
  });

  it('throws ValidationError for empty vectors', () => {
    expect(() => subtract([], [])).toThrow(ValidationError);
  });
});

describe('scale', () => {
  it('scales by positive factor', () => {
    const result = scale([1, 2, 3], 2);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([2, 4, 6]);
  });

  it('scales by 0', () => {
    const result = scale([1, 2, 3], 0);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([0, 0, 0]);
  });

  it('scales by 1 (identity)', () => {
    const result = scale([1, 2, 3], 1);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([1, 2, 3]);
  });

  it('scales by negative factor', () => {
    const result = scale([1, 2, 3], -1);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([-1, -2, -3]);
  });

  it('throws ValidationError for empty vector', () => {
    expect(() => scale([], 2)).toThrow(ValidationError);
  });
});

describe('isNormalized', () => {
  it('returns true for a unit vector', () => {
    expect(isNormalized([0.6, 0.8])).toBe(true);
  });

  it('returns true for [1, 0, 0]', () => {
    expect(isNormalized([1, 0, 0])).toBe(true);
  });

  it('returns false for non-unit vector', () => {
    expect(isNormalized([1, 1, 1])).toBe(false);
  });

  it('respects custom tolerance', () => {
    // magnitude of [1, 0.001] ≈ 1.0000005, within 0.01 tolerance
    expect(isNormalized([1, 0.001], 0.01)).toBe(true);
    // but not within 1e-10 tolerance
    expect(isNormalized([1, 0.001], 1e-10)).toBe(false);
  });

  it('returns false for zero vector', () => {
    expect(isNormalized([0, 0, 0])).toBe(false);
  });

  it('throws ValidationError for empty vector', () => {
    expect(() => isNormalized([])).toThrow(ValidationError);
  });

  it('accepts Float32Array input', () => {
    expect(isNormalized(new Float32Array([0.6, 0.8]))).toBe(true);
  });
});

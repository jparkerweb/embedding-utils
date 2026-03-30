import { describe, it, expect } from 'vitest';
import { normalize, magnitude, add, subtract, scale } from '../../src/math/vector';
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
});

describe('normalize', () => {
  it('produces unit vector (magnitude ~1)', () => {
    const result = normalize([3, 4]);
    const mag = Math.sqrt(result[0] ** 2 + result[1] ** 2);
    expect(mag).toBeCloseTo(1, 10);
  });

  it('normalizes known vector correctly', () => {
    const result = normalize([3, 4]);
    expect(result[0]).toBeCloseTo(3 / 5, 10);
    expect(result[1]).toBeCloseTo(4 / 5, 10);
  });

  it('returns zero vector for zero input', () => {
    const result = normalize([0, 0, 0]);
    expect(result).toEqual([0, 0, 0]);
  });

  it('throws ValidationError for empty vector', () => {
    expect(() => normalize([])).toThrow(ValidationError);
  });
});

describe('add', () => {
  it('adds element-wise correctly', () => {
    expect(add([1, 2, 3], [4, 5, 6])).toEqual([5, 7, 9]);
  });

  it('throws ValidationError for dimension mismatch', () => {
    expect(() => add([1, 2], [1])).toThrow(ValidationError);
  });

  it('throws ValidationError for empty vectors', () => {
    expect(() => add([], [])).toThrow(ValidationError);
  });
});

describe('subtract', () => {
  it('subtracts element-wise correctly', () => {
    expect(subtract([4, 5, 6], [1, 2, 3])).toEqual([3, 3, 3]);
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
    expect(scale([1, 2, 3], 2)).toEqual([2, 4, 6]);
  });

  it('scales by 0', () => {
    expect(scale([1, 2, 3], 0)).toEqual([0, 0, 0]);
  });

  it('scales by 1 (identity)', () => {
    expect(scale([1, 2, 3], 1)).toEqual([1, 2, 3]);
  });

  it('scales by negative factor', () => {
    expect(scale([1, 2, 3], -1)).toEqual([-1, -2, -3]);
  });

  it('throws ValidationError for empty vector', () => {
    expect(() => scale([], 2)).toThrow(ValidationError);
  });
});

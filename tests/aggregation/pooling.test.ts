import { describe, it, expect } from 'vitest';
import { maxPooling, minPooling } from '../../src/aggregation/pooling';
import { ValidationError } from '../../src/types';

describe('maxPooling', () => {
  it('returns element-wise max across vectors', () => {
    const result = maxPooling([
      [1, 5, 3],
      [4, 2, 6],
      [3, 8, 1],
    ]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([4, 8, 6]);
  });

  it('returns the vector itself for a single input', () => {
    const result = maxPooling([[7, 3, 9]]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([7, 3, 9]);
  });

  it('throws ValidationError for empty array', () => {
    expect(() => maxPooling([])).toThrow(ValidationError);
  });

  it('throws ValidationError for mixed dimensions', () => {
    expect(() => maxPooling([[1, 2], [1, 2, 3]])).toThrow(ValidationError);
  });

  it('handles negative values correctly', () => {
    const result = maxPooling([
      [-5, -1],
      [-3, -4],
    ]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([-3, -1]);
  });

  it('accepts Float32Array inputs', () => {
    const result = maxPooling([new Float32Array([1, 5]), new Float32Array([4, 2])]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([4, 5]);
  });
});

describe('minPooling', () => {
  it('returns element-wise min across vectors', () => {
    const result = minPooling([
      [1, 5, 3],
      [4, 2, 6],
      [3, 8, 1],
    ]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([1, 2, 1]);
  });

  it('returns the vector itself for a single input', () => {
    const result = minPooling([[7, 3, 9]]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([7, 3, 9]);
  });

  it('throws ValidationError for empty array', () => {
    expect(() => minPooling([])).toThrow(ValidationError);
  });

  it('throws ValidationError for mixed dimensions', () => {
    expect(() => minPooling([[1, 2], [1, 2, 3]])).toThrow(ValidationError);
  });

  it('handles negative values correctly', () => {
    const result = minPooling([
      [-5, -1],
      [-3, -4],
    ]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([-5, -4]);
  });
});

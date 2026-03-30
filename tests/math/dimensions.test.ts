import { describe, it, expect } from 'vitest';
import { truncateDimensions } from '../../src/math/dimensions';
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

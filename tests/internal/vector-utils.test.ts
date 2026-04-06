import { describe, it, expect } from 'vitest';
import { toFloat32, isVector } from '../../src/internal/vector-utils';

describe('toFloat32', () => {
  it('converts number[] to Float32Array', () => {
    const input = [1, 2, 3];
    const result = toFloat32(input);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([1, 2, 3]);
  });

  it('returns same reference for Float32Array input', () => {
    const input = new Float32Array([1, 2, 3]);
    const result = toFloat32(input);
    expect(result).toBe(input);
  });

  it('handles empty array', () => {
    const result = toFloat32([]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(0);
  });

  it('preserves values accurately', () => {
    const input = [0.1, 0.2, 0.3, -0.5, 1.0];
    const result = toFloat32(input);
    expect(result).toBeInstanceOf(Float32Array);
    for (let i = 0; i < input.length; i++) {
      expect(result[i]).toBeCloseTo(input[i], 6);
    }
  });

  it('handles empty Float32Array', () => {
    const input = new Float32Array([]);
    const result = toFloat32(input);
    expect(result).toBe(input);
    expect(result.length).toBe(0);
  });
});

describe('isVector', () => {
  it('returns true for number[]', () => {
    expect(isVector([1, 2, 3])).toBe(true);
  });

  it('returns true for empty number[]', () => {
    expect(isVector([])).toBe(true);
  });

  it('returns true for Float32Array', () => {
    expect(isVector(new Float32Array([1, 2, 3]))).toBe(true);
  });

  it('returns true for empty Float32Array', () => {
    expect(isVector(new Float32Array([]))).toBe(true);
  });

  it('returns false for string', () => {
    expect(isVector('hello')).toBe(false);
  });

  it('returns false for null', () => {
    expect(isVector(null)).toBe(false);
  });

  it('returns false for undefined', () => {
    expect(isVector(undefined)).toBe(false);
  });

  it('returns false for plain object', () => {
    expect(isVector({ 0: 1, 1: 2 })).toBe(false);
  });

  it('returns false for Set', () => {
    expect(isVector(new Set([1, 2, 3]))).toBe(false);
  });

  it('returns false for array with non-number elements', () => {
    expect(isVector([1, 'two', 3])).toBe(false);
  });
});

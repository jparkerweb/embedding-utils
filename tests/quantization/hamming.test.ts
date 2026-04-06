import { describe, it, expect } from 'vitest';
import { hammingDistance, hammingSimilarity } from '../../src/quantization/hamming';
import { ValidationError } from '../../src/types';

describe('hammingDistance', () => {
  it('computes distance for known bit patterns', () => {
    // 0xFF = 11111111, 0x00 = 00000000 → 8 bits differ per byte
    const a = new Uint8Array([0xff, 0x00]);
    const b = new Uint8Array([0x00, 0xff]);
    expect(hammingDistance(a, b)).toBe(16);
  });

  it('returns 0 for identical vectors', () => {
    const a = new Uint8Array([0xab, 0xcd, 0xef]);
    expect(hammingDistance(a, a)).toBe(0);
  });

  it('returns 0 for empty vectors', () => {
    const a = new Uint8Array([]);
    const b = new Uint8Array([]);
    expect(hammingDistance(a, b)).toBe(0);
  });

  it('throws ValidationError for mismatched lengths', () => {
    const a = new Uint8Array([0xff]);
    const b = new Uint8Array([0xff, 0x00]);
    expect(() => hammingDistance(a, b)).toThrow(ValidationError);
  });

  it('counts individual bit differences', () => {
    // 0b10101010 vs 0b10101011 → 1 bit differs
    const a = new Uint8Array([0xaa]);
    const b = new Uint8Array([0xab]);
    expect(hammingDistance(a, b)).toBe(1);
  });
});

describe('hammingSimilarity', () => {
  it('returns 1 for identical vectors', () => {
    const a = new Uint8Array([0xab, 0xcd]);
    expect(hammingSimilarity(a, a, 16)).toBe(1);
  });

  it('returns 1 - (distance / totalBits)', () => {
    const a = new Uint8Array([0xff, 0x00]);
    const b = new Uint8Array([0x00, 0xff]);
    // distance = 16, totalBits = 16 → similarity = 0
    expect(hammingSimilarity(a, b, 16)).toBe(0);
  });

  it('computes correct similarity for partial difference', () => {
    // 1 bit differs out of 8
    const a = new Uint8Array([0xaa]); // 10101010
    const b = new Uint8Array([0xab]); // 10101011
    expect(hammingSimilarity(a, b, 8)).toBeCloseTo(7 / 8, 10);
  });

  it('throws ValidationError for mismatched lengths', () => {
    const a = new Uint8Array([0xff]);
    const b = new Uint8Array([0xff, 0x00]);
    expect(() => hammingSimilarity(a, b, 16)).toThrow(ValidationError);
  });
});

import { describe, it, expect } from 'vitest';
import { estimateMemorySavings } from '../../src/quantization/info';

describe('estimateMemorySavings', () => {
  const dims = 384;
  const count = 10000;
  const originalBytes = count * dims * 4; // 15,360,000

  it('calculates fp16 (stored as float32, no savings)', () => {
    const result = estimateMemorySavings(dims, count, 'fp16');
    expect(result.originalBytes).toBe(originalBytes);
    expect(result.quantizedBytes).toBe(originalBytes);
    expect(result.savings).toBe(0);
    expect(result.ratio).toBe(1);
  });

  it('calculates int8 savings (4x reduction)', () => {
    const result = estimateMemorySavings(dims, count, 'int8');
    expect(result.originalBytes).toBe(originalBytes);
    expect(result.quantizedBytes).toBe(count * dims); // 3,840,000
    expect(result.savings).toBe(originalBytes - count * dims);
    expect(result.ratio).toBe(0.25);
  });

  it('calculates uint8 savings (4x reduction)', () => {
    const result = estimateMemorySavings(dims, count, 'uint8');
    expect(result.originalBytes).toBe(originalBytes);
    expect(result.quantizedBytes).toBe(count * dims);
    expect(result.savings).toBe(originalBytes - count * dims);
    expect(result.ratio).toBe(0.25);
  });

  it('calculates binary savings (32x reduction)', () => {
    const result = estimateMemorySavings(dims, count, 'binary');
    const expectedQuantized = count * Math.ceil(dims / 8); // 480,000
    expect(result.originalBytes).toBe(originalBytes);
    expect(result.quantizedBytes).toBe(expectedQuantized);
    expect(result.savings).toBe(originalBytes - expectedQuantized);
    expect(result.ratio).toBeCloseTo(expectedQuantized / originalBytes, 6);
  });

  it('handles non-byte-aligned dimensions for binary', () => {
    // 10 dims → ceil(10/8) = 2 bytes per vector
    const result = estimateMemorySavings(10, 100, 'binary');
    expect(result.quantizedBytes).toBe(100 * 2);
  });

  it('handles single embedding', () => {
    const result = estimateMemorySavings(768, 1, 'int8');
    expect(result.originalBytes).toBe(768 * 4);
    expect(result.quantizedBytes).toBe(768);
  });
});

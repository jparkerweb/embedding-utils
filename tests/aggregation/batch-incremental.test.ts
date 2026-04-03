import { describe, it, expect } from 'vitest';
import {
  batchIncrementalAverage,
  averageEmbeddings,
} from '../../src/aggregation/average';
import { ValidationError } from '../../src/types';

describe('batchIncrementalAverage', () => {
  it('produces result equivalent to full re-average for a batch of 10 vectors', () => {
    const vectors = Array.from({ length: 10 }, (_, i) => [
      Math.sin(i),
      Math.cos(i),
      i * 0.1,
    ]);
    const fullAvg = averageEmbeddings(vectors);

    // Incrementally: start with first vector, then batch the remaining 9
    let currentAvg = vectors[0];
    currentAvg = batchIncrementalAverage(currentAvg, vectors.slice(1), 1);

    for (let d = 0; d < fullAvg.length; d++) {
      expect(currentAvg[d]).toBeCloseTo(fullAvg[d], 10);
    }
  });

  it('produces equivalent result when batching in groups of 3', () => {
    const vectors = Array.from({ length: 9 }, (_, i) => [i + 1, (i + 1) * 2, (i + 1) * 3]);
    const fullAvg = averageEmbeddings(vectors);

    let currentAvg = averageEmbeddings(vectors.slice(0, 3));
    currentAvg = batchIncrementalAverage(currentAvg, vectors.slice(3, 6), 3);
    currentAvg = batchIncrementalAverage(currentAvg, vectors.slice(6, 9), 6);

    for (let d = 0; d < fullAvg.length; d++) {
      expect(currentAvg[d]).toBeCloseTo(fullAvg[d], 10);
    }
  });

  it('throws ValidationError for empty newEmbeddings array', () => {
    expect(() => batchIncrementalAverage([1, 2, 3], [], 5)).toThrow(ValidationError);
  });

  it('throws ValidationError for dimension mismatch across batch items', () => {
    expect(() =>
      batchIncrementalAverage([1, 2, 3], [[4, 5, 6], [7, 8]], 1),
    ).toThrow(ValidationError);
  });

  it('throws ValidationError when batch item dimension mismatches currentAvg', () => {
    expect(() =>
      batchIncrementalAverage([1, 2], [[3, 4, 5]], 1),
    ).toThrow(ValidationError);
  });

  it('returns the blended value for a single-item batch', () => {
    const currentAvg = [2, 4, 6];
    const result = batchIncrementalAverage(currentAvg, [[4, 8, 12]], 1);
    // (2*1 + 4) / 2 = 3, (4*1 + 8) / 2 = 6, (6*1 + 12) / 2 = 9
    expect(result).toEqual([3, 6, 9]);
  });

  it('with count=0, single-item batch returns that item itself', () => {
    const item = [7, 8, 9];
    // count=0 means currentAvg has no weight; result = (0*currentAvg + item) / 1 = item
    const result = batchIncrementalAverage([0, 0, 0], [item], 0);
    expect(result).toEqual(item);
  });
});

import { describe, it, expect } from 'vitest';
import {
  averageEmbeddings,
  weightedAverage,
  incrementalAverage,
  centroid,
} from '../../src/aggregation/average';
import { ValidationError } from '../../src/types';

describe('averageEmbeddings', () => {
  it('returns the same vector for two identical vectors', () => {
    const v = [1, 2, 3];
    const result = averageEmbeddings([v, v]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([1, 2, 3]);
  });

  it('returns zero vector for two opposing vectors', () => {
    const result = averageEmbeddings([
      [1, -1, 0.5],
      [-1, 1, -0.5],
    ]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([0, 0, 0]);
  });

  it('computes correct mean for known numeric values', () => {
    const result = averageEmbeddings([
      [2, 4, 6],
      [4, 8, 12],
      [6, 12, 18],
    ]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([4, 8, 12]);
  });

  it('throws ValidationError for empty array', () => {
    expect(() => averageEmbeddings([])).toThrow(ValidationError);
  });

  it('throws ValidationError for mixed dimensions', () => {
    expect(() => averageEmbeddings([[1, 2], [1, 2, 3]])).toThrow(ValidationError);
  });

  it('returns the vector itself for a single input', () => {
    const result = averageEmbeddings([[3, 6, 9]]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([3, 6, 9]);
  });

  it('accepts Float32Array inputs', () => {
    const result = averageEmbeddings([new Float32Array([1, 2]), new Float32Array([3, 4])]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([2, 3]);
  });

  it('accepts mixed number[] and Float32Array inputs', () => {
    const result = averageEmbeddings([[1, 2], new Float32Array([3, 4])]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([2, 3]);
  });
});

describe('weightedAverage', () => {
  it('matches simple average when weights are equal', () => {
    const embeddings = [
      [2, 4],
      [4, 8],
    ];
    const result = weightedAverage(embeddings, [1, 1]);
    const avg = averageEmbeddings(embeddings);
    expect(Array.from(result)).toEqual(Array.from(avg));
  });

  it('excludes zero-weighted embeddings', () => {
    const result = weightedAverage(
      [
        [10, 20],
        [1, 2],
      ],
      [0, 1],
    );
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([1, 2]);
  });

  it('computes correct weighted average for known values', () => {
    const result = weightedAverage(
      [
        [0, 0],
        [10, 10],
      ],
      [1, 3],
    );
    // (0*1 + 10*3) / (1+3) = 7.5
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([7.5, 7.5]);
  });

  it('throws ValidationError when weights/embeddings length mismatch', () => {
    expect(() => weightedAverage([[1, 2]], [1, 2])).toThrow(ValidationError);
  });

  it('throws ValidationError for empty inputs', () => {
    expect(() => weightedAverage([], [])).toThrow(ValidationError);
  });
});

describe('incrementalAverage', () => {
  it('matches batch average after N incremental updates', () => {
    const data = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ];

    let avg: number[] | Float32Array = data[0];
    for (let i = 1; i < data.length; i++) {
      avg = incrementalAverage(avg, data[i], i);
    }

    const batchAvg = averageEmbeddings(data);
    expect(avg).toBeInstanceOf(Float32Array);
    for (let i = 0; i < avg.length; i++) {
      expect(avg[i]).toBeCloseTo(batchAvg[i], 5);
    }
  });

  it('returns the new embedding when count is 0', () => {
    const result = incrementalAverage([0, 0, 0], [5, 10, 15], 0);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([5, 10, 15]);
  });

  it('throws ValidationError for dimension mismatch', () => {
    expect(() => incrementalAverage([1, 2], [1, 2, 3], 1)).toThrow(ValidationError);
  });
});

describe('centroid', () => {
  it('behaves the same as averageEmbeddings', () => {
    const embeddings = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ];
    const c = centroid(embeddings);
    const a = averageEmbeddings(embeddings);
    expect(Array.from(c)).toEqual(Array.from(a));
  });

  it('returns the vector itself for a single input', () => {
    const result = centroid([[5, 10, 15]]);
    expect(result).toBeInstanceOf(Float32Array);
    expect(Array.from(result)).toEqual([5, 10, 15]);
  });
});

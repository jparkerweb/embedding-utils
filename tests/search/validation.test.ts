import { describe, it, expect } from 'vitest';
import { topK } from '../../src/search/topk';
import { aboveThreshold } from '../../src/search/threshold';
import { similarityMatrix } from '../../src/search/matrix';
import { DimensionMismatchError, ValidationError } from '../../src/types';

describe('Search dimension validation', () => {
  const query3d = [1, 0, 0];
  const corpus2d = [[1, 0], [0, 1]];

  it('topK throws DimensionMismatchError for mismatched query/corpus', () => {
    expect(() => topK(query3d, corpus2d, 1)).toThrow(DimensionMismatchError);
    expect(() => topK(query3d, corpus2d, 1)).toThrow(/Query dimension \(3\) does not match corpus dimension \(2\)/);
  });

  it('aboveThreshold throws DimensionMismatchError for mismatched query/corpus', () => {
    expect(() => aboveThreshold(query3d, corpus2d, 0.5)).toThrow(DimensionMismatchError);
    expect(() => aboveThreshold(query3d, corpus2d, 0.5)).toThrow(/Query dimension \(3\) does not match corpus dimension \(2\)/);
  });

  it('similarityMatrix throws DimensionMismatchError for mixed dimensions', () => {
    const mixed = [[1, 0, 0], [1, 0]];
    expect(() => similarityMatrix(mixed)).toThrow(DimensionMismatchError);
  });

  it('topK works with empty corpus', () => {
    const results = topK([1, 0, 0], [], 1);
    expect(results).toHaveLength(0);
  });

  it('aboveThreshold works with empty corpus', () => {
    const results = aboveThreshold([1, 0, 0], [], 0.5);
    expect(results).toHaveLength(0);
  });
});

describe('Search parameter validation', () => {
  const query = [1, 0, 0];
  const corpus = [[1, 0, 0], [0, 1, 0]];

  it('topK throws ValidationError for k = 0', () => {
    expect(() => topK(query, corpus, 0)).toThrow(ValidationError);
    expect(() => topK(query, corpus, 0)).toThrow(/k must be a positive integer/);
  });

  it('topK throws ValidationError for negative k', () => {
    expect(() => topK(query, corpus, -1)).toThrow(ValidationError);
  });

  it('topK throws ValidationError for non-integer k', () => {
    expect(() => topK(query, corpus, 1.5)).toThrow(ValidationError);
  });

  it('topK accepts k = 1', () => {
    const results = topK(query, corpus, 1);
    expect(results).toHaveLength(1);
  });

  it('aboveThreshold throws ValidationError for Infinity threshold', () => {
    expect(() => aboveThreshold(query, corpus, Infinity)).toThrow(ValidationError);
    expect(() => aboveThreshold(query, corpus, Infinity)).toThrow(/threshold must be a finite number/);
  });

  it('aboveThreshold throws ValidationError for NaN threshold', () => {
    expect(() => aboveThreshold(query, corpus, NaN)).toThrow(ValidationError);
  });

  it('aboveThreshold accepts valid threshold', () => {
    const results = aboveThreshold(query, corpus, 0.5);
    expect(results).toBeDefined();
  });

  it('similarityMatrix throws ValidationError for empty array', () => {
    expect(() => similarityMatrix([])).toThrow(ValidationError);
  });
});

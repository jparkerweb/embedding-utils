import { describe, it, expect } from 'vitest';
import { validateVector, validateVectorPair, validateEmbeddings } from '../../src/internal/validation';
import { ValidationError, DimensionMismatchError } from '../../src/types';

describe('validateVector', () => {
  it('passes for valid input', () => {
    expect(() => validateVector([1, 2, 3])).not.toThrow();
  });

  it('throws for null', () => {
    expect(() => validateVector(null)).toThrow(ValidationError);
  });

  it('throws for undefined', () => {
    expect(() => validateVector(undefined)).toThrow(ValidationError);
  });

  it('throws for non-array', () => {
    expect(() => validateVector('not an array')).toThrow(ValidationError);
  });

  it('throws for empty array', () => {
    expect(() => validateVector([])).toThrow(ValidationError);
  });

  it('throws for NaN element', () => {
    expect(() => validateVector([1, NaN, 3])).toThrow(ValidationError);
    expect(() => validateVector([1, NaN, 3])).toThrow('finite number');
  });

  it('throws for Infinity element', () => {
    expect(() => validateVector([1, Infinity, 3])).toThrow(ValidationError);
  });

  it('throws for -Infinity element', () => {
    expect(() => validateVector([1, -Infinity, 3])).toThrow(ValidationError);
  });

  it('uses custom name in error message', () => {
    expect(() => validateVector(null, 'query')).toThrow('query');
  });
});

describe('validateVectorPair', () => {
  it('passes for matching vectors', () => {
    expect(() => validateVectorPair([1, 2], [3, 4])).not.toThrow();
  });

  it('throws for empty vectors', () => {
    expect(() => validateVectorPair([], [])).toThrow(ValidationError);
  });

  it('throws DimensionMismatchError for mismatched dimensions', () => {
    expect(() => validateVectorPair([1, 2], [1, 2, 3])).toThrow(DimensionMismatchError);
  });
});

describe('validateEmbeddings', () => {
  it('passes for valid embeddings', () => {
    expect(() => validateEmbeddings([[1, 2], [3, 4]])).not.toThrow();
  });

  it('throws for empty array', () => {
    expect(() => validateEmbeddings([])).toThrow(ValidationError);
  });

  it('throws DimensionMismatchError for mixed dimensions', () => {
    expect(() => validateEmbeddings([[1, 2], [1, 2, 3]])).toThrow(DimensionMismatchError);
  });
});

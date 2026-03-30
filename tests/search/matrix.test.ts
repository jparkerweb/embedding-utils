import { describe, it, expect } from 'vitest';
import { similarityMatrix } from '../../src/search/matrix';
import { ValidationError } from '../../src/types';
import { normalize } from '../../src/math/vector';

describe('similarityMatrix', () => {
  it('diagonal is 1.0 for cosine on normalized vectors', () => {
    const embeddings = [normalize([1, 2, 3]), normalize([4, 5, 6]), normalize([7, 8, 9])];
    const matrix = similarityMatrix(embeddings);
    for (let i = 0; i < embeddings.length; i++) {
      expect(matrix[i][i]).toBeCloseTo(1.0, 10);
    }
  });

  it('produces a symmetric matrix', () => {
    const embeddings = [
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
    ];
    const matrix = similarityMatrix(embeddings);
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix.length; j++) {
        expect(matrix[i][j]).toBeCloseTo(matrix[j][i], 10);
      }
    }
  });

  it('computes known 3x3 values', () => {
    const embeddings = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    const matrix = similarityMatrix(embeddings);
    // Orthogonal unit vectors: diagonal = 1, off-diagonal = 0
    expect(matrix[0][0]).toBeCloseTo(1);
    expect(matrix[0][1]).toBeCloseTo(0);
    expect(matrix[0][2]).toBeCloseTo(0);
    expect(matrix[1][1]).toBeCloseTo(1);
    expect(matrix[1][2]).toBeCloseTo(0);
    expect(matrix[2][2]).toBeCloseTo(1);
  });

  it('returns [[1]] for single embedding', () => {
    const matrix = similarityMatrix([[5, 3, 1]]);
    expect(matrix).toHaveLength(1);
    expect(matrix[0]).toHaveLength(1);
    expect(matrix[0][0]).toBeCloseTo(1);
  });

  it('supports custom metric', () => {
    const embeddings = [
      [1, 0],
      [0, 1],
    ];
    const matrix = similarityMatrix(embeddings, { metric: 'dot' });
    expect(matrix[0][0]).toBe(1);
    expect(matrix[0][1]).toBe(0);
  });

  it('throws ValidationError for empty array', () => {
    expect(() => similarityMatrix([])).toThrow(ValidationError);
  });
});

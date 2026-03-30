import { ValidationError } from '../types';

function validateEmbeddings(embeddings: number[][]): void {
  if (embeddings.length === 0) {
    throw new ValidationError('Embeddings array must be non-empty');
  }
  const dim = embeddings[0].length;
  for (let i = 1; i < embeddings.length; i++) {
    if (embeddings[i].length !== dim) {
      throw new ValidationError(
        `Dimension mismatch: expected ${dim}, got ${embeddings[i].length} at index ${i}`,
      );
    }
  }
}

/**
 * Applies max pooling across embeddings, taking the maximum value per dimension.
 * @param embeddings - Array of equal-dimension embedding vectors
 * @returns A single vector with the max value at each dimension
 * @throws {ValidationError} If the array is empty or dimensions mismatch
 * @example
 * maxPooling([[1, 0, 3], [2, 5, 1]]); // [2, 5, 3]
 */
export function maxPooling(embeddings: number[][]): number[] {
  validateEmbeddings(embeddings);
  const dim = embeddings[0].length;
  const result = [...embeddings[0]];
  for (let i = 1; i < embeddings.length; i++) {
    for (let j = 0; j < dim; j++) {
      if (embeddings[i][j] > result[j]) {
        result[j] = embeddings[i][j];
      }
    }
  }
  return result;
}

/**
 * Applies min pooling across embeddings, taking the minimum value per dimension.
 * @param embeddings - Array of equal-dimension embedding vectors
 * @returns A single vector with the min value at each dimension
 * @throws {ValidationError} If the array is empty or dimensions mismatch
 * @example
 * minPooling([[1, 0, 3], [2, 5, 1]]); // [1, 0, 1]
 */
export function minPooling(embeddings: number[][]): number[] {
  validateEmbeddings(embeddings);
  const dim = embeddings[0].length;
  const result = [...embeddings[0]];
  for (let i = 1; i < embeddings.length; i++) {
    for (let j = 0; j < dim; j++) {
      if (embeddings[i][j] < result[j]) {
        result[j] = embeddings[i][j];
      }
    }
  }
  return result;
}

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
 * Computes the element-wise average of multiple embeddings.
 * @param embeddings - Array of equal-dimension embedding vectors
 * @returns A single averaged embedding vector
 * @throws {ValidationError} If the array is empty or dimensions mismatch
 * @example
 * averageEmbeddings([[1, 0], [0, 1]]); // [0.5, 0.5]
 */
export function averageEmbeddings(embeddings: number[][]): number[] {
  validateEmbeddings(embeddings);
  const dim = embeddings[0].length;
  const n = embeddings.length;
  const result = new Array<number>(dim).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < dim; j++) {
      result[j] += embeddings[i][j];
    }
  }
  for (let j = 0; j < dim; j++) {
    result[j] /= n;
  }
  return result;
}

/**
 * Computes a weighted average of multiple embeddings.
 * @param embeddings - Array of equal-dimension embedding vectors
 * @param weights - Weight for each embedding (must match length of embeddings)
 * @returns A single weighted-average embedding vector
 * @throws {ValidationError} If arrays are empty, dimensions mismatch, or lengths differ
 * @example
 * weightedAverage([[1, 0], [0, 1]], [3, 1]); // [0.75, 0.25]
 */
export function weightedAverage(embeddings: number[][], weights: number[]): number[] {
  validateEmbeddings(embeddings);
  if (embeddings.length !== weights.length) {
    throw new ValidationError(
      `Embeddings length (${embeddings.length}) must match weights length (${weights.length})`,
    );
  }
  const dim = embeddings[0].length;
  const result = new Array<number>(dim).fill(0);
  let totalWeight = 0;
  for (let i = 0; i < embeddings.length; i++) {
    totalWeight += weights[i];
    for (let j = 0; j < dim; j++) {
      result[j] += embeddings[i][j] * weights[i];
    }
  }
  for (let j = 0; j < dim; j++) {
    result[j] /= totalWeight;
  }
  return result;
}

/**
 * Updates a running average with a new embedding using Welford's method.
 * @param currentAvg - The current average embedding
 * @param newEmbedding - The new embedding to incorporate
 * @param count - The number of embeddings already averaged (before this one)
 * @returns Updated average embedding
 * @throws {ValidationError} If vector dimensions mismatch
 * @example
 * incrementalAverage([0.5, 0.5], [1, 0], 2); // [0.667, 0.333]
 */
export function incrementalAverage(
  currentAvg: number[],
  newEmbedding: number[],
  count: number,
): number[] {
  if (currentAvg.length !== newEmbedding.length) {
    throw new ValidationError(
      `Dimension mismatch: ${currentAvg.length} vs ${newEmbedding.length}`,
    );
  }
  const dim = currentAvg.length;
  const result = new Array<number>(dim);
  const divisor = count + 1;
  for (let i = 0; i < dim; i++) {
    result[i] = currentAvg[i] + (newEmbedding[i] - currentAvg[i]) / divisor;
  }
  return result;
}

/**
 * Computes the centroid (mean) of a set of embeddings. Alias for {@link averageEmbeddings}.
 * @param embeddings - Array of equal-dimension embedding vectors
 * @returns The centroid embedding vector
 * @throws {ValidationError} If the array is empty or dimensions mismatch
 * @example
 * centroid([[1, 0], [0, 1]]); // [0.5, 0.5]
 */
export function centroid(embeddings: number[][]): number[] {
  return averageEmbeddings(embeddings);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pooling Operations
//
// Element-wise max and min across multiple embeddings. These capture extreme
// signals rather than averages, useful for preserving the strongest (or weakest)
// semantic features across a set of vectors.
// ─────────────────────────────────────────────────────────────────────────────

import type { Vector } from '../types';
import { validateEmbeddings } from '../internal/validation';

/**
 * Applies max pooling across embeddings, taking the maximum value per dimension.
 *
 * Captures the "strongest signal" across a set of embeddings. If any one
 * embedding in the set activates a dimension strongly, that signal is
 * preserved in the pooled result.
 *
 * @param embeddings - Array of equal-dimension embedding vectors
 * @returns A single vector with the max value at each dimension
 * @throws {ValidationError} If the array is empty or dimensions mismatch
 *
 * @example
 * maxPooling([[1, 0, 3], [2, 5, 1]]); // Float32Array [2, 5, 3]
 */
export function maxPooling(embeddings: Vector[]): Float32Array {
  validateEmbeddings(embeddings);
  const dim = embeddings[0].length;
  const result = new Float32Array(embeddings[0]);
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
 *
 * Captures the "weakest signal" across a set of embeddings. Useful for
 * detecting what's *missing* or consistently low across a set.
 *
 * @param embeddings - Array of equal-dimension embedding vectors
 * @returns A single vector with the min value at each dimension
 * @throws {ValidationError} If the array is empty or dimensions mismatch
 *
 * @example
 * minPooling([[1, 0, 3], [2, 5, 1]]); // Float32Array [1, 0, 1]
 */
export function minPooling(embeddings: Vector[]): Float32Array {
  validateEmbeddings(embeddings);
  const dim = embeddings[0].length;
  const result = new Float32Array(embeddings[0]);
  for (let i = 1; i < embeddings.length; i++) {
    for (let j = 0; j < dim; j++) {
      if (embeddings[i][j] < result[j]) {
        result[j] = embeddings[i][j];
      }
    }
  }
  return result;
}

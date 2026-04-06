// ─────────────────────────────────────────────────────────────────────────────
// Dimension Manipulation
//
// Functions for changing the dimensionality of embedding vectors. Currently
// supports Matryoshka-style truncation (keeping the first N dimensions).
// ─────────────────────────────────────────────────────────────────────────────

import { ValidationError } from '../types';
import type { Vector } from '../types';

/**
 * Truncates embedding(s) to a target number of dimensions (Matryoshka-style).
 *
 * Matryoshka representation learning trains models so that the first N
 * dimensions of a vector carry the most important information. This means
 * you can safely truncate from, say, 1536 dims to 256 dims with minimal
 * quality loss — saving 80%+ storage and speeding up similarity search.
 *
 * **Supported models:** OpenAI text-embedding-3-*, Cohere embed-v3+, and
 * other Matryoshka-trained models. Using this with non-Matryoshka models
 * will produce degraded results.
 *
 * **Overloaded:** Accepts either a single vector or a batch of vectors.
 * Always returns Float32Array output.
 *
 * @param input - A single embedding vector or array of embedding vectors
 * @param targetDims - The desired number of dimensions (must be > 0 and <= vector length)
 * @returns Truncated vector(s) as Float32Array
 * @throws {ValidationError} If targetDims is <= 0 or exceeds vector length
 *
 * @example
 * // Single vector
 * truncateDimensions([1, 2, 3, 4], 2); // Float32Array [1, 2]
 *
 * @example
 * // Batch of vectors
 * truncateDimensions([[1, 2, 3], [4, 5, 6]], 2); // [Float32Array [1, 2], Float32Array [4, 5]]
 *
 * @example
 * // Reduce OpenAI 1536-dim embeddings to 256 for cheaper storage
 * const compact = truncateDimensions(embeddings, 256);
 */
export function truncateDimensions(embedding: Vector, targetDims: number): Float32Array;
export function truncateDimensions(embeddings: Vector[], targetDims: number): Float32Array[];
export function truncateDimensions(
  input: Vector | Vector[],
  targetDims: number
): Float32Array | Float32Array[] {
  if (targetDims <= 0) {
    throw new ValidationError('targetDims must be greater than 0');
  }

  // Check if input is a batch (array of vectors)
  if (Array.isArray(input) && input.length > 0 && (Array.isArray(input[0]) || input[0] instanceof Float32Array)) {
    const batch = input as Vector[];
    return batch.map((v) => truncateSingle(v, targetDims));
  }

  return truncateSingle(input as Vector, targetDims);
}

function truncateSingle(v: Vector, targetDims: number): Float32Array {
  if (targetDims > v.length) {
    throw new ValidationError(
      `targetDims (${targetDims}) exceeds vector length (${v.length})`
    );
  }
  if (v instanceof Float32Array) {
    if (targetDims === v.length) return v;
    return v.slice(0, targetDims);
  }
  // number[] input
  if (targetDims === v.length) return new Float32Array(v);
  return new Float32Array(v.slice(0, targetDims));
}

/**
 * Validates that all embeddings have consistent dimensions.
 *
 * If `expectedDim` is provided, checks all embeddings against it. Otherwise
 * infers the expected dimension from the first embedding.
 *
 * @param embeddings - Array of embedding vectors to validate
 * @param expectedDim - Optional expected dimension count
 * @returns Object with `valid` (boolean), `dimension` (inferred or expected), and `mismatches` (indices of bad vectors)
 *
 * @example
 * validateDimensions([[1,2], [3,4]]); // { valid: true, dimension: 2, mismatches: [] }
 * validateDimensions([[1,2], [3,4,5]]); // { valid: false, dimension: 2, mismatches: [1] }
 */
export function validateDimensions(
  embeddings: Vector[],
  expectedDim?: number,
): { valid: boolean; dimension: number; mismatches: number[] } {
  if (embeddings.length === 0) {
    return { valid: true, dimension: expectedDim ?? 0, mismatches: [] };
  }

  const dim = expectedDim ?? embeddings[0].length;
  const mismatches: number[] = [];

  for (let i = 0; i < embeddings.length; i++) {
    if (embeddings[i].length !== dim) {
      mismatches.push(i);
    }
  }

  return { valid: mismatches.length === 0, dimension: dim, mismatches };
}

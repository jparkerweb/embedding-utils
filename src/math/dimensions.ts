// ─────────────────────────────────────────────────────────────────────────────
// Dimension Manipulation
//
// Functions for changing the dimensionality of embedding vectors. Currently
// supports Matryoshka-style truncation (keeping the first N dimensions).
// ─────────────────────────────────────────────────────────────────────────────

import { ValidationError } from '../types';

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
 * **Overloaded:** Accepts either a single vector (`number[]`) or a batch
 * of vectors (`number[][]`). The return type matches the input shape.
 *
 * @param input - A single embedding vector or array of embedding vectors
 * @param targetDims - The desired number of dimensions (must be > 0 and <= vector length)
 * @returns Truncated vector(s) matching the input shape
 * @throws {ValidationError} If targetDims is <= 0 or exceeds vector length
 *
 * @example
 * // Single vector
 * truncateDimensions([1, 2, 3, 4], 2); // [1, 2]
 *
 * @example
 * // Batch of vectors
 * truncateDimensions([[1, 2, 3], [4, 5, 6]], 2); // [[1, 2], [4, 5]]
 *
 * @example
 * // Reduce OpenAI 1536-dim embeddings to 256 for cheaper storage
 * const compact = truncateDimensions(embeddings, 256);
 */
export function truncateDimensions(embedding: number[], targetDims: number): number[];
export function truncateDimensions(embeddings: number[][], targetDims: number): number[][];
export function truncateDimensions(
  input: number[] | number[][],
  targetDims: number
): number[] | number[][] {
  if (targetDims <= 0) {
    throw new ValidationError('targetDims must be greater than 0');
  }

  if (Array.isArray(input[0])) {
    const batch = input as number[][];
    return batch.map((v) => truncateSingle(v, targetDims));
  }

  return truncateSingle(input as number[], targetDims);
}

function truncateSingle(v: number[], targetDims: number): number[] {
  if (targetDims > v.length) {
    throw new ValidationError(
      `targetDims (${targetDims}) exceeds vector length (${v.length})`
    );
  }
  if (targetDims === v.length) return v;
  return v.slice(0, targetDims);
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
  embeddings: number[][],
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

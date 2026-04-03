// ─────────────────────────────────────────────────────────────────────────────
// Distance Functions
//
// These functions measure how far apart two vectors are. Unlike similarity
// functions (where higher = more similar), distance functions return values
// where lower = more similar.
//
// When used in search or clustering, distances are automatically converted to
// similarity scores using the formula: similarity = 1 / (1 + distance).
// ─────────────────────────────────────────────────────────────────────────────

import { validateVectorPair } from '../internal/validation';
import { cosineSimilarity } from './similarity';

/**
 * Computes the Euclidean (L2) distance between two vectors.
 *
 * This is the "straight line" distance in high-dimensional space:
 * `sqrt(sum((a[i] - b[i])^2))`.
 *
 * **When to use:** Useful for anomaly detection — if a new data point's
 * embedding is far from the cluster centroid, flag it as an outlier. Also
 * commonly used in k-means-style clustering where absolute position matters.
 *
 * **Relationship to cosine:** Cosine similarity ignores magnitude (scale)
 * while euclidean distance considers it. Two vectors pointing in the same
 * direction but with different magnitudes will have cosine similarity ≈ 1
 * but euclidean distance > 0.
 *
 * **In search/clustering:** Automatically converted to similarity via
 * `1 / (1 + distance)` so that higher values still mean "more similar."
 *
 * @param a - First vector (must have the same length as `b`)
 * @param b - Second vector (must have the same length as `a`)
 * @returns The Euclidean distance (always >= 0). Zero means identical vectors.
 * @throws {ValidationError} If either vector is empty or dimensions differ
 *
 * @example
 * euclideanDistance([0, 0], [3, 4]); // 5  (the classic 3-4-5 triangle)
 * euclideanDistance([1, 2], [1, 2]); // 0  (identical vectors)
 */
export function euclideanDistance(a: number[], b: number[]): number {
  validateVectorPair(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/**
 * Computes the Manhattan (L1) distance between two vectors.
 *
 * Also called "taxicab distance" — the sum of absolute differences along
 * each dimension: `sum(|a[i] - b[i]|)`.
 *
 * **When to use:** More robust to noisy or high-dimensional data than
 * euclidean distance. A good choice when embeddings have many dimensions
 * or you're running on constrained hardware (no square root needed).
 *
 * **Computational advantage:** Slightly cheaper than euclidean because it
 * avoids squaring and square root operations.
 *
 * **In search/clustering:** Automatically converted to similarity via
 * `1 / (1 + distance)` so that higher values still mean "more similar."
 *
 * @param a - First vector (must have the same length as `b`)
 * @param b - Second vector (must have the same length as `a`)
 * @returns The Manhattan distance (always >= 0). Zero means identical vectors.
 * @throws {ValidationError} If either vector is empty or dimensions differ
 *
 * @example
 * manhattanDistance([0, 0], [3, 4]); // 7  (|3-0| + |4-0|)
 * manhattanDistance([1, 2], [1, 2]); // 0  (identical vectors)
 */
export function manhattanDistance(a: number[], b: number[]): number {
  validateVectorPair(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum;
}

/**
 * Computes the cosine distance between two vectors: `1 - cosineSimilarity(a, b)`.
 *
 * **Range:** 0 (identical direction) to 2 (opposite direction).
 *
 * @param a - First vector (must have the same length as `b`)
 * @param b - Second vector (must have the same length as `a`)
 * @returns The cosine distance (0 = identical, 1 = orthogonal, 2 = opposite)
 * @throws {ValidationError} If either vector is empty or dimensions differ
 *
 * @example
 * cosineDistance([1, 0], [1, 0]);   // 0  (identical)
 * cosineDistance([1, 0], [0, 1]);   // 1  (orthogonal)
 * cosineDistance([1, 0], [-1, 0]);  // 2  (opposite)
 */
export function cosineDistance(a: number[], b: number[]): number {
  validateVectorPair(a, b);
  return 1 - cosineSimilarity(a, b);
}

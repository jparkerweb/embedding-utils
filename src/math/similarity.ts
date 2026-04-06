// ─────────────────────────────────────────────────────────────────────────────
// Similarity Functions
//
// These functions measure how similar two vectors are. They are the core
// building blocks used by search, clustering, and aggregation modules.
//
// Both functions validate inputs and throw ValidationError on empty or
// mismatched-dimension vectors.
// ─────────────────────────────────────────────────────────────────────────────

import type { Vector } from '../types';
import { validateVectorPair } from '../internal/validation';

/**
 * Computes the dot product (inner product) of two vectors.
 *
 * The dot product is a fundamental operation used internally by cosine
 * similarity and many other vector math functions. It returns the sum
 * of element-wise products: `a[0]*b[0] + a[1]*b[1] + ... + a[n]*b[n]`.
 *
 * **When to use directly:** Some embedding models (e.g., OpenAI's
 * text-embedding-3-* with dimension truncation) are optimized for dot-product
 * ranking. Check your model's documentation — if it says "use dot product,"
 * use this instead of cosine similarity.
 *
 * **Relationship to cosine similarity:** For L2-normalized vectors (magnitude = 1),
 * dot product equals cosine similarity. Many embedding models return pre-normalized
 * vectors, so dot product is a faster alternative when you know inputs are normalized.
 *
 * @param a - First vector (must have the same length as `b`)
 * @param b - Second vector (must have the same length as `a`)
 * @returns The dot product scalar value. Range depends on input magnitudes.
 * @throws {ValidationError} If either vector is empty or dimensions differ
 *
 * @example
 * dotProduct([1, 2, 3], [4, 5, 6]); // 32  (1*4 + 2*5 + 3*6)
 *
 * @example
 * // For normalized embeddings, dot product ≈ cosine similarity
 * // dotProduct(normalize(a), normalize(b)) ≈ cosineSimilarity(a, b)
 */
export function dotProduct(a: Vector, b: Vector): number {
  validateVectorPair(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * Computes the cosine similarity between two vectors.
 *
 * Cosine similarity measures the angle between two vectors, ignoring their
 * magnitude. It is the most widely used metric for comparing embedding vectors
 * because it focuses on semantic direction rather than scale.
 *
 * **Formula:** `cos(θ) = (a · b) / (||a|| × ||b||)`
 *
 * **Return range:** -1 (opposite) to 1 (identical direction). For most
 * embedding models, values fall in the 0 to 1 range since embeddings
 * typically don't have negative directions.
 *
 * **Zero vector handling:** Returns 0 if either vector has zero magnitude,
 * avoiding division-by-zero errors.
 *
 * **When to use:** This is the default metric for semantic search, duplicate
 * detection, recommendation engines, and topic clustering. It is the go-to
 * choice unless your model documentation specifically recommends another metric.
 *
 * **Performance note:** This function computes dot product and magnitudes in a
 * single pass (one loop) for efficiency.
 *
 * @param a - First vector (must have the same length as `b`)
 * @param b - Second vector (must have the same length as `a`)
 * @returns Similarity score between -1 and 1. Returns 0 if either vector is all zeros.
 * @throws {ValidationError} If either vector is empty or dimensions differ
 *
 * @example
 * cosineSimilarity([1, 0], [0, 1]);   // 0   (perpendicular — unrelated)
 * cosineSimilarity([1, 0], [1, 0]);   // 1   (identical direction — same topic)
 * cosineSimilarity([1, 0], [-1, 0]);  // -1  (opposite direction)
 *
 * @example
 * // Compare a query to candidate documents
 * const scores = documents.map(doc => cosineSimilarity(queryEmbed, doc));
 */
export function cosineSimilarity(a: Vector, b: Vector): number {
  validateVectorPair(a, b);

  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }

  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  if (denom === 0) return 0;
  return dot / denom;
}

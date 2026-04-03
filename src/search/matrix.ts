// ─────────────────────────────────────────────────────────────────────────────
// Similarity Matrix
//
// Compute a full NxN pairwise similarity matrix. Used for content audits,
// overlap visualization (heatmaps), and cluster quality analysis.
// ─────────────────────────────────────────────────────────────────────────────

import { computeScore } from '../internal/metrics';
import { validateEmbeddings } from '../internal/validation';
import { ValidationError } from '../types';
import type { SearchOptions } from '../types';

/**
 * Computes a pairwise NxN similarity matrix for a set of embeddings.
 *
 * The result is a symmetric matrix where `matrix[i][j]` is the similarity
 * between embedding `i` and embedding `j`. The diagonal contains self-similarity
 * scores (1.0 for cosine on non-zero vectors).
 *
 * **Optimization:** Only computes the upper triangle and mirrors to the lower
 * triangle, performing N*(N-1)/2 comparisons instead of N^2.
 *
 * **When to use:**
 * - Content audits: find redundant documentation among support articles
 * - Overlap visualization: power heatmaps showing which items cover similar ground
 * - Cluster quality analysis: inspect inter-cluster vs intra-cluster distances
 *
 * **Complexity:** O(n^2) in both time and space. For very large sets (10,000+),
 * consider sampling or using {@link topK} per item instead.
 *
 * @param embeddings - Array of embedding vectors (must be non-empty)
 * @param options - Optional similarity metric (defaults to 'cosine')
 * @returns A symmetric NxN matrix of similarity scores
 * @throws {ValidationError} If the embeddings array is empty
 *
 * @example
 * const matrix = similarityMatrix(embeddings);
 * // matrix[0][1] === matrix[1][0]  (symmetric)
 * // matrix[0][0] === 1.0           (self-similarity for cosine)
 */
export function similarityMatrix(
  embeddings: number[][],
  options?: SearchOptions,
): number[][] {
  if (embeddings.length === 0) {
    throw new ValidationError('Embeddings array must be non-empty');
  }
  validateEmbeddings(embeddings);

  const metric = options?.metric ?? 'cosine';
  const n = embeddings.length;
  const matrix: number[][] = Array.from({ length: n }, () => new Array<number>(n));

  // Compute upper triangle and mirror
  for (let i = 0; i < n; i++) {
    matrix[i][i] = computeScore(embeddings[i], embeddings[i], metric);
    for (let j = i + 1; j < n; j++) {
      const score = computeScore(embeddings[i], embeddings[j], metric);
      matrix[i][j] = score;
      matrix[j][i] = score;
    }
  }

  return matrix;
}

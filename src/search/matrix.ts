// ─────────────────────────────────────────────────────────────────────────────
// Similarity Matrix
//
// Compute a full NxN pairwise similarity matrix. Used for content audits,
// overlap visualization (heatmaps), and cluster quality analysis.
// ─────────────────────────────────────────────────────────────────────────────

import { computeScore } from '../internal/metrics';
import { validateEmbeddings } from '../internal/validation';
import { ValidationError } from '../types';
import type { SearchOptions, Vector } from '../types';

/**
 * Computes a pairwise NxN similarity matrix for a set of embeddings.
 *
 * @param embeddings - Array of embedding vectors (must be non-empty)
 * @param options - Optional similarity metric (defaults to 'cosine')
 * @returns A symmetric NxN matrix of similarity scores
 * @throws {ValidationError} If the embeddings array is empty
 */
export function similarityMatrix(
  embeddings: Vector[],
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

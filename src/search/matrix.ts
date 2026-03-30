import { cosineSimilarity, dotProduct } from '../math/similarity';
import { euclideanDistance, manhattanDistance } from '../math/distance';
import { ValidationError } from '../types';
import type { SimilarityMetric } from '../types';

function computeScore(a: number[], b: number[], metric: SimilarityMetric): number {
  switch (metric) {
    case 'cosine':
      return cosineSimilarity(a, b);
    case 'dot':
      return dotProduct(a, b);
    case 'euclidean':
      return 1 / (1 + euclideanDistance(a, b));
    case 'manhattan':
      return 1 / (1 + manhattanDistance(a, b));
  }
}

/**
 * Computes a pairwise similarity matrix for a set of embeddings.
 * @param embeddings - Array of embedding vectors
 * @param options - Optional similarity metric (defaults to cosine)
 * @returns A symmetric NxN matrix of similarity scores
 * @throws {ValidationError} If the embeddings array is empty
 * @example
 * similarityMatrix([[1, 0], [0, 1]]); // [[1, 0], [0, 1]]
 */
export function similarityMatrix(
  embeddings: number[][],
  options?: { metric?: SimilarityMetric },
): number[][] {
  if (embeddings.length === 0) {
    throw new ValidationError('Embeddings array must be non-empty');
  }

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

import { computeScore } from '../internal/metrics';
import { DimensionMismatchError, ValidationError } from '../types';
import type { SimilarityMetric } from '../types';

/**
 * Computes element-wise similarity between two lists of embeddings.
 *
 * For each index `i`, computes the similarity between `listA[i]` and `listB[i]`
 * using the specified metric. Both lists must have the same length, and each
 * pair must have matching dimensions.
 *
 * @param listA - First list of embedding vectors
 * @param listB - Second list of embedding vectors (same length as listA)
 * @param metric - Similarity metric to use. Default: 'cosine'
 * @returns Array of similarity scores, one per pair
 */
export function pairwiseSimilarity(
  listA: number[][],
  listB: number[][],
  metric: SimilarityMetric = 'cosine',
): number[] {
  if (listA.length !== listB.length) {
    throw new ValidationError(
      `Lists must have the same length, got ${listA.length} and ${listB.length}`,
    );
  }

  const scores: number[] = new Array(listA.length);
  for (let i = 0; i < listA.length; i++) {
    if (listA[i].length !== listB[i].length) {
      throw new DimensionMismatchError(
        `Dimension mismatch at index ${i}: ${listA[i].length} vs ${listB[i].length}`,
      );
    }
    scores[i] = computeScore(listA[i], listB[i], metric);
  }

  return scores;
}

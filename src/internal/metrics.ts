import { cosineSimilarity, dotProduct } from '../math/similarity';
import { euclideanDistance, manhattanDistance } from '../math/distance';
import type { SimilarityMetric, Vector } from '../types';
import { ValidationError } from '../types';

/**
 * Computes a similarity score between two vectors using the specified metric.
 * For distance-based metrics (euclidean, manhattan), converts to similarity
 * using the formula: `1 / (1 + distance)` so that higher = more similar.
 * @internal
 */
export function computeScore(a: Vector, b: Vector, metric: SimilarityMetric): number {
  switch (metric) {
    case 'cosine':
      return cosineSimilarity(a, b);
    case 'dot':
      return dotProduct(a, b);
    case 'euclidean':
      return 1 / (1 + euclideanDistance(a, b));
    case 'manhattan':
      return 1 / (1 + manhattanDistance(a, b));
    default:
      throw new ValidationError(`Unknown similarity metric: ${(metric as string)}`);
  }
}

/**
 * Computes a distance between two vectors using the specified metric.
 * For similarity-based metrics (cosine, dot), converts to distance
 * using `1 - similarity`.
 * @internal
 */
export function computeDistance(a: Vector, b: Vector, metric: SimilarityMetric): number {
  switch (metric) {
    case 'cosine':
      return 1 - cosineSimilarity(a, b);
    case 'dot':
      return 1 - dotProduct(a, b);
    case 'euclidean':
      return euclideanDistance(a, b);
    case 'manhattan':
      return manhattanDistance(a, b);
    default:
      throw new ValidationError(`Unknown similarity metric: ${(metric as string)}`);
  }
}

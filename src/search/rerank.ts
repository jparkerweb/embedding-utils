import { computeScore } from '../internal/metrics';
import type { SimilarityMetric, Vector } from '../types';

/**
 * Re-ranks search results by combining original scores with recomputed similarity.
 *
 * @param results - Array of search results with index, score, and embedding
 * @param query - The query embedding to re-score against
 * @param options - Optional metric and weight configuration
 * @returns Re-ranked results sorted by combined score descending
 */
export function rerankResults(
  results: Array<{ index: number; score: number; embedding: Vector }>,
  query: Vector,
  options?: {
    metric?: SimilarityMetric;
    weights?: { original?: number; rerank?: number };
  },
): Array<{ index: number; score: number }> {
  const metric = options?.metric ?? 'cosine';
  const originalWeight = options?.weights?.original ?? 0.5;
  const rerankWeight = options?.weights?.rerank ?? 0.5;

  const reranked = results.map((result) => ({
    index: result.index,
    score: originalWeight * result.score + rerankWeight * computeScore(query, result.embedding, metric),
  }));

  reranked.sort((a, b) => b.score - a.score);
  return reranked;
}

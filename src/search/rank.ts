import { topK } from './topk';
import type { SimilarityMetric, SearchResult } from '../types';

/**
 * Ranks all corpus embeddings by similarity to a query (descending).
 * @param query - The query embedding vector
 * @param corpus - Array of candidate embedding vectors
 * @param options - Optional similarity metric and labels
 * @returns All corpus items as search results sorted by descending similarity
 * @example
 * rankBySimilarity([1, 0], [[0, 1], [1, 0], [0.5, 0.5]]);
 */
export function rankBySimilarity(
  query: number[],
  corpus: number[][],
  options?: { metric?: SimilarityMetric; labels?: string[] },
): SearchResult[] {
  return topK(query, corpus, corpus.length, options);
}

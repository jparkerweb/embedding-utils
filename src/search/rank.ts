import { topK } from './topk';
import type { SearchOptions, SearchResult, Vector } from '../types';

/**
 * Ranks **all** corpus embeddings by similarity to a query (descending).
 *
 * @param query - The query embedding vector
 * @param corpus - Array of candidate embedding vectors
 * @param options - Optional similarity metric (default: 'cosine') and labels
 * @returns All corpus items as SearchResult objects sorted by descending similarity
 */
export function rankBySimilarity(
  query: Vector,
  corpus: Vector[],
  options?: SearchOptions,
): SearchResult[] {
  if (corpus.length === 0) return [];
  return topK(query, corpus, corpus.length, options);
}

import { topK } from './topk';
import type { SearchOptions, SearchResult } from '../types';

/**
 * Ranks **all** corpus embeddings by similarity to a query (descending).
 *
 * Unlike {@link topK} which returns only K results, this returns scores for
 * every item in the corpus. Implemented as `topK(query, corpus, corpus.length)`.
 *
 * **When to use:** Recommendation feeds — a user just read an article, rank
 * your entire library by relevance to build a "more like this" feed. Also
 * useful for full-corpus analysis where you need every score.
 *
 * @param query - The query embedding vector
 * @param corpus - Array of candidate embedding vectors
 * @param options - Optional similarity metric (default: 'cosine') and labels
 * @returns All corpus items as SearchResult objects sorted by descending similarity
 *
 * @example
 * const ranked = rankBySimilarity(queryEmbed, corpus, { labels });
 * ranked.forEach(r => console.log(`${r.label}: ${r.score.toFixed(3)}`));
 */
export function rankBySimilarity(
  query: number[],
  corpus: number[][],
  options?: SearchOptions,
): SearchResult[] {
  if (corpus.length === 0) return [];
  return topK(query, corpus, corpus.length, options);
}

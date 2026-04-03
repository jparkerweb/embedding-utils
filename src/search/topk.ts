// ─────────────────────────────────────────────────────────────────────────────
// Top-K Search
//
// Find the K most similar embeddings to a query. This is the primary entry
// point for semantic search, RAG pipelines, and recommendation engines.
// ─────────────────────────────────────────────────────────────────────────────

import { computeScore } from '../internal/metrics';
import { MinHeap } from '../internal/heap';
import { DimensionMismatchError, ValidationError } from '../types';
import type { SearchOptions, SearchResult } from '../types';

/**
 * Finds the top K most similar embeddings to a query vector.
 *
 * This is the workhorse function for semantic search and RAG pipelines. Given
 * a query embedding and a corpus of candidate embeddings, it returns the K
 * closest matches sorted by descending similarity.
 *
 * **How it works:** Computes the similarity score between the query and every
 * corpus vector, sorts by score, and returns the top K results. This is a
 * brute-force linear scan — no approximate nearest neighbor indexing.
 *
 * **Labels:** Pass an optional `labels` array (same length as corpus) to
 * attach human-readable identifiers to results. Useful for mapping back to
 * document IDs, filenames, or topic names.
 *
 * **Metrics:** Supports all four metrics via `options.metric`. Distance-based
 * metrics (euclidean, manhattan) are automatically converted to similarity
 * scores using `1 / (1 + distance)`.
 *
 * @param query - The query embedding vector
 * @param corpus - Array of candidate embedding vectors to search
 * @param k - Number of results to return. If k <= 0, returns empty array.
 * @param options - Optional search options (metric, labels, filter)
 * @returns Array of SearchResult objects sorted by descending similarity score
 *
 * @example
 * const results = topK(queryEmbed, documentEmbeds, 5, {
 *   labels: documentTitles,
 * });
 * results[0].label; // => 'Most relevant document title'
 * results[0].score; // => 0.92
 */
export function topK(
  query: number[],
  corpus: number[][],
  k: number,
  options?: SearchOptions,
): SearchResult[] {
  if (k < 1 || !Number.isInteger(k)) {
    throw new ValidationError(`k must be a positive integer, got ${k}`);
  }

  const metric = options?.metric ?? 'cosine';
  const labels = options?.labels;
  const filter = options?.filter;

  if (corpus.length > 0 && query.length !== corpus[0].length) {
    throw new DimensionMismatchError(
      `Query dimension (${query.length}) does not match corpus dimension (${corpus[0].length})`,
    );
  }

  // Use heap-based selection when k is small relative to corpus size (O(n log k))
  // Fall back to sort when k >= corpus.length / 2 (sort is comparable at that point)
  if (k < corpus.length / 2) {
    // Max-heap by score: largest score at top so we can evict it when a better one arrives
    const heap = new MinHeap<SearchResult>((a, b) => a.score - b.score);

    for (let i = 0; i < corpus.length; i++) {
      if (filter && !filter(i, labels?.[i])) continue;

      const score = computeScore(query, corpus[i], metric);
      const result: SearchResult = {
        index: i,
        score,
        embedding: corpus[i],
        ...(labels?.[i] != null ? { label: labels[i] } : {}),
      };

      if (heap.size < k) {
        heap.push(result);
      } else if (score > heap.peek()!.score) {
        heap.pop();
        heap.push(result);
      }
    }

    // Drain heap and reverse for descending order
    const results: SearchResult[] = [];
    while (heap.size > 0) {
      results.push(heap.pop()!);
    }
    results.reverse();
    return results;
  }

  // Sort-based path for large k
  const scored: SearchResult[] = [];
  for (let i = 0; i < corpus.length; i++) {
    if (filter && !filter(i, labels?.[i])) continue;

    scored.push({
      index: i,
      score: computeScore(query, corpus[i], metric),
      embedding: corpus[i],
      ...(labels?.[i] != null ? { label: labels[i] } : {}),
    });
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

/**
 * Finds the top K most similar embeddings for each of multiple queries (batch search).
 *
 * Convenience wrapper that calls {@link topK} for each query. Use this when
 * processing a batch of queries in one operation instead of looping manually.
 *
 * **When to use:** Batch search — e.g., 50 customer questions came in overnight
 * and you want to find matching FAQ articles for all of them in one call.
 *
 * @param queries - Array of query embedding vectors
 * @param corpus - Array of candidate embedding vectors
 * @param k - Number of results per query
 * @param options - Optional similarity metric and labels
 * @returns Array of SearchResult arrays, one per query (same order as input queries)
 *
 * @example
 * const results = topKMulti(queryEmbeds, corpus, 3);
 * results[0]; // top 3 matches for first query
 * results[1]; // top 3 matches for second query
 */
export function topKMulti(
  queries: number[][],
  corpus: number[][],
  k: number,
  options?: SearchOptions,
): SearchResult[][] {
  return queries.map((query) => topK(query, corpus, k, options));
}

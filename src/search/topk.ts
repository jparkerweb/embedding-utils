// ─────────────────────────────────────────────────────────────────────────────
// Top-K Search
//
// Find the K most similar embeddings to a query. This is the primary entry
// point for semantic search, RAG pipelines, and recommendation engines.
// ─────────────────────────────────────────────────────────────────────────────

import { computeScore } from '../internal/metrics';
import { toFloat32 } from '../internal/vector-utils';
import { MinHeap } from '../internal/heap';
import { DimensionMismatchError, ValidationError } from '../types';
import type { SearchOptions, SearchResult, Vector } from '../types';

/**
 * Finds the top K most similar embeddings to a query vector.
 *
 * @param query - The query embedding vector
 * @param corpus - Array of candidate embedding vectors to search
 * @param k - Number of results to return. If k <= 0, returns empty array.
 * @param options - Optional search options (metric, labels, filter)
 * @returns Array of SearchResult objects sorted by descending similarity score
 */
export function topK(
  query: Vector,
  corpus: Vector[],
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
        embedding: toFloat32(corpus[i]),
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
      embedding: toFloat32(corpus[i]),
      ...(labels?.[i] != null ? { label: labels[i] } : {}),
    });
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

/**
 * Finds the top K most similar embeddings for each of multiple queries (batch search).
 *
 * @param queries - Array of query embedding vectors
 * @param corpus - Array of candidate embedding vectors
 * @param k - Number of results per query
 * @param options - Optional similarity metric and labels
 * @returns Array of SearchResult arrays, one per query (same order as input queries)
 */
export function topKMulti(
  queries: Vector[],
  corpus: Vector[],
  k: number,
  options?: SearchOptions,
): SearchResult[][] {
  return queries.map((query) => topK(query, corpus, k, options));
}

import { cosineSimilarity, dotProduct } from '../math/similarity';
import { euclideanDistance, manhattanDistance } from '../math/distance';
import type { SimilarityMetric, SearchResult } from '../types';

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
 * Finds the top K most similar embeddings to a query.
 * @param query - The query embedding vector
 * @param corpus - Array of candidate embedding vectors
 * @param k - Number of results to return
 * @param options - Optional similarity metric and labels
 * @returns Array of search results sorted by descending similarity
 * @example
 * topK([1, 0], [[1, 0], [0, 1], [0.5, 0.5]], 2);
 * // [{ index: 0, score: 1, ... }, { index: 2, score: 0.707, ... }]
 */
export function topK(
  query: number[],
  corpus: number[][],
  k: number,
  options?: { metric?: SimilarityMetric; labels?: string[] },
): SearchResult[] {
  if (k <= 0) return [];

  const metric = options?.metric ?? 'cosine';
  const labels = options?.labels;

  const scored: SearchResult[] = corpus.map((embedding, index) => ({
    index,
    score: computeScore(query, embedding, metric),
    embedding,
    ...(labels?.[index] != null ? { label: labels[index] } : {}),
  }));

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

/**
 * Finds the top K most similar embeddings for each of multiple queries.
 * @param queries - Array of query embedding vectors
 * @param corpus - Array of candidate embedding vectors
 * @param k - Number of results per query
 * @param options - Optional similarity metric and labels
 * @returns Array of search result arrays, one per query
 * @example
 * topKMulti([[1, 0], [0, 1]], corpus, 3);
 */
export function topKMulti(
  queries: number[][],
  corpus: number[][],
  k: number,
  options?: { metric?: SimilarityMetric; labels?: string[] },
): SearchResult[][] {
  return queries.map((query) => topK(query, corpus, k, options));
}

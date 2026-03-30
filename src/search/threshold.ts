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
 * Finds all embeddings with similarity above a threshold.
 * @param query - The query embedding vector
 * @param corpus - Array of candidate embedding vectors
 * @param threshold - Minimum similarity score to include
 * @param options - Optional similarity metric and labels
 * @returns Array of search results sorted by descending similarity
 * @example
 * aboveThreshold([1, 0], [[1, 0], [0, 1]], 0.5); // only the first match
 */
export function aboveThreshold(
  query: number[],
  corpus: number[][],
  threshold: number,
  options?: { metric?: SimilarityMetric; labels?: string[] },
): SearchResult[] {
  const metric = options?.metric ?? 'cosine';
  const labels = options?.labels;

  const results: SearchResult[] = [];
  for (let i = 0; i < corpus.length; i++) {
    const score = computeScore(query, corpus[i], metric);
    if (score >= threshold) {
      results.push({
        index: i,
        score,
        embedding: corpus[i],
        ...(labels?.[i] != null ? { label: labels[i] } : {}),
      });
    }
  }

  results.sort((a, b) => b.score - a.score);
  return results;
}

/**
 * Removes near-duplicate embeddings based on a similarity threshold.
 * @param embeddings - Array of embedding vectors to deduplicate
 * @param threshold - Similarity above which two embeddings are considered duplicates
 * @param options - Optional similarity metric and labels
 * @returns Object with deduplicated embeddings, their original indices, and optional labels
 * @example
 * deduplicate([[1, 0], [1, 0.01], [0, 1]], 0.99);
 * // { embeddings: [[1, 0], [0, 1]], indices: [0, 2] }
 */
export function deduplicate(
  embeddings: number[][],
  threshold: number,
  options?: { metric?: SimilarityMetric; labels?: string[] },
): { embeddings: number[][]; indices: number[]; labels?: string[] } {
  const metric = options?.metric ?? 'cosine';
  const labels = options?.labels;

  const kept: number[][] = [];
  const keptIndices: number[] = [];
  const keptLabels: string[] = [];

  for (let i = 0; i < embeddings.length; i++) {
    let isDuplicate = false;
    for (const existing of kept) {
      if (computeScore(embeddings[i], existing, metric) >= threshold) {
        isDuplicate = true;
        break;
      }
    }
    if (!isDuplicate) {
      kept.push(embeddings[i]);
      keptIndices.push(i);
      if (labels) {
        keptLabels.push(labels[i]);
      }
    }
  }

  return {
    embeddings: kept,
    indices: keptIndices,
    ...(labels ? { labels: keptLabels } : {}),
  };
}

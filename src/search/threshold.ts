// ─────────────────────────────────────────────────────────────────────────────
// Threshold-Based Search & Deduplication
//
// Functions that filter embeddings by a similarity threshold rather than
// returning a fixed count. Use when you need "all good matches" or want to
// remove near-duplicates.
// ─────────────────────────────────────────────────────────────────────────────

import { computeScore } from '../internal/metrics';
import { toFloat32 } from '../internal/vector-utils';
import { DimensionMismatchError, ValidationError } from '../types';
import type { SearchOptions, SearchResult, Vector } from '../types';

/**
 * Finds all embeddings with similarity above a threshold.
 *
 * @param query - The query embedding vector
 * @param corpus - Array of candidate embedding vectors
 * @param threshold - Minimum similarity score to include in results
 * @param options - Optional similarity metric (default: 'cosine') and labels
 * @returns Array of search results sorted by descending similarity. May be empty.
 */
export function aboveThreshold(
  query: Vector,
  corpus: Vector[],
  threshold: number,
  options?: SearchOptions,
): SearchResult[] {
  if (!Number.isFinite(threshold)) {
    throw new ValidationError(`threshold must be a finite number, got ${threshold}`);
  }

  const metric = options?.metric ?? 'cosine';
  const labels = options?.labels;
  const filter = options?.filter;

  if (corpus.length > 0 && query.length !== corpus[0].length) {
    throw new DimensionMismatchError(
      `Query dimension (${query.length}) does not match corpus dimension (${corpus[0].length})`,
    );
  }

  const results: SearchResult[] = [];
  for (let i = 0; i < corpus.length; i++) {
    if (filter && !filter(i, labels?.[i])) continue;

    const score = computeScore(query, corpus[i], metric);
    if (score >= threshold) {
      results.push({
        index: i,
        score,
        embedding: toFloat32(corpus[i]),
        ...(labels?.[i] != null ? { label: labels[i] } : {}),
      });
    }
  }

  results.sort((a, b) => b.score - a.score);
  return results;
}

/**
 * Removes near-duplicate embeddings based on a similarity threshold.
 *
 * @param embeddings - Array of embedding vectors to deduplicate
 * @param threshold - Similarity at or above which two embeddings are considered duplicates.
 * @param options - Optional similarity metric (default: 'cosine') and labels
 * @returns Object with deduplicated embeddings as Float32Array[], their original indices, and optional labels
 */
export function deduplicate(
  embeddings: Vector[],
  threshold: number,
  options?: SearchOptions,
): { embeddings: Float32Array[]; indices: number[]; labels?: string[] } {
  const metric = options?.metric ?? 'cosine';
  const labels = options?.labels;

  const kept: Float32Array[] = [];
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
      kept.push(toFloat32(embeddings[i]));
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

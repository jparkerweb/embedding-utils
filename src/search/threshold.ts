// ─────────────────────────────────────────────────────────────────────────────
// Threshold-Based Search & Deduplication
//
// Functions that filter embeddings by a similarity threshold rather than
// returning a fixed count. Use when you need "all good matches" or want to
// remove near-duplicates.
// ─────────────────────────────────────────────────────────────────────────────

import { computeScore } from '../internal/metrics';
import { DimensionMismatchError, ValidationError } from '../types';
import type { SearchOptions, SearchResult } from '../types';

/**
 * Finds all embeddings with similarity above a threshold.
 *
 * Unlike {@link topK} which returns a fixed number of results, this function
 * returns a variable number — from zero (nothing matches) to all corpus items.
 *
 * **When to use:** Intent matching where you only want confident matches. For
 * example, a voice assistant should only act on matches above 0.85 — anything
 * below is "I don't understand." Also useful for topic detection pipelines
 * where each topic has its own similarity threshold.
 *
 * @param query - The query embedding vector
 * @param corpus - Array of candidate embedding vectors
 * @param threshold - Minimum similarity score to include in results
 * @param options - Optional similarity metric (default: 'cosine') and labels
 * @returns Array of search results sorted by descending similarity. May be empty.
 *
 * @example
 * // Only return confident matches
 * const matches = aboveThreshold(queryEmbed, corpus, 0.85, { labels: topics });
 * if (matches.length === 0) console.log('No confident match found');
 */
export function aboveThreshold(
  query: number[],
  corpus: number[][],
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
 *
 * Iterates through embeddings sequentially. For each one, compares against
 * all previously kept embeddings. If any existing embedding has similarity
 * >= threshold, the new one is discarded as a duplicate. First occurrence wins.
 *
 * **When to use:** Content deduplication. You scraped 10,000 product listings
 * and many are reposts with slightly different wording. Deduplicate at 0.95
 * to collapse near-identical listings, keeping the first occurrence.
 *
 * **Order-dependent:** The first occurrence of each "duplicate group" is kept.
 * If order matters, sort your embeddings before calling this function.
 *
 * @param embeddings - Array of embedding vectors to deduplicate
 * @param threshold - Similarity at or above which two embeddings are considered duplicates.
 *                    Higher = stricter (only very similar items removed).
 * @param options - Optional similarity metric (default: 'cosine') and labels
 * @returns Object with deduplicated embeddings, their original indices, and optional labels
 *
 * @example
 * const result = deduplicate(embeddings, 0.95, { labels });
 * console.log(`Removed ${embeddings.length - result.embeddings.length} duplicates`);
 */
export function deduplicate(
  embeddings: number[][],
  threshold: number,
  options?: SearchOptions,
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

import type { RankedItem, NormalizationMethod } from '../types';
import { ValidationError } from '../types';

/**
 * Merges multiple ranked lists using Reciprocal Rank Fusion (RRF).
 *
 * For each item across all lists, sums `1 / (k + rank)` where rank is
 * the 1-based position in each list. Returns a merged list sorted by
 * fused score descending.
 *
 * @param lists - Array of ranked item lists to fuse
 * @param options - Optional configuration: `k` parameter (default 60)
 * @returns Merged list sorted by fused RRF score descending
 * @throws {ValidationError} If lists is empty array of non-arrays
 */
export function fuseRankedLists(
  lists: RankedItem[][],
  options?: { k?: number },
): RankedItem[] {
  if (lists.length === 0) return [];

  const k = options?.k ?? 60;
  const scoreMap = new Map<string, number>();
  const orderMap = new Map<string, number>();
  let insertOrder = 0;

  for (const list of lists) {
    if (!Array.isArray(list)) {
      throw new ValidationError('Each element in lists must be an array');
    }
    for (let rank = 0; rank < list.length; rank++) {
      const item = list[rank];
      const rrfScore = 1 / (k + rank + 1); // rank is 1-based
      scoreMap.set(item.id, (scoreMap.get(item.id) ?? 0) + rrfScore);
      if (!orderMap.has(item.id)) {
        orderMap.set(item.id, insertOrder++);
      }
    }
  }

  const result: RankedItem[] = [];
  for (const [id, score] of scoreMap) {
    result.push({ id, score });
  }

  result.sort((a, b) => {
    const diff = b.score - a.score;
    if (diff !== 0) return diff;
    // Stable order: maintain insertion order for ties
    return orderMap.get(a.id)! - orderMap.get(b.id)!;
  });

  return result;
}

/**
 * Normalizes an array of scores using the specified method.
 *
 * @param scores - Array of numeric scores
 * @param method - Normalization method: 'min-max', 'z-score', or 'sigmoid'
 * @returns New array of normalized scores
 */
export function normalizeScores(scores: number[], method: NormalizationMethod): number[] {
  if (scores.length === 0) return [];

  switch (method) {
    case 'min-max': {
      const min = Math.min(...scores);
      const max = Math.max(...scores);
      const range = max - min;
      if (range === 0) return scores.map(() => 1);
      return scores.map((s) => (s - min) / range);
    }
    case 'z-score': {
      const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
      const variance = scores.reduce((a, b) => a + (b - mean) ** 2, 0) / scores.length;
      const std = Math.sqrt(variance);
      if (std === 0) return scores.map(() => 0);
      return scores.map((s) => (s - mean) / std);
    }
    case 'sigmoid': {
      return scores.map((s) => 1 / (1 + Math.exp(-s)));
    }
  }
}

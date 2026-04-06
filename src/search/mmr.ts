import { computeScore } from '../internal/metrics';
import type { MMROptions, Vector } from '../types';

/**
 * Maximal Marginal Relevance (MMR) search for diverse results.
 *
 * @param query - The query embedding vector
 * @param corpus - Array of candidate embedding vectors
 * @param k - Number of results to return
 * @param options - MMR options (lambda, fetchK, metric, labels)
 * @returns Array of results sorted by selection order (most relevant first)
 */
export function mmrSearch(
  query: Vector,
  corpus: Vector[],
  k: number,
  options?: MMROptions,
): Array<{ index: number; score: number }> {
  const metric = options?.metric ?? 'cosine';
  const lambda = options?.lambda ?? 0.5;
  const fetchK = options?.fetchK ?? k * 4;

  if (k <= 0 || corpus.length === 0) return [];

  // Compute query similarity for all corpus items
  const querySims: Array<{ index: number; score: number }> = corpus.map((embedding, index) => ({
    index,
    score: computeScore(query, embedding, metric),
  }));

  // Pre-filter to top fetchK candidates by query similarity
  querySims.sort((a, b) => b.score - a.score);
  const candidates = new Set(querySims.slice(0, Math.max(fetchK, k)).map((c) => c.index));
  const querySimMap = new Map(querySims.map((c) => [c.index, c.score]));

  const selected: Array<{ index: number; score: number }> = [];
  const actualK = Math.min(k, candidates.size);

  for (let step = 0; step < actualK; step++) {
    let bestIndex = -1;
    let bestMmrScore = -Infinity;

    for (const idx of candidates) {
      const relevance = querySimMap.get(idx)!;

      // Max similarity to already selected items
      let maxSimToSelected = 0;
      for (const sel of selected) {
        const sim = computeScore(corpus[idx], corpus[sel.index], metric);
        if (sim > maxSimToSelected) maxSimToSelected = sim;
      }

      const mmrScore = lambda * relevance - (1 - lambda) * maxSimToSelected;
      if (mmrScore > bestMmrScore) {
        bestMmrScore = mmrScore;
        bestIndex = idx;
      }
    }

    if (bestIndex === -1) break;

    selected.push({ index: bestIndex, score: querySimMap.get(bestIndex)! });
    candidates.delete(bestIndex);
  }

  return selected;
}

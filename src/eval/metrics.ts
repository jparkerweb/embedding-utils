/**
 * Retrieval evaluation metrics for measuring search quality.
 *
 * Provides standard IR metrics: recall@k, NDCG, MRR, and mean average precision.
 * All functions are pure — no state, no side effects.
 */

/**
 * Computes recall at k: the fraction of relevant items found in the top-k retrieved results.
 *
 * @param retrieved - Ordered list of retrieved item identifiers
 * @param relevant - Set or array of relevant item identifiers
 * @param k - Optional cutoff; defaults to retrieved.length
 * @returns Recall score in [0, 1]
 */
export function recallAtK(
  retrieved: string[],
  relevant: Set<string> | string[],
  k?: number,
): number {
  const relevantSet = relevant instanceof Set ? relevant : new Set(relevant);
  if (relevantSet.size === 0 || retrieved.length === 0) return 0;

  const cutoff = k !== undefined ? Math.min(k, retrieved.length) : retrieved.length;
  let hits = 0;
  for (let i = 0; i < cutoff; i++) {
    if (relevantSet.has(retrieved[i])) hits++;
  }
  return hits / relevantSet.size;
}

/**
 * Computes Normalized Discounted Cumulative Gain (NDCG).
 *
 * Uses the standard formula: DCG = sum((2^rel_i - 1) / log2(i + 2)) for i starting at 0.
 * NDCG = DCG / IDCG where IDCG is the DCG of the ideal ranking.
 *
 * @param retrieved - Ordered list of retrieved item identifiers
 * @param relevanceScores - Map or record of item id to graded relevance score
 * @param k - Optional cutoff; defaults to retrieved.length
 * @returns NDCG score in [0, 1]
 */
export function ndcg(
  retrieved: string[],
  relevanceScores: Map<string, number> | Record<string, number>,
  k?: number,
): number {
  if (retrieved.length === 0) return 0;

  const getScore = (id: string): number => {
    if (relevanceScores instanceof Map) return relevanceScores.get(id) ?? 0;
    return relevanceScores[id] ?? 0;
  };

  const cutoff = k !== undefined ? Math.min(k, retrieved.length) : retrieved.length;

  // Compute DCG
  let dcg = 0;
  for (let i = 0; i < cutoff; i++) {
    const rel = getScore(retrieved[i]);
    dcg += (Math.pow(2, rel) - 1) / Math.log2(i + 2);
  }

  // Compute IDCG: sort all relevance scores descending, take top cutoff
  const allScores: number[] =
    relevanceScores instanceof Map
      ? Array.from(relevanceScores.values())
      : Object.values(relevanceScores);
  allScores.sort((a, b) => b - a);

  let idcg = 0;
  const idealCutoff = Math.min(cutoff, allScores.length);
  for (let i = 0; i < idealCutoff; i++) {
    idcg += (Math.pow(2, allScores[i]) - 1) / Math.log2(i + 2);
  }

  if (idcg === 0) return 0;
  return dcg / idcg;
}

/**
 * Computes Mean Reciprocal Rank (MRR).
 *
 * Returns 1 / rank of the first relevant result. Returns 0 if no relevant result is found.
 *
 * @param retrieved - Ordered list of retrieved item identifiers
 * @param relevant - Set or array of relevant item identifiers
 * @returns MRR score in [0, 1]
 */
export function mrr(
  retrieved: string[],
  relevant: Set<string> | string[],
): number {
  const relevantSet = relevant instanceof Set ? relevant : new Set(relevant);
  if (relevantSet.size === 0 || retrieved.length === 0) return 0;

  for (let i = 0; i < retrieved.length; i++) {
    if (relevantSet.has(retrieved[i])) {
      return 1 / (i + 1);
    }
  }
  return 0;
}

/**
 * Computes Mean Average Precision (MAP).
 *
 * Average of precision values computed at each position where a relevant document is found.
 *
 * @param retrieved - Ordered list of retrieved item identifiers
 * @param relevant - Set or array of relevant item identifiers
 * @returns MAP score in [0, 1]
 */
export function meanAveragePrecision(
  retrieved: string[],
  relevant: Set<string> | string[],
): number {
  const relevantSet = relevant instanceof Set ? relevant : new Set(relevant);
  if (relevantSet.size === 0 || retrieved.length === 0) return 0;

  let hits = 0;
  let sumPrecision = 0;
  for (let i = 0; i < retrieved.length; i++) {
    if (relevantSet.has(retrieved[i])) {
      hits++;
      sumPrecision += hits / (i + 1);
    }
  }

  if (hits === 0) return 0;
  return sumPrecision / relevantSet.size;
}

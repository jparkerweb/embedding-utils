import type { SimilarityMetric, Vector } from '../types';
import { clusterEmbeddings } from './cluster';
import { computeDistance } from '../internal/metrics';
import { computeCentroid } from '../internal/clustering';

interface OptimalKOptions {
  minK?: number;
  maxK?: number;
  metric?: SimilarityMetric;
  method?: 'elbow' | 'silhouette';
}

/**
 * Computes the mean silhouette coefficient for a given set of cluster assignments.
 */
function meanSilhouette(
  embeddings: Vector[],
  assignments: number[],
  k: number,
  metric: SimilarityMetric,
): number {
  if (k <= 1) return 0;

  const n = embeddings.length;
  let totalSilhouette = 0;

  for (let i = 0; i < n; i++) {
    const ci = assignments[i];

    // a = average distance to own cluster members
    let aSum = 0;
    let aCount = 0;
    for (let j = 0; j < n; j++) {
      if (j !== i && assignments[j] === ci) {
        aSum += computeDistance(embeddings[i], embeddings[j], metric);
        aCount++;
      }
    }
    const a = aCount > 0 ? aSum / aCount : 0;

    // b = min average distance to any other cluster
    let b = Infinity;
    for (let ck = 0; ck < k; ck++) {
      if (ck === ci) continue;
      let bSum = 0;
      let bCount = 0;
      for (let j = 0; j < n; j++) {
        if (assignments[j] === ck) {
          bSum += computeDistance(embeddings[i], embeddings[j], metric);
          bCount++;
        }
      }
      if (bCount > 0) {
        const avg = bSum / bCount;
        if (avg < b) b = avg;
      }
    }
    if (b === Infinity) b = 0;

    const denom = Math.max(a, b);
    totalSilhouette += denom === 0 ? 0 : (b - a) / denom;
  }

  return totalSilhouette / n;
}

/**
 * Computes within-cluster sum of distances (WCSD) for elbow method.
 */
function withinClusterSumOfDistances(
  embeddings: Vector[],
  assignments: number[],
  k: number,
  metric: SimilarityMetric,
): number {
  // Compute centroid for each cluster
  const clusterMembers: Vector[][] = Array.from({ length: k }, () => []);
  for (let i = 0; i < embeddings.length; i++) {
    clusterMembers[assignments[i]].push(embeddings[i]);
  }

  let totalDist = 0;
  for (let c = 0; c < k; c++) {
    if (clusterMembers[c].length === 0) continue;
    const centroid = computeCentroid(clusterMembers[c]);
    for (const member of clusterMembers[c]) {
      totalDist += computeDistance(member, centroid, metric);
    }
  }
  return totalDist;
}

/**
 * Runs clustering for a given k and returns flat assignment array.
 */
function clusterAndAssign(
  embeddings: Vector[],
  k: number,
  metric: SimilarityMetric,
): number[] {
  // Use labels to track original indices through clustering
  const indexLabels = embeddings.map((_, i) => String(i));
  const clusters = clusterEmbeddings(embeddings, {
    maxClusters: k,
    minClusterSize: 1,
    similarityThreshold: 0,
    metric,
  }, indexLabels);

  // Build assignment map using labels (original indices)
  const assignments = new Array<number>(embeddings.length).fill(0);
  for (let ci = 0; ci < clusters.length; ci++) {
    if (clusters[ci].labels) {
      for (const label of clusters[ci].labels!) {
        const idx = parseInt(label, 10);
        if (idx >= 0 && idx < embeddings.length) {
          assignments[idx] = ci;
        }
      }
    }
  }
  return assignments;
}

/**
 * Finds the optimal number of clusters (k) for a dataset using the elbow or
 * silhouette method.
 *
 * **Note:** This is computationally expensive — it runs clustering for each k
 * in [minK, maxK]. Intended for exploration/tuning, not production hot paths.
 *
 * @param embeddings - Array of embedding vectors to analyze
 * @param options - Configuration: minK, maxK, metric, method
 * @returns The optimal k value
 */
export function findOptimalK(
  embeddings: Vector[],
  options?: OptimalKOptions,
): number {
  const method = options?.method ?? 'silhouette';
  const metric = options?.metric ?? 'cosine';
  const minK = options?.minK ?? 2;
  const maxK = options?.maxK ?? Math.min(10, Math.floor(embeddings.length / 2));

  if (minK < 2) {
    throw new Error('minK must be >= 2');
  }
  if (maxK < minK) {
    return minK;
  }

  if (method === 'silhouette') {
    let bestK = minK;
    let bestScore = -Infinity;
    for (let k = minK; k <= maxK; k++) {
      const assignments = clusterAndAssign(embeddings, k, metric);
      const score = meanSilhouette(embeddings, assignments, k, metric);
      if (score > bestScore) {
        bestScore = score;
        bestK = k;
      }
    }
    return bestK;
  }

  // Elbow method: find k at maximum second derivative of WCSD
  const wcsd: number[] = [];
  for (let k = minK; k <= maxK; k++) {
    const assignments = clusterAndAssign(embeddings, k, metric);
    wcsd.push(withinClusterSumOfDistances(embeddings, assignments, k, metric));
  }

  if (wcsd.length <= 2) return minK;

  // Second derivative: wcsd[i-1] - 2*wcsd[i] + wcsd[i+1]
  let bestElbow = 0;
  let bestSecondDeriv = -Infinity;
  for (let i = 1; i < wcsd.length - 1; i++) {
    const secondDeriv = wcsd[i - 1] - 2 * wcsd[i] + wcsd[i + 1];
    if (secondDeriv > bestSecondDeriv) {
      bestSecondDeriv = secondDeriv;
      bestElbow = i;
    }
  }

  return minK + bestElbow;
}

/**
 * Computes the silhouette coefficient for each k in [minK, maxK].
 * Useful for visualization and manual inspection of clustering quality.
 *
 * @param embeddings - Array of embedding vectors to analyze
 * @param options - Configuration: minK, maxK, metric
 * @returns Array of { k, silhouette } sorted by k ascending
 */
export function silhouetteByK(
  embeddings: Vector[],
  options?: Omit<OptimalKOptions, 'method'>,
): Array<{ k: number; silhouette: number }> {
  const metric = options?.metric ?? 'cosine';
  const minK = options?.minK ?? 2;
  const maxK = options?.maxK ?? Math.min(10, Math.floor(embeddings.length / 2));

  const results: Array<{ k: number; silhouette: number }> = [];
  for (let k = minK; k <= maxK; k++) {
    const assignments = clusterAndAssign(embeddings, k, metric);
    const silhouette = meanSilhouette(embeddings, assignments, k, metric);
    results.push({ k, silhouette });
  }
  return results;
}

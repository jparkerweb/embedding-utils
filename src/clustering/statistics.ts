import type { ClusterStats, SimilarityMetric, Vector } from '../types';
import { computeScore, computeDistance } from '../internal/metrics';

/**
 * Computes statistics for a cluster: similarity distribution, radius, and outliers.
 *
 * @param cluster - Object with centroid and members arrays
 * @param metric - Similarity metric (default: 'cosine')
 * @returns ClusterStats with min/max/mean/median similarity, radius, and outlier indices
 */
export function clusterStats(
  cluster: { centroid: Vector; members: Vector[] },
  metric: SimilarityMetric = 'cosine',
): ClusterStats {
  const { centroid, members } = cluster;

  if (members.length === 0) {
    return {
      minSimilarity: 0,
      maxSimilarity: 0,
      meanSimilarity: 0,
      medianSimilarity: 0,
      radius: 0,
      outliers: [],
    };
  }

  const similarities = members.map((m) => computeScore(m, centroid, metric));
  const distances = members.map((m) => computeDistance(m, centroid, metric));

  const sorted = [...similarities].sort((a, b) => a - b);
  const n = sorted.length;
  const mid = Math.floor(n / 2);
  const median = n % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];

  const mean = similarities.reduce((s, v) => s + v, 0) / n;
  const variance = similarities.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
  const stddev = Math.sqrt(variance);
  const outlierThreshold = mean - 2 * stddev;

  const outliers: number[] = [];
  for (let i = 0; i < n; i++) {
    if (similarities[i] < outlierThreshold) {
      outliers.push(i);
    }
  }

  return {
    minSimilarity: sorted[0],
    maxSimilarity: sorted[n - 1],
    meanSimilarity: mean,
    medianSimilarity: median,
    radius: Math.max(...distances),
    outliers,
  };
}

/**
 * Detects outlier members in a cluster — those with similarity to the centroid
 * below `mean - threshold * stddev`.
 *
 * @param cluster - Object with centroid and members arrays
 * @param options - threshold (number of stddevs, default: 2) and metric
 * @returns Array of member indices that are outliers
 */
export function detectOutliers(
  cluster: { centroid: Vector; members: Vector[] },
  options?: { threshold?: number; metric?: SimilarityMetric },
): number[] {
  const metric = options?.metric ?? 'cosine';
  const threshold = options?.threshold ?? 2;
  const { centroid, members } = cluster;

  if (members.length === 0) return [];

  const similarities = members.map((m) => computeScore(m, centroid, metric));
  const mean = similarities.reduce((s, v) => s + v, 0) / similarities.length;
  const variance = similarities.reduce((s, v) => s + (v - mean) ** 2, 0) / similarities.length;
  const stddev = Math.sqrt(variance);
  const cutoff = mean - threshold * stddev;

  const outliers: number[] = [];
  for (let i = 0; i < similarities.length; i++) {
    if (similarities[i] < cutoff) {
      outliers.push(i);
    }
  }
  return outliers;
}

/**
 * Computes the distance/dissimilarity between old and new centroids.
 * Useful for detecting when clusters have stabilized or shifted significantly.
 *
 * @param oldCentroid - Previous centroid vector
 * @param newCentroid - Current centroid vector
 * @param metric - Similarity metric (default: 'cosine'). Returns `1 - similarity`.
 * @returns Distance between centroids (0 = identical, ~1 = orthogonal for cosine)
 */
export function centroidDrift(
  oldCentroid: Vector,
  newCentroid: Vector,
  metric: SimilarityMetric = 'cosine',
): number {
  return computeDistance(oldCentroid, newCentroid, metric);
}

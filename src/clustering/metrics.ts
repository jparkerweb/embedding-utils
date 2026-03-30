import type { Cluster, SimilarityMetric } from '../types';
import { cosineSimilarity, dotProduct } from '../math/similarity';
import { euclideanDistance, manhattanDistance } from '../math/distance';

function getDistance(a: number[], b: number[], metric: SimilarityMetric): number {
  switch (metric) {
    case 'cosine':
      return 1 - cosineSimilarity(a, b);
    case 'dot':
      return 1 - dotProduct(a, b);
    case 'euclidean':
      return euclideanDistance(a, b);
    case 'manhattan':
      return manhattanDistance(a, b);
  }
}

function getSimilarity(a: number[], b: number[], metric: SimilarityMetric): number {
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
 * Computes the average pairwise similarity within a cluster.
 * @param cluster - The cluster to measure
 * @param metric - Similarity metric to use (defaults to 'cosine')
 * @returns Cohesion score between 0 and 1 (1.0 for single-member clusters)
 * @example
 * cohesionScore(cluster, 'cosine'); // 0.95
 */
export function cohesionScore(
  cluster: Cluster,
  metric: SimilarityMetric = 'cosine',
): number {
  if (cluster.members.length <= 1) return 1.0;

  let totalSim = 0;
  let pairs = 0;
  for (let i = 0; i < cluster.members.length; i++) {
    for (let j = i + 1; j < cluster.members.length; j++) {
      totalSim += getSimilarity(cluster.members[i], cluster.members[j], metric);
      pairs++;
    }
  }
  return totalSim / pairs;
}

/**
 * Computes the mean silhouette score across all clusters.
 * @param clusters - Array of clusters to evaluate
 * @param metric - Similarity metric to use (defaults to 'cosine')
 * @returns Silhouette score between -1 and 1 (0 for single cluster)
 * @example
 * silhouetteScore(clusters, 'cosine'); // 0.72
 */
export function silhouetteScore(
  clusters: Cluster[],
  metric: SimilarityMetric = 'cosine',
): number {
  if (clusters.length <= 1) return 0;

  let totalSilhouette = 0;
  let totalPoints = 0;

  for (let ci = 0; ci < clusters.length; ci++) {
    const cluster = clusters[ci];

    for (const member of cluster.members) {
      // a = average distance to own cluster members
      let a = 0;
      if (cluster.members.length > 1) {
        let sumDist = 0;
        for (const other of cluster.members) {
          if (other !== member) {
            sumDist += getDistance(member, other, metric);
          }
        }
        a = sumDist / (cluster.members.length - 1);
      }

      // b = min average distance to any other cluster
      let b = Infinity;
      for (let oi = 0; oi < clusters.length; oi++) {
        if (oi === ci) continue;
        let sumDist = 0;
        for (const other of clusters[oi].members) {
          sumDist += getDistance(member, other, metric);
        }
        const avgDist = sumDist / clusters[oi].members.length;
        if (avgDist < b) b = avgDist;
      }

      const s = Math.max(a, b) === 0 ? 0 : (b - a) / Math.max(a, b);
      totalSilhouette += s;
      totalPoints++;
    }
  }

  return totalSilhouette / totalPoints;
}

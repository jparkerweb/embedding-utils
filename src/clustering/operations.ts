import type { Cluster, SimilarityMetric } from '../types';
import { cosineSimilarity, dotProduct } from '../math/similarity';
import { euclideanDistance, manhattanDistance } from '../math/distance';

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

function computeCentroid(members: number[][]): number[] {
  const dims = members[0].length;
  const centroid = new Array<number>(dims).fill(0);
  for (const member of members) {
    for (let i = 0; i < dims; i++) {
      centroid[i] += member[i];
    }
  }
  for (let i = 0; i < dims; i++) {
    centroid[i] /= members.length;
  }
  return centroid;
}

function computeCohesion(members: number[][], metric: SimilarityMetric): number {
  if (members.length <= 1) return 1.0;
  let totalSim = 0;
  let pairs = 0;
  for (let i = 0; i < members.length; i++) {
    for (let j = i + 1; j < members.length; j++) {
      totalSim += getSimilarity(members[i], members[j], metric);
      pairs++;
    }
  }
  return totalSim / pairs;
}

/**
 * Assigns an embedding to the most similar cluster.
 * @param embedding - The embedding vector to assign
 * @param clusters - Array of existing clusters
 * @param options - Optional metric and minimum threshold
 * @returns Object with clusterIndex (-1 if below threshold) and similarity score
 * @example
 * assignToCluster(embedding, clusters, { threshold: 0.8 });
 * // { clusterIndex: 2, similarity: 0.92 }
 */
export function assignToCluster(
  embedding: number[],
  clusters: Cluster[],
  options?: { metric?: SimilarityMetric; threshold?: number },
): { clusterIndex: number; similarity: number } {
  const metric = options?.metric ?? 'cosine';
  const threshold = options?.threshold ?? 0;

  let bestIndex = -1;
  let bestSim = -Infinity;

  for (let i = 0; i < clusters.length; i++) {
    const sim = getSimilarity(embedding, clusters[i].centroid, metric);
    if (sim > bestSim) {
      bestSim = sim;
      bestIndex = i;
    }
  }

  if (bestSim < threshold) {
    return { clusterIndex: -1, similarity: bestSim };
  }

  return { clusterIndex: bestIndex, similarity: bestSim };
}

/**
 * Merges two clusters into one, recomputing the centroid and cohesion.
 * @param a - First cluster
 * @param b - Second cluster
 * @param metric - Similarity metric to use (defaults to 'cosine')
 * @returns A new merged cluster
 * @example
 * const merged = mergeClusters(clusterA, clusterB);
 */
export function mergeClusters(
  a: Cluster,
  b: Cluster,
  metric: SimilarityMetric = 'cosine',
): Cluster {
  const members = [...a.members, ...b.members];
  const labels =
    a.labels || b.labels
      ? [...(a.labels || []), ...(b.labels || [])]
      : undefined;

  return {
    centroid: computeCentroid(members),
    members,
    labels,
    size: members.length,
    cohesion: computeCohesion(members, metric),
  };
}

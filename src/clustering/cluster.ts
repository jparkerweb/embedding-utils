import type { Cluster, ClusteringConfig, SimilarityMetric } from '../types';
import { cosineSimilarity, dotProduct } from '../math/similarity';
import { euclideanDistance, manhattanDistance } from '../math/distance';

const DEFAULT_CONFIG: Required<ClusteringConfig> = {
  similarityThreshold: 0.9,
  minClusterSize: 5,
  maxClusters: 5,
  metric: 'cosine',
};

function getSimilarity(a: number[], b: number[], metric: SimilarityMetric): number {
  switch (metric) {
    case 'cosine':
      return cosineSimilarity(a, b);
    case 'dot':
      return dotProduct(a, b);
    case 'euclidean':
      // Convert distance to similarity: 1 / (1 + d)
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

function findMostSimilarPair(
  clusters: Cluster[],
  metric: SimilarityMetric,
): [number, number] {
  let bestI = 0;
  let bestJ = 1;
  let bestSim = -Infinity;
  for (let i = 0; i < clusters.length; i++) {
    for (let j = i + 1; j < clusters.length; j++) {
      const sim = getSimilarity(clusters[i].centroid, clusters[j].centroid, metric);
      if (sim > bestSim) {
        bestSim = sim;
        bestI = i;
        bestJ = j;
      }
    }
  }
  return [bestI, bestJ];
}

function mergeTwoClusters(a: Cluster, b: Cluster, metric: SimilarityMetric): Cluster {
  const members = [...a.members, ...b.members];
  const labels = a.labels || b.labels
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

/**
 * Clusters embeddings using greedy agglomerative clustering.
 * @param embeddings - Array of embedding vectors to cluster
 * @param config - Optional clustering configuration (threshold, min size, max clusters, metric)
 * @returns Array of clusters, each with centroid, members, size, and cohesion
 * @example
 * const clusters = clusterEmbeddings(embeddings, { similarityThreshold: 0.9 });
 */
export function clusterEmbeddings(
  embeddings: number[][],
  config?: ClusteringConfig,
): Cluster[] {
  if (embeddings.length === 0) return [];

  const cfg = { ...DEFAULT_CONFIG, ...config };
  const { similarityThreshold, minClusterSize, maxClusters, metric } = cfg;

  // Greedy agglomerative clustering
  const clusters: Cluster[] = [];

  for (const embedding of embeddings) {
    let bestClusterIdx = -1;
    let bestSim = -Infinity;

    for (let i = 0; i < clusters.length; i++) {
      const sim = getSimilarity(embedding, clusters[i].centroid, metric);
      if (sim >= similarityThreshold && sim > bestSim) {
        bestSim = sim;
        bestClusterIdx = i;
      }
    }

    if (bestClusterIdx >= 0) {
      // Assign to existing cluster
      const cluster = clusters[bestClusterIdx];
      cluster.members.push(embedding);
      cluster.size = cluster.members.length;
      cluster.centroid = computeCentroid(cluster.members);
    } else {
      // Create new cluster
      clusters.push({
        centroid: [...embedding],
        members: [embedding],
        size: 1,
        cohesion: 1.0,
      });
    }
  }

  // Filter clusters below minClusterSize
  let result = clusters.filter((c) => c.size >= minClusterSize);

  // Merge until at maxClusters limit
  while (result.length > maxClusters) {
    const [i, j] = findMostSimilarPair(result, metric);
    const merged = mergeTwoClusters(result[i], result[j], metric);
    // Remove j first (higher index) then i
    result.splice(j, 1);
    result.splice(i, 1);
    result.push(merged);
  }

  // Compute final cohesion for all clusters
  for (const cluster of result) {
    cluster.cohesion = computeCohesion(cluster.members, metric);
  }

  return result;
}

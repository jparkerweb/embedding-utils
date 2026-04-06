import type { Cluster, ClusteringConfig, SimilarityMetric, Vector } from '../types';
import { ValidationError } from '../types';
import { computeScore } from '../internal/metrics';
import { computeCentroid, computePairwiseCohesion } from '../internal/clustering';
import { toFloat32 } from '../internal/vector-utils';
import { shuffleArray } from '../internal/random';

const DEFAULT_CONFIG: Required<ClusteringConfig> = {
  similarityThreshold: 0.9,
  minClusterSize: 5,
  maxClusters: 5,
  metric: 'cosine',
  assignmentStrategy: 'centroid',
  shuffle: false,
  shuffleSeed: 42,
};

function findMostSimilarPair(
  clusters: Cluster[],
  metric: SimilarityMetric,
): [number, number] {
  let bestI = 0;
  let bestJ = 1;
  let bestSim = -Infinity;
  for (let i = 0; i < clusters.length; i++) {
    for (let j = i + 1; j < clusters.length; j++) {
      const sim = computeScore(clusters[i].centroid, clusters[j].centroid, metric);
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
    cohesion: computePairwiseCohesion(members, metric),
  };
}

/**
 * Clusters embeddings using greedy agglomerative clustering.
 *
 * @param embeddings - Array of embedding vectors to cluster
 * @param config - Optional clustering configuration
 * @param labels - Optional labels corresponding to each embedding
 * @returns Array of clusters with Float32Array centroids and members
 */
export function clusterEmbeddings(
  embeddings: Vector[],
  config?: ClusteringConfig,
  labels?: string[],
): Cluster[] {
  if (embeddings.length === 0) return [];

  if (labels && labels.length !== embeddings.length) {
    throw new ValidationError(
      `Labels length (${labels.length}) must match embeddings length (${embeddings.length})`,
    );
  }

  const cfg = { ...DEFAULT_CONFIG, ...config };

  if (cfg.maxClusters < 1) {
    throw new ValidationError(`maxClusters must be >= 1, got ${cfg.maxClusters}`);
  }
  if (cfg.minClusterSize < 0) {
    throw new ValidationError(`minClusterSize must be >= 0, got ${cfg.minClusterSize}`);
  }
  const { similarityThreshold, minClusterSize, maxClusters, metric, assignmentStrategy, shuffle, shuffleSeed } = cfg;

  // Convert all inputs to Float32Array up front
  const float32Embeddings = embeddings.map((e) => toFloat32(e));

  // Optionally shuffle input for order-independent clustering
  let input: Float32Array[];
  let inputLabels: string[] | undefined;
  if (shuffle) {
    const indices = Array.from({ length: float32Embeddings.length }, (_, i) => i);
    const shuffled = shuffleArray(indices, shuffleSeed);
    input = shuffled.map((i) => float32Embeddings[i]);
    inputLabels = labels ? shuffled.map((i) => labels[i]) : undefined;
  } else {
    input = float32Embeddings;
    inputLabels = labels;
  }

  // Greedy agglomerative clustering
  const clusters: Cluster[] = [];

  for (let idx = 0; idx < input.length; idx++) {
    const embedding = input[idx];
    let bestClusterIdx = -1;
    let bestSim = -Infinity;

    for (let i = 0; i < clusters.length; i++) {
      let sim: number;
      if (assignmentStrategy === 'average-similarity') {
        let total = 0;
        for (const member of clusters[i].members) {
          total += computeScore(embedding, member, metric);
        }
        sim = total / clusters[i].members.length;
      } else {
        sim = computeScore(embedding, clusters[i].centroid, metric);
      }
      if (sim >= similarityThreshold && sim > bestSim) {
        bestSim = sim;
        bestClusterIdx = i;
      }
    }

    if (bestClusterIdx >= 0) {
      // Assign to existing cluster with incremental centroid update O(d)
      const cluster = clusters[bestClusterIdx];
      cluster.members.push(embedding);
      if (inputLabels) {
        if (!cluster.labels) cluster.labels = [];
        cluster.labels.push(inputLabels[idx]);
      }
      cluster.size = cluster.members.length;
      const newSize = cluster.size;
      const centroid = cluster.centroid;
      for (let d = 0; d < centroid.length; d++) {
        centroid[d] += (embedding[d] - centroid[d]) / newSize;
      }
    } else {
      // Create new cluster
      clusters.push({
        centroid: new Float32Array(embedding),
        members: [embedding],
        labels: inputLabels ? [inputLabels[idx]] : undefined,
        size: 1,
        cohesion: 1.0,
      });
    }
  }

  // Separate clusters into valid (>= minClusterSize) and small (< minClusterSize).
  const validClusters = clusters.filter((c) => c.size >= minClusterSize);
  const smallClusters = clusters.filter((c) => c.size < minClusterSize);

  let result: Cluster[];

  if (validClusters.length > 0) {
    // Redistribute small cluster members into the nearest valid cluster
    for (const small of smallClusters) {
      for (let m = 0; m < small.members.length; m++) {
        const member = small.members[m];
        let bestIdx = 0;
        let bestSim = -Infinity;
        for (let i = 0; i < validClusters.length; i++) {
          const sim = computeScore(member, validClusters[i].centroid, metric);
          if (sim > bestSim) {
            bestSim = sim;
            bestIdx = i;
          }
        }
        validClusters[bestIdx].members.push(member);
        validClusters[bestIdx].size = validClusters[bestIdx].members.length;
        if (small.labels) {
          if (!validClusters[bestIdx].labels) validClusters[bestIdx].labels = [];
          validClusters[bestIdx].labels!.push(small.labels[m]);
        }
      }
    }
    // Recompute centroids after redistribution
    for (const cluster of validClusters) {
      cluster.centroid = computeCentroid(cluster.members);
    }
    result = validClusters;
  } else if (clusters.length > 0) {
    // All clusters are small -- combine everything into one cluster
    const allMembers: Float32Array[] = [];
    const allLabels: string[] = [];
    let hasLabels = false;
    for (const cluster of clusters) {
      allMembers.push(...cluster.members);
      if (cluster.labels) {
        hasLabels = true;
        allLabels.push(...cluster.labels);
      }
    }
    result = [{
      centroid: computeCentroid(allMembers),
      members: allMembers,
      labels: hasLabels ? allLabels : undefined,
      size: allMembers.length,
      cohesion: 1.0,
    }];
  } else {
    result = [];
  }

  // Merge until at maxClusters limit
  while (result.length > maxClusters) {
    const [i, j] = findMostSimilarPair(result, metric);
    const merged = mergeTwoClusters(result[i], result[j], metric);
    result.splice(j, 1);
    result.splice(i, 1);
    result.push(merged);
  }

  // Compute final cohesion for all clusters
  for (const cluster of result) {
    cluster.cohesion = computePairwiseCohesion(cluster.members, metric);
  }

  return result;
}

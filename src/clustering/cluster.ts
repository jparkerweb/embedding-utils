import type { Cluster, ClusteringConfig, SimilarityMetric } from '../types';
import { ValidationError } from '../types';
import { computeScore } from '../internal/metrics';
import { computeCentroid, computePairwiseCohesion } from '../internal/clustering';
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
 * ## Algorithm
 *
 * 1. **Greedy assignment:** Iterate through embeddings sequentially. For each
 *    embedding, find the most similar existing cluster centroid. If the
 *    similarity meets the threshold, assign it to that cluster and recompute
 *    the centroid. Otherwise, create a new cluster.
 *
 * 2. **Small-cluster redistribution:** Clusters with fewer members than
 *    `minClusterSize` have their members redistributed to the nearest valid
 *    cluster. If ALL clusters are small, they are combined into one cluster.
 *    This ensures no data points are silently discarded.
 *
 * 3. **Merge to limit:** If there are still more clusters than `maxClusters`,
 *    the two most similar clusters (by centroid similarity) are merged
 *    repeatedly until the limit is reached.
 *
 * 4. **Final cohesion:** Pairwise cohesion is computed for all resulting clusters.
 *
 * ## Configuration
 *
 * Use {@link getPreset} for common configurations, or pass a custom
 * {@link ClusteringConfig} object. Defaults:
 * - `similarityThreshold`: 0.9
 * - `minClusterSize`: 5
 * - `maxClusters`: 5
 * - `metric`: 'cosine'
 *
 * ## Use cases
 *
 * - **Topic discovery:** Group support tickets, reviews, or documents into
 *   natural topic clusters without predefined categories.
 * - **Training data clustering:** Group training phrases by semantic similarity
 *   to create multiple weighted-average embeddings per topic (as used by
 *   fast-topic-analysis).
 * - **Content deduplication:** Cluster near-identical content and keep only
 *   cluster centroids.
 *
 * @param embeddings - Array of embedding vectors to cluster
 * @param config - Optional clustering configuration (threshold, min size, max clusters, metric)
 * @param labels - Optional labels corresponding to each embedding (must match length).
 *                 Labels are preserved through clustering, redistribution, and merging,
 *                 so each cluster's `labels` array maps 1:1 to its `members` array.
 * @returns Array of clusters, each with centroid, members, labels (if provided), size, and cohesion.
 *          Returns empty array for empty input.
 *
 * @example
 * // Basic clustering with defaults
 * const clusters = clusterEmbeddings(embeddings);
 *
 * @example
 * // Custom configuration
 * const clusters = clusterEmbeddings(embeddings, {
 *   similarityThreshold: 0.85,
 *   minClusterSize: 3,
 *   maxClusters: 10,
 *   metric: 'cosine',
 * });
 *
 * @example
 * // Using a preset
 * const clusters = clusterEmbeddings(embeddings, getPreset('high-precision'));
 *
 * @example
 * // Disable clustering (legacy mode — single cluster with all embeddings)
 * const clusters = clusterEmbeddings(embeddings, getPreset('legacy'));
 *
 * @example
 * // With labels to track which phrases belong to which cluster
 * const phrases = ['hello world', 'goodbye world', 'hi there'];
 * const clusters = clusterEmbeddings(embeddings, {}, phrases);
 * clusters[0].labels; // ['hello world', 'hi there'] — phrases in this cluster
 */
export function clusterEmbeddings(
  embeddings: number[][],
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

  // Optionally shuffle input for order-independent clustering
  // When labels are provided, we need to keep indices synchronized
  let input: number[][];
  let inputLabels: string[] | undefined;
  if (shuffle) {
    const indices = Array.from({ length: embeddings.length }, (_, i) => i);
    const shuffled = shuffleArray(indices, shuffleSeed);
    input = shuffled.map((i) => embeddings[i]);
    inputLabels = labels ? shuffled.map((i) => labels[i]) : undefined;
  } else {
    input = embeddings;
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
        // Average similarity to all current members
        let total = 0;
        for (const member of clusters[i].members) {
          total += computeScore(embedding, member, metric);
        }
        sim = total / clusters[i].members.length;
      } else {
        // Default: compare to centroid
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
        centroid: [...embedding],
        members: [embedding],
        labels: inputLabels ? [inputLabels[idx]] : undefined,
        size: 1,
        cohesion: 1.0,
      });
    }
  }

  // Separate clusters into valid (>= minClusterSize) and small (< minClusterSize).
  // Small clusters have their members redistributed to the nearest valid cluster
  // rather than being silently dropped. This preserves all data points and matches
  // the behavior of topic analysis pipelines like fast-topic-analysis.
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
    const allMembers: number[][] = [];
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
    // Remove j first (higher index) then i
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

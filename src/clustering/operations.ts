import type { Cluster, SimilarityMetric } from '../types';
import { computeScore } from '../internal/metrics';
import { computeCentroid, computePairwiseCohesion } from '../internal/clustering';

/**
 * Assigns an embedding to the most similar existing cluster.
 *
 * This is the core function for **live classification** — when a new document,
 * support ticket, or data point arrives, classify it into an existing cluster
 * without re-clustering everything.
 *
 * **How it works:** Computes the similarity between the embedding and each
 * cluster's centroid, then returns the index of the best match. If the best
 * match is below the threshold, returns -1 (the embedding is an outlier or
 * represents a new topic).
 *
 * **Incremental pipeline usage:** After assigning to a cluster, update the
 * centroid using {@link batchIncrementalAverage} to keep the cluster
 * representation current without re-processing all historical data.
 *
 * @param embedding - The embedding vector to assign
 * @param clusters - Array of existing clusters (must have centroids)
 * @param options - Optional metric ('cosine' default) and minimum threshold (0 default)
 * @returns Object with:
 *   - `clusterIndex`: Index into the clusters array, or -1 if below threshold
 *   - `similarity`: Similarity score to the best-matching cluster centroid
 *
 * @example
 * const { clusterIndex, similarity } = assignToCluster(newEmbedding, clusters, {
 *   threshold: 0.8,
 * });
 * if (clusterIndex === -1) {
 *   console.log('New topic detected — no cluster match');
 * }
 *
 * @example
 * // Incremental topic pipeline: assign + update centroid
 * const { clusterIndex } = assignToCluster(newEmbed, clusters);
 * if (clusterIndex >= 0) {
 *   clusters[clusterIndex].centroid = batchIncrementalAverage(
 *     clusters[clusterIndex].centroid,
 *     [newEmbed],
 *     clusters[clusterIndex].size,
 *   );
 *   clusters[clusterIndex].size++;
 * }
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
    const sim = computeScore(embedding, clusters[i].centroid, metric);
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
 *
 * **When to use:** Manual cluster refinement after auto-clustering. For
 * example, after auto-clustering support tickets, you notice "password reset"
 * and "login issues" ended up as separate clusters but your team treats them
 * as one topic. Merge them programmatically.
 *
 * The merged cluster:
 * - Combines all members from both clusters
 * - Concatenates labels (if either cluster has them)
 * - Recomputes the centroid as the mean of all combined members
 * - Recomputes pairwise cohesion for the combined members
 *
 * @param a - First cluster to merge
 * @param b - Second cluster to merge
 * @param metric - Similarity metric for cohesion computation (defaults to 'cosine')
 * @returns A new merged cluster (original clusters are not modified)
 *
 * @example
 * const merged = mergeClusters(passwordCluster, loginCluster);
 * console.log(`Merged: ${merged.size} items, cohesion ${merged.cohesion.toFixed(2)}`);
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
    cohesion: computePairwiseCohesion(members, metric),
  };
}

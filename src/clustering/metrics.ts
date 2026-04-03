import type { Cluster, SimilarityMetric } from '../types';
import { computeScore, computeDistance } from '../internal/metrics';
import { computePairwiseCohesion } from '../internal/clustering';

/**
 * Computes the average **pairwise** similarity within a cluster.
 *
 * This measures how similar every member is to every other member, providing
 * a thorough assessment of internal cluster quality. A high score means all
 * members are semantically close to each other (tight cluster).
 *
 * **Formula:** Average of `similarity(member_i, member_j)` for all i < j pairs.
 *
 * **Complexity:** O(n^2) where n = cluster.members.length. For large clusters,
 * consider using {@link centroidCohesion} instead (O(n) complexity).
 *
 * **Comparison with centroidCohesion:**
 * - `cohesionScore` (this function): Pairwise — measures member-to-member similarity
 * - `centroidCohesion`: Centroid — measures member-to-centroid similarity
 *
 * Both are valid metrics. Pairwise cohesion is stricter (catches cases where
 * members are spread around the centroid but far from each other).
 *
 * @param cluster - The cluster to measure (must have centroid and members)
 * @param metric - Similarity metric to use (defaults to 'cosine')
 * @returns Cohesion score between 0 and 1. Returns 1.0 for single-member clusters.
 *
 * @example
 * const quality = cohesionScore(cluster);
 * if (quality < 0.7) console.warn('Cluster is loosely grouped');
 */
export function cohesionScore(
  cluster: Cluster,
  metric: SimilarityMetric = 'cosine',
): number {
  return computePairwiseCohesion(cluster.members, metric);
}

/**
 * Computes the average similarity of all cluster members to the cluster centroid.
 *
 * This is a different cohesion metric than {@link cohesionScore} (which uses pairwise
 * similarity). Centroid-based cohesion measures how tightly members cluster around
 * the center point, and is the formula used by topic analysis pipelines like
 * fast-topic-analysis.
 *
 * - A score of 1.0 means all members are identical to the centroid (perfect cohesion).
 * - A score near 0 means members are orthogonal to the centroid (poor cohesion).
 * - Returns 1.0 for single-member clusters (trivially cohesive).
 *
 * This metric is computationally cheaper than pairwise cohesion: O(n) comparisons
 * vs O(n^2), making it preferable for large clusters.
 *
 * @param cluster - The cluster to measure (must have centroid and members)
 * @param metric - Similarity metric to use (defaults to 'cosine')
 * @returns Cohesion score, typically between 0 and 1 for cosine metric
 *
 * @example
 * const score = centroidCohesion(cluster);           // default cosine
 * const score = centroidCohesion(cluster, 'dot');     // dot product
 *
 * @example
 * // Quality check after clustering
 * for (const cluster of clusters) {
 *   const quality = centroidCohesion(cluster);
 *   if (quality < 0.7) console.warn('Loose cluster detected');
 * }
 */
export function centroidCohesion(
  cluster: Cluster,
  metric: SimilarityMetric = 'cosine',
): number {
  if (cluster.members.length <= 1) return 1.0;

  let totalSim = 0;
  for (const member of cluster.members) {
    totalSim += computeScore(member, cluster.centroid, metric);
  }
  return totalSim / cluster.members.length;
}

/**
 * Computes the mean silhouette score across all clusters.
 *
 * The silhouette score evaluates **overall clustering quality** by measuring
 * how well each data point fits within its assigned cluster compared to the
 * nearest neighboring cluster.
 *
 * **Score interpretation:**
 * - Near +1.0: Clusters are well-separated and internally cohesive (excellent)
 * - Near 0.0: Points are on or near decision boundaries (ambiguous clustering)
 * - Near -1.0: Points are in the wrong cluster (poor clustering)
 *
 * **Use this to tune your `similarityThreshold`:** Run clustering at several
 * threshold values and pick the one that maximizes silhouette score.
 *
 * **Algorithm (per point):**
 * 1. `a` = average distance to own cluster members (intra-cluster)
 * 2. `b` = minimum average distance to any other cluster (nearest-cluster)
 * 3. `silhouette = (b - a) / max(a, b)`
 * 4. Returns the mean across all points in all clusters.
 *
 * @param clusters - Array of clusters to evaluate
 * @param metric - Similarity metric to use (defaults to 'cosine')
 * @returns Silhouette score between -1 and 1. Returns 0 for a single cluster.
 *
 * @example
 * const quality = silhouetteScore(clusters);
 * console.log(`Clustering quality: ${quality.toFixed(2)}`); // e.g., 0.72
 *
 * @example
 * // Compare different thresholds to find the optimal clustering
 * for (const threshold of [0.7, 0.8, 0.85, 0.9, 0.95]) {
 *   const clusters = clusterEmbeddings(embeddings, { similarityThreshold: threshold });
 *   console.log(`threshold=${threshold}: silhouette=${silhouetteScore(clusters).toFixed(3)}`);
 * }
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
            sumDist += computeDistance(member, other, metric);
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
          sumDist += computeDistance(member, other, metric);
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

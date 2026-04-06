import type { Cluster, ClusteringConfig, ClusterStats, SimilarityMetric, Vector } from '../types';
import { computeScore } from '../internal/metrics';
import { computePairwiseCohesion } from '../internal/clustering';
import { toFloat32 } from '../internal/vector-utils';
import { incrementalAverage } from '../aggregation/average';
import { clusterEmbeddings } from './cluster';
import { clusterStats } from './statistics';

interface InternalCluster {
  centroid: Float32Array;
  members: Float32Array[];
  labels: string[];
}

/**
 * Online clustering that assigns embeddings incrementally to the nearest cluster
 * or creates new clusters when no existing cluster is similar enough.
 *
 * Use {@link rebalance} to re-optimize assignments via full reclustering.
 */
export class IncrementalClusterer {
  private clusters: InternalCluster[] = [];
  private config: ClusteringConfig;
  private metric: SimilarityMetric;
  private threshold: number;
  private totalSize = 0;

  constructor(config?: ClusteringConfig) {
    this.config = config ?? {};
    this.metric = config?.metric ?? 'cosine';
    this.threshold = config?.similarityThreshold ?? 0.9;
  }

  /**
   * Assigns an embedding to the nearest cluster or creates a new one if
   * similarity to all existing centroids is below the threshold.
   */
  addEmbedding(embedding: Vector, label?: string): void {
    const f32 = toFloat32(embedding);
    let bestIdx = -1;
    let bestScore = -Infinity;

    for (let i = 0; i < this.clusters.length; i++) {
      const score = computeScore(f32, this.clusters[i].centroid, this.metric);
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }

    if (bestIdx >= 0 && bestScore >= this.threshold) {
      const cluster = this.clusters[bestIdx];
      cluster.centroid = incrementalAverage(cluster.centroid, f32, cluster.members.length);
      cluster.members.push(f32);
      if (label !== undefined) cluster.labels.push(label);
    } else {
      this.clusters.push({
        centroid: new Float32Array(f32),
        members: [f32],
        labels: label !== undefined ? [label] : [],
      });
    }

    this.totalSize++;
  }

  /** Calls {@link addEmbedding} for each item in the batch. */
  addBatch(embeddings: Vector[], labels?: string[]): void {
    for (let i = 0; i < embeddings.length; i++) {
      this.addEmbedding(embeddings[i], labels?.[i]);
    }
  }

  /**
   * Re-optimizes cluster assignments by running the full clustering algorithm
   * on all accumulated embeddings.
   */
  rebalance(): void {
    if (this.totalSize === 0) return;

    const allMembers: Float32Array[] = [];
    const allLabels: string[] = [];

    for (const cluster of this.clusters) {
      for (let i = 0; i < cluster.members.length; i++) {
        allMembers.push(cluster.members[i]);
        allLabels.push(cluster.labels[i] ?? '');
      }
    }

    const result = clusterEmbeddings(allMembers, {
      ...this.config,
      maxClusters: this.config.maxClusters ?? this.clusters.length,
    });

    this.clusters = result.map((c) => ({
      centroid: c.centroid,
      members: c.members,
      labels: c.labels ?? [],
    }));
  }

  /** Returns the current cluster state as an array of {@link Cluster} objects. */
  getClusters(): Cluster[] {
    return this.clusters.map((c) => ({
      centroid: c.centroid,
      members: c.members,
      labels: c.labels.length > 0 ? c.labels : undefined,
      size: c.members.length,
      cohesion: computePairwiseCohesion(c.members, this.metric),
    }));
  }

  /** Returns statistics for each cluster. */
  getStats(): ClusterStats[] {
    return this.clusters.map((c) => clusterStats(c, this.metric));
  }

  /** Total number of embeddings added. */
  get size(): number {
    return this.totalSize;
  }
}

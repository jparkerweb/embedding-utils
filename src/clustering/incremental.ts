import type { Cluster, ClusteringConfig, ClusterStats, SimilarityMetric } from '../types';
import { computeScore } from '../internal/metrics';
import { computePairwiseCohesion } from '../internal/clustering';
import { incrementalAverage } from '../aggregation/average';
import { clusterEmbeddings } from './cluster';
import { clusterStats } from './statistics';

interface InternalCluster {
  centroid: number[];
  members: number[][];
  labels: string[];
}

/**
 * Online clustering that assigns embeddings incrementally to the nearest cluster
 * or creates new clusters when no existing cluster is similar enough.
 *
 * Use {@link rebalance} to re-optimize assignments via full reclustering.
 *
 * @example
 * const clusterer = new IncrementalClusterer({ similarityThreshold: 0.8 });
 * clusterer.addEmbedding([0.1, 0.2, 0.3], 'doc-1');
 * clusterer.addEmbedding([0.1, 0.2, 0.31], 'doc-2'); // joins existing cluster
 * clusterer.addEmbedding([0.9, 0.8, 0.7], 'doc-3'); // creates new cluster
 * console.log(clusterer.getClusters()); // 2 clusters
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
  addEmbedding(embedding: number[], label?: string): void {
    let bestIdx = -1;
    let bestScore = -Infinity;

    for (let i = 0; i < this.clusters.length; i++) {
      const score = computeScore(embedding, this.clusters[i].centroid, this.metric);
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }

    if (bestIdx >= 0 && bestScore >= this.threshold) {
      const cluster = this.clusters[bestIdx];
      cluster.centroid = incrementalAverage(cluster.centroid, embedding, cluster.members.length);
      cluster.members.push(embedding);
      if (label !== undefined) cluster.labels.push(label);
    } else {
      this.clusters.push({
        centroid: embedding.slice(),
        members: [embedding],
        labels: label !== undefined ? [label] : [],
      });
    }

    this.totalSize++;
  }

  /** Calls {@link addEmbedding} for each item in the batch. */
  addBatch(embeddings: number[][], labels?: string[]): void {
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

    const allMembers: number[][] = [];
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

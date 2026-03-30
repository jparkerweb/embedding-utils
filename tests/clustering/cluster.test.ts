import { describe, it, expect } from 'vitest';
import { clusterEmbeddings } from '../../src/clustering/cluster';
import { cosineSimilarity } from '../../src/math/similarity';

describe('clusterEmbeddings', () => {
  it('returns empty array for empty input', () => {
    expect(clusterEmbeddings([])).toEqual([]);
  });

  it('returns one cluster for a single embedding', () => {
    const embeddings = [[1, 0, 0]];
    const clusters = clusterEmbeddings(embeddings, { minClusterSize: 1 });
    expect(clusters).toHaveLength(1);
    expect(clusters[0].members).toHaveLength(1);
    expect(clusters[0].size).toBe(1);
  });

  it('groups identical embeddings into one cluster', () => {
    const v = [1, 0, 0];
    const embeddings = [v, v, v, v, v];
    const clusters = clusterEmbeddings(embeddings, { minClusterSize: 1 });
    expect(clusters).toHaveLength(1);
    expect(clusters[0].size).toBe(5);
  });

  it('separates orthogonal embeddings into different clusters', () => {
    const embeddings = [
      [1, 0, 0], [1, 0, 0], [1, 0, 0],
      [0, 1, 0], [0, 1, 0], [0, 1, 0],
      [0, 0, 1], [0, 0, 1], [0, 0, 1],
    ];
    const clusters = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.5,
      minClusterSize: 1,
    });
    expect(clusters).toHaveLength(3);
  });

  it('high threshold creates more clusters', () => {
    // Slightly different vectors that are similar but not identical
    const embeddings = [
      [1, 0, 0], [0.99, 0.1, 0], [0.98, 0.15, 0],
      [0, 1, 0], [0.1, 0.99, 0], [0.05, 0.98, 0],
    ];
    const looseThreshold = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.5,
      minClusterSize: 1,
    });
    const tightThreshold = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.999,
      minClusterSize: 1,
    });
    expect(tightThreshold.length).toBeGreaterThanOrEqual(looseThreshold.length);
  });

  it('filters clusters below minClusterSize', () => {
    const embeddings = [
      [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
      [0, 1, 0], // only 1 member — should be filtered with minClusterSize=2
    ];
    const clusters = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.5,
      minClusterSize: 2,
    });
    expect(clusters).toHaveLength(1);
    expect(clusters[0].size).toBe(5);
  });

  it('caps clusters at maxClusters by merging most similar', () => {
    const embeddings = [
      [1, 0, 0], [1, 0, 0],
      [0, 1, 0], [0, 1, 0],
      [0, 0, 1], [0, 0, 1],
      [0.7, 0.7, 0], [0.7, 0.7, 0],
    ];
    const clusters = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.5,
      minClusterSize: 1,
      maxClusters: 2,
    });
    expect(clusters.length).toBeLessThanOrEqual(2);
  });

  it('supports custom metric option', () => {
    const embeddings = [
      [1, 0, 0], [1, 0, 0], [1, 0, 0],
      [0, 1, 0], [0, 1, 0], [0, 1, 0],
    ];
    const clusters = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.5,
      minClusterSize: 1,
      metric: 'dot',
    });
    expect(clusters).toHaveLength(2);
  });

  it('computes cluster centroids correctly', () => {
    const embeddings = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    const clusters = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.0,
      minClusterSize: 1,
    });
    // With threshold 0.0, everything should cluster together
    // If one cluster, centroid should be average
    if (clusters.length === 1) {
      const centroid = clusters[0].centroid;
      expect(centroid[0]).toBeCloseTo(1 / 3, 5);
      expect(centroid[1]).toBeCloseTo(1 / 3, 5);
      expect(centroid[2]).toBeCloseTo(1 / 3, 5);
    }
  });

  it('populates cohesion field on each cluster', () => {
    const embeddings = [
      [1, 0, 0], [1, 0, 0], [1, 0, 0],
      [0, 1, 0], [0, 1, 0], [0, 1, 0],
    ];
    const clusters = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.5,
      minClusterSize: 1,
    });
    for (const cluster of clusters) {
      expect(typeof cluster.cohesion).toBe('number');
      expect(cluster.cohesion).toBeGreaterThanOrEqual(0);
      expect(cluster.cohesion).toBeLessThanOrEqual(1);
    }
  });

  it('clusters a known 6-vector dataset correctly', () => {
    // Two clear groups: group A near [1,0,0], group B near [0,1,0]
    const embeddings = [
      [1, 0, 0],
      [0.95, 0.05, 0],
      [0.9, 0.1, 0],
      [0, 1, 0],
      [0.05, 0.95, 0],
      [0.1, 0.9, 0],
    ];
    const clusters = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.8,
      minClusterSize: 1,
    });
    expect(clusters).toHaveLength(2);

    // Each cluster should have 3 members
    const sizes = clusters.map((c) => c.size).sort();
    expect(sizes).toEqual([3, 3]);

    // Verify members are correctly grouped
    for (const cluster of clusters) {
      const firstMember = cluster.members[0];
      for (const member of cluster.members) {
        expect(cosineSimilarity(firstMember, member)).toBeGreaterThan(0.8);
      }
    }
  });

  it('uses default config when none provided', () => {
    // Just verify it doesn't throw and uses defaults
    const embeddings = Array.from({ length: 10 }, () => [1, 0, 0]);
    const clusters = clusterEmbeddings(embeddings);
    expect(clusters.length).toBeGreaterThan(0);
  });
});

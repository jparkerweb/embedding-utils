import { describe, it, expect } from 'vitest';
import { clusterEmbeddings } from '../../src/clustering/cluster';

describe('cluster redistribution edge cases', () => {
  it('redistributes when all clusters are below minimum size', () => {
    // Create embeddings spread across different directions so initial clustering
    // creates multiple small clusters, all below minClusterSize
    const embeddings = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [-1, 0, 0],
      [0, -1, 0],
    ];

    const clusters = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.95,  // very high so each gets its own cluster initially
      minClusterSize: 10,         // larger than total count, forcing all-small path
      maxClusters: 5,
    });

    // All should be combined into a single cluster
    expect(clusters).toHaveLength(1);
    expect(clusters[0].size).toBe(embeddings.length);
  });

  it('preserves all members after redistribution', () => {
    const embeddings = [
      [1, 0, 0],
      [0.99, 0.01, 0],   // close to first
      [0.98, 0.02, 0],   // close to first
      [0, 1, 0],          // outlier, will be in small cluster
    ];

    const clusters = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.9,
      minClusterSize: 3,
      maxClusters: 5,
    });

    // All members should be preserved across all clusters
    const totalMembers = clusters.reduce((sum, c) => sum + c.size, 0);
    expect(totalMembers).toBe(embeddings.length);
  });

  it('does not redistribute when minClusterSize is 1', () => {
    const embeddings = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];

    const clusters = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.99, // very high — likely one per cluster
      minClusterSize: 1,         // no redistribution needed
      maxClusters: 10,
    });

    // With minClusterSize=1, every cluster is valid. Each direction should form its own cluster.
    expect(clusters.length).toBeGreaterThanOrEqual(1);
    const totalMembers = clusters.reduce((sum, c) => sum + c.size, 0);
    expect(totalMembers).toBe(embeddings.length);
  });

  it('handles redistribution when some clusters are valid and some are small', () => {
    // 3 similar vectors (will form valid cluster with minSize=3) + 1 outlier (small cluster)
    const embeddings = [
      [1, 0, 0],
      [0.98, 0.02, 0],
      [0.97, 0.03, 0],
      [0, 0, 1],         // outlier
    ];

    const clusters = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.9,
      minClusterSize: 3,
      maxClusters: 5,
    });

    // Outlier should be redistributed into the valid cluster
    const totalMembers = clusters.reduce((sum, c) => sum + c.size, 0);
    expect(totalMembers).toBe(embeddings.length);
    for (const c of clusters) {
      expect(c.size).toBeGreaterThanOrEqual(3);
    }
  });
});

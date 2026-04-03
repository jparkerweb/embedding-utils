import { describe, it, expect } from 'vitest';
import { IncrementalClusterer } from '../../src/clustering/incremental';

describe('integration: IncrementalClusterer pipeline', () => {
  it('adds embeddings one by one, rebalances, and verifies cluster quality', () => {
    const clusterer = new IncrementalClusterer({
      similarityThreshold: 0.7,
      minClusterSize: 1,
      maxClusters: 3,
    });

    // Group A: near [1, 0, 0]
    const groupA = [
      [0.95, 0.05, 0.0],
      [0.90, 0.10, 0.0],
      [0.92, 0.08, 0.0],
      [0.88, 0.12, 0.0],
      [0.93, 0.07, 0.0],
    ];

    // Group B: near [0, 1, 0]
    const groupB = [
      [0.05, 0.95, 0.0],
      [0.10, 0.90, 0.0],
      [0.08, 0.92, 0.0],
      [0.12, 0.88, 0.0],
      [0.07, 0.93, 0.0],
    ];

    // Add one by one (interleaved)
    for (let i = 0; i < 5; i++) {
      clusterer.addEmbedding(groupA[i], `a${i}`);
      clusterer.addEmbedding(groupB[i], `b${i}`);
    }

    expect(clusterer.size).toBe(10);

    const beforeRebalance = clusterer.getClusters();
    expect(beforeRebalance.length).toBeGreaterThanOrEqual(2);

    // Record cohesion before rebalance
    const cohesionBefore = beforeRebalance.map((c) => c.cohesion);

    // Rebalance
    clusterer.rebalance();

    const afterRebalance = clusterer.getClusters();
    expect(afterRebalance.length).toBeGreaterThanOrEqual(1);

    // Total members should be preserved
    const totalMembers = afterRebalance.reduce((sum, c) => sum + c.size, 0);
    expect(totalMembers).toBe(10);

    // After rebalance, all clusters should have non-zero cohesion
    for (const cluster of afterRebalance) {
      expect(cluster.cohesion).toBeGreaterThan(0);
    }

    // getStats should return stats for each cluster
    const stats = clusterer.getStats();
    expect(stats.length).toBe(afterRebalance.length);
    for (const s of stats) {
      expect(s.meanSimilarity).toBeGreaterThan(0);
    }
  });

  it('handles adding all similar embeddings to a single cluster', () => {
    const clusterer = new IncrementalClusterer({
      similarityThreshold: 0.5,
    });

    // All very similar
    for (let i = 0; i < 10; i++) {
      clusterer.addEmbedding([1, 0.01 * i, 0]);
    }

    expect(clusterer.getClusters()).toHaveLength(1);
    expect(clusterer.size).toBe(10);
  });
});

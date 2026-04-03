import { describe, it, expect } from 'vitest';
import { IncrementalClusterer } from '../../src/clustering/incremental';

describe('IncrementalClusterer', () => {
  it('creates a cluster when adding the first embedding', () => {
    const clusterer = new IncrementalClusterer({ similarityThreshold: 0.8 });
    clusterer.addEmbedding([1, 0, 0], 'first');
    expect(clusterer.size).toBe(1);
    const clusters = clusterer.getClusters();
    expect(clusters).toHaveLength(1);
    expect(clusters[0].size).toBe(1);
    expect(clusters[0].labels).toEqual(['first']);
  });

  it('adds a similar embedding to the same cluster', () => {
    const clusterer = new IncrementalClusterer({ similarityThreshold: 0.8 });
    clusterer.addEmbedding([1, 0, 0], 'a');
    clusterer.addEmbedding([0.99, 0.01, 0], 'b');
    expect(clusterer.size).toBe(2);
    const clusters = clusterer.getClusters();
    expect(clusters).toHaveLength(1);
    expect(clusters[0].size).toBe(2);
  });

  it('creates a new cluster for dissimilar embeddings', () => {
    const clusterer = new IncrementalClusterer({ similarityThreshold: 0.8 });
    clusterer.addEmbedding([1, 0, 0]);
    clusterer.addEmbedding([0, 1, 0]);
    expect(clusterer.size).toBe(2);
    const clusters = clusterer.getClusters();
    expect(clusters).toHaveLength(2);
  });

  it('rebalance re-optimizes cluster assignments', () => {
    const clusterer = new IncrementalClusterer({
      similarityThreshold: 0.5,
      minClusterSize: 1,
      maxClusters: 2,
    });

    // Add embeddings that form two natural groups
    clusterer.addEmbedding([1, 0, 0]);
    clusterer.addEmbedding([0.95, 0.05, 0]);
    clusterer.addEmbedding([0, 1, 0]);
    clusterer.addEmbedding([0.05, 0.95, 0]);

    const beforeRebalance = clusterer.getClusters();
    clusterer.rebalance();
    const afterRebalance = clusterer.getClusters();

    // Should still have reasonable clusters after rebalance
    expect(afterRebalance.length).toBeGreaterThanOrEqual(1);
    expect(afterRebalance.length).toBeLessThanOrEqual(2);
  });

  it('getStats returns per-cluster statistics', () => {
    const clusterer = new IncrementalClusterer({ similarityThreshold: 0.5 });
    clusterer.addEmbedding([1, 0, 0]);
    clusterer.addEmbedding([0.95, 0.05, 0]);
    clusterer.addEmbedding([0, 1, 0]);

    const stats = clusterer.getStats();
    expect(stats.length).toBeGreaterThanOrEqual(1);
    for (const s of stats) {
      expect(s).toHaveProperty('meanSimilarity');
      expect(s).toHaveProperty('radius');
      expect(s).toHaveProperty('outliers');
    }
  });

  it('addBatch adds multiple embeddings', () => {
    const clusterer = new IncrementalClusterer({ similarityThreshold: 0.8 });
    clusterer.addBatch(
      [[1, 0, 0], [0.99, 0.01, 0], [0, 1, 0]],
      ['a', 'b', 'c'],
    );
    expect(clusterer.size).toBe(3);
    const clusters = clusterer.getClusters();
    expect(clusters.length).toBeGreaterThanOrEqual(2);
  });

  it('returns empty clusters and stats when nothing has been added', () => {
    const clusterer = new IncrementalClusterer();
    expect(clusterer.size).toBe(0);
    expect(clusterer.getClusters()).toEqual([]);
    expect(clusterer.getStats()).toEqual([]);
  });

  it('rebalance on empty clusterer is a no-op', () => {
    const clusterer = new IncrementalClusterer();
    clusterer.rebalance();
    expect(clusterer.size).toBe(0);
  });
});

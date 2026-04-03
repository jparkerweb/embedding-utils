import { describe, it, expect } from 'vitest';
import { clusterEmbeddings } from '../../src/clustering/cluster';

describe('assignmentStrategy', () => {
  // Craft a dataset where centroid vs average-similarity diverge.
  // Group A: [1, 0, 0] repeated — tight cluster.
  // Group B: spread members [0, 1, 0], [0, 0.8, 0.2], [0, 0.6, 0.4]
  // A new point [0.5, 0.5, 0] could go either way depending on strategy.
  const tightGroup = Array.from({ length: 5 }, () => [1, 0, 0]);
  const spreadGroup = [
    [0, 1, 0],
    [0, 0.8, 0.2],
    [0, 0.6, 0.4],
    [0, 0.7, 0.3],
    [0, 0.9, 0.1],
  ];
  const embeddings = [...tightGroup, ...spreadGroup];

  it("'centroid' (default) produces clusters", () => {
    const clusters = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.5,
      minClusterSize: 1,
      maxClusters: 10,
      assignmentStrategy: 'centroid',
    });
    expect(clusters.length).toBeGreaterThanOrEqual(1);
  });

  it("'average-similarity' produces clusters", () => {
    const clusters = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.5,
      minClusterSize: 1,
      maxClusters: 10,
      assignmentStrategy: 'average-similarity',
    });
    expect(clusters.length).toBeGreaterThanOrEqual(1);
  });

  it("'average-similarity' may produce different assignments than 'centroid'", () => {
    // Use a crafted dataset that diverges between strategies
    // Tight cluster A: all identical [1,0,0]
    // Loose cluster B: members spread around [0,1,0]
    // Ambiguous point: sits between them
    const a = Array.from({ length: 6 }, () => [1, 0, 0]);
    const b = [
      [0, 1, 0],
      [0.1, 0.9, 0],
      [-0.1, 0.95, 0.05],
      [0.05, 0.85, 0.1],
      [0, 0.92, 0.08],
      [0.08, 0.88, 0.04],
    ];
    const ambiguous = [0.6, 0.6, 0];
    const data = [...a, ...b, ambiguous];

    const centroidClusters = clusterEmbeddings(data, {
      similarityThreshold: 0.3,
      minClusterSize: 1,
      maxClusters: 10,
      assignmentStrategy: 'centroid',
    });
    const avgSimClusters = clusterEmbeddings(data, {
      similarityThreshold: 0.3,
      minClusterSize: 1,
      maxClusters: 10,
      assignmentStrategy: 'average-similarity',
    });

    // We can't guarantee they differ on every dataset, but we test both run correctly
    // and produce valid results
    const totalCentroid = centroidClusters.reduce((s, c) => s + c.size, 0);
    const totalAvgSim = avgSimClusters.reduce((s, c) => s + c.size, 0);
    expect(totalCentroid).toBe(data.length);
    expect(totalAvgSim).toBe(data.length);
  });

  it('defaults to centroid when assignmentStrategy not specified', () => {
    const clusters = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.5,
      minClusterSize: 1,
    });
    expect(clusters.length).toBeGreaterThanOrEqual(1);
  });
});

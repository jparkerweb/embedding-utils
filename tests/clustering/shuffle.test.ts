import { describe, it, expect } from 'vitest';
import { clusterEmbeddings } from '../../src/clustering/cluster';

describe('shuffle option', () => {
  // Dataset where input order matters for greedy clustering
  const embeddings = [
    [1, 0, 0],
    [0.95, 0.05, 0],
    [0.9, 0.1, 0],
    [0, 1, 0],
    [0.05, 0.95, 0],
    [0.1, 0.9, 0],
    [0, 0, 1],
    [0.05, 0.05, 0.9],
    [0.1, 0.1, 0.8],
  ];

  it('shuffle: true with same seed produces identical results on repeated runs', () => {
    const run1 = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.8,
      minClusterSize: 1,
      shuffle: true,
      shuffleSeed: 123,
    });
    const run2 = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.8,
      minClusterSize: 1,
      shuffle: true,
      shuffleSeed: 123,
    });
    expect(run1.length).toBe(run2.length);
    for (let i = 0; i < run1.length; i++) {
      expect(run1[i].size).toBe(run2[i].size);
      expect(run1[i].centroid).toEqual(run2[i].centroid);
    }
  });

  it('shuffle: true with different seeds may produce different results', () => {
    const seed123 = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.8,
      minClusterSize: 1,
      shuffle: true,
      shuffleSeed: 123,
    });
    const seed456 = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.8,
      minClusterSize: 1,
      shuffle: true,
      shuffleSeed: 456,
    });
    // Both should produce valid clustering with all members accounted for
    const total123 = seed123.reduce((s, c) => s + c.size, 0);
    const total456 = seed456.reduce((s, c) => s + c.size, 0);
    expect(total123).toBe(embeddings.length);
    expect(total456).toBe(embeddings.length);
  });

  it('shuffle: false (default) preserves original order', () => {
    const noShuffle1 = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.8,
      minClusterSize: 1,
      shuffle: false,
    });
    const noShuffle2 = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.8,
      minClusterSize: 1,
    });
    // Should be identical without shuffle
    expect(noShuffle1.length).toBe(noShuffle2.length);
    for (let i = 0; i < noShuffle1.length; i++) {
      expect(noShuffle1[i].size).toBe(noShuffle2[i].size);
    }
  });

  it('shuffle: true defaults to seed 42 when shuffleSeed not provided', () => {
    const withDefault = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.8,
      minClusterSize: 1,
      shuffle: true,
    });
    const withExplicit42 = clusterEmbeddings(embeddings, {
      similarityThreshold: 0.8,
      minClusterSize: 1,
      shuffle: true,
      shuffleSeed: 42,
    });
    expect(withDefault.length).toBe(withExplicit42.length);
    for (let i = 0; i < withDefault.length; i++) {
      expect(withDefault[i].centroid).toEqual(withExplicit42[i].centroid);
    }
  });
});

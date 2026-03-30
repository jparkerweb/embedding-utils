import { describe, it, expect } from 'vitest';
import { CLUSTERING_PRESETS, getPreset } from '../../src/clustering/presets';
import { clusterEmbeddings } from '../../src/clustering/cluster';

describe('CLUSTERING_PRESETS', () => {
  it('HIGH_PRECISION has threshold >= 0.95', () => {
    expect(CLUSTERING_PRESETS.HIGH_PRECISION.similarityThreshold).toBeGreaterThanOrEqual(0.95);
  });

  it('BALANCED has threshold between 0.85 and 0.9', () => {
    const t = CLUSTERING_PRESETS.BALANCED.similarityThreshold!;
    expect(t).toBeGreaterThanOrEqual(0.85);
    expect(t).toBeLessThanOrEqual(0.9);
  });

  it('PERFORMANCE has threshold between 0.7 and 0.8', () => {
    const t = CLUSTERING_PRESETS.PERFORMANCE.similarityThreshold!;
    expect(t).toBeGreaterThanOrEqual(0.7);
    expect(t).toBeLessThanOrEqual(0.8);
  });

  it('each preset returns a valid ClusteringConfig', () => {
    for (const preset of Object.values(CLUSTERING_PRESETS)) {
      expect(preset).toHaveProperty('similarityThreshold');
      expect(preset).toHaveProperty('minClusterSize');
      expect(preset).toHaveProperty('maxClusters');
      expect(typeof preset.similarityThreshold).toBe('number');
      expect(typeof preset.minClusterSize).toBe('number');
      expect(typeof preset.maxClusters).toBe('number');
    }
  });
});

describe('getPreset', () => {
  it('returns HIGH_PRECISION preset', () => {
    expect(getPreset('high-precision')).toEqual(CLUSTERING_PRESETS.HIGH_PRECISION);
  });

  it('returns BALANCED preset', () => {
    expect(getPreset('balanced')).toEqual(CLUSTERING_PRESETS.BALANCED);
  });

  it('returns PERFORMANCE preset', () => {
    expect(getPreset('performance')).toEqual(CLUSTERING_PRESETS.PERFORMANCE);
  });

  it('presets work with clusterEmbeddings', () => {
    const embeddings = [
      [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
      [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
    ];
    const preset = getPreset('balanced');
    const clusters = clusterEmbeddings(embeddings, preset);
    expect(clusters.length).toBeGreaterThan(0);
  });
});

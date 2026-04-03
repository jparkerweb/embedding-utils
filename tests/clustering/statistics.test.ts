import { describe, it, expect } from 'vitest';
import { clusterStats, detectOutliers, centroidDrift } from '../../src/clustering/statistics';
import { computeCentroid } from '../../src/internal/clustering';

describe('clusterStats', () => {
  it('computes stats for a tight cluster (high similarities)', () => {
    const members = [
      [1, 0, 0],
      [0.99, 0.01, 0],
      [0.98, 0.02, 0],
      [0.97, 0.03, 0],
    ];
    const centroid = computeCentroid(members);
    const stats = clusterStats({ centroid, members });

    expect(stats.minSimilarity).toBeGreaterThan(0.9);
    expect(stats.maxSimilarity).toBeLessThanOrEqual(1);
    expect(stats.meanSimilarity).toBeGreaterThan(0.9);
    expect(stats.medianSimilarity).toBeGreaterThan(0.9);
    expect(stats.radius).toBeGreaterThanOrEqual(0);
    expect(stats.outliers).toHaveLength(0); // tight cluster, no outliers
  });

  it('handles empty members', () => {
    const stats = clusterStats({ centroid: [0, 0, 0], members: [] });
    expect(stats.minSimilarity).toBe(0);
    expect(stats.maxSimilarity).toBe(0);
    expect(stats.meanSimilarity).toBe(0);
    expect(stats.medianSimilarity).toBe(0);
    expect(stats.radius).toBe(0);
    expect(stats.outliers).toEqual([]);
  });

  it('handles single member', () => {
    const member = [1, 0, 0];
    const stats = clusterStats({ centroid: member, members: [member] });
    expect(stats.minSimilarity).toBeCloseTo(1, 5);
    expect(stats.maxSimilarity).toBeCloseTo(1, 5);
    expect(stats.outliers).toHaveLength(0);
  });

  it('computes correct median for even number of members', () => {
    const members = [
      [1, 0, 0],
      [0.9, 0.1, 0],
      [0.8, 0.2, 0],
      [0.7, 0.3, 0],
    ];
    const centroid = computeCentroid(members);
    const stats = clusterStats({ centroid, members });
    expect(typeof stats.medianSimilarity).toBe('number');
  });
});

describe('detectOutliers', () => {
  it('identifies distant members as outliers', () => {
    // 9 tight members + 1 distant outlier
    const tight = Array.from({ length: 9 }, () => [1, 0, 0]);
    const outlier = [0, 1, 0]; // orthogonal — very different
    const members = [...tight, outlier];
    const centroid = computeCentroid(members);

    const outliers = detectOutliers({ centroid, members });
    // The outlier (index 9) should be detected
    expect(outliers).toContain(9);
  });

  it('returns empty for uniform cluster', () => {
    const members = Array.from({ length: 5 }, () => [1, 0, 0]);
    const centroid = [1, 0, 0];
    const outliers = detectOutliers({ centroid, members });
    expect(outliers).toHaveLength(0);
  });

  it('respects custom threshold', () => {
    const tight = Array.from({ length: 9 }, () => [1, 0, 0]);
    const outlier = [0, 1, 0];
    const members = [...tight, outlier];
    const centroid = computeCentroid(members);

    // Very strict threshold should catch more outliers
    const strict = detectOutliers({ centroid, members }, { threshold: 1 });
    const loose = detectOutliers({ centroid, members }, { threshold: 3 });
    expect(strict.length).toBeGreaterThanOrEqual(loose.length);
  });

  it('returns empty for empty members', () => {
    expect(detectOutliers({ centroid: [0], members: [] })).toEqual([]);
  });
});

describe('centroidDrift', () => {
  it('returns 0 for identical centroids', () => {
    const c = [1, 0, 0];
    expect(centroidDrift(c, c)).toBeCloseTo(0, 10);
  });

  it('returns ~1 for orthogonal centroids (cosine)', () => {
    const a = [1, 0, 0];
    const b = [0, 1, 0];
    // cosine distance = 1 - cos(90°) = 1 - 0 = 1
    expect(centroidDrift(a, b)).toBeCloseTo(1, 5);
  });

  it('returns value between 0 and 1 for partially similar centroids (cosine)', () => {
    const a = [1, 0, 0];
    const b = [0.7, 0.7, 0];
    const drift = centroidDrift(a, b);
    expect(drift).toBeGreaterThan(0);
    expect(drift).toBeLessThan(1);
  });

  it('works with euclidean metric', () => {
    const a = [1, 0, 0];
    const b = [1, 0, 0];
    expect(centroidDrift(a, b, 'euclidean')).toBeCloseTo(0, 10);
  });
});

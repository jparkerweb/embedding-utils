import { describe, it, expect } from 'vitest';
import { findOptimalK, silhouetteByK } from '../../src/clustering/optimal';

describe('findOptimalK', () => {
  // Three well-separated clusters in 3D
  const clusterA = Array.from({ length: 10 }, (_, i) => [1 + i * 0.01, 0, 0]);
  const clusterB = Array.from({ length: 10 }, (_, i) => [0, 1 + i * 0.01, 0]);
  const clusterC = Array.from({ length: 10 }, (_, i) => [0, 0, 1 + i * 0.01]);
  const wellSeparated = [...clusterA, ...clusterB, ...clusterC];

  it('returns a reasonable k for well-separated clusters (silhouette)', () => {
    const k = findOptimalK(wellSeparated, {
      minK: 2,
      maxK: 5,
      method: 'silhouette',
    });
    expect(k).toBeGreaterThanOrEqual(2);
    expect(k).toBeLessThanOrEqual(5);
  });

  it('returns a reasonable k for well-separated clusters (elbow)', () => {
    const k = findOptimalK(wellSeparated, {
      minK: 2,
      maxK: 5,
      method: 'elbow',
    });
    expect(k).toBeGreaterThanOrEqual(2);
    expect(k).toBeLessThanOrEqual(5);
  });

  it('defaults to silhouette method', () => {
    const k = findOptimalK(wellSeparated, { minK: 2, maxK: 4 });
    expect(k).toBeGreaterThanOrEqual(2);
    expect(k).toBeLessThanOrEqual(4);
  });

  it('returns minK when maxK < minK', () => {
    const k = findOptimalK(wellSeparated, { minK: 5, maxK: 3 });
    expect(k).toBe(5);
  });

  it('uses sensible defaults for minK and maxK', () => {
    const k = findOptimalK(wellSeparated);
    expect(k).toBeGreaterThanOrEqual(2);
  });
});

describe('silhouetteByK', () => {
  const clusterA = Array.from({ length: 8 }, (_, i) => [1 + i * 0.01, 0, 0]);
  const clusterB = Array.from({ length: 8 }, (_, i) => [0, 1 + i * 0.01, 0]);
  const data = [...clusterA, ...clusterB];

  it('returns array of correct length', () => {
    const results = silhouetteByK(data, { minK: 2, maxK: 4 });
    expect(results).toHaveLength(3); // k=2,3,4
  });

  it('each entry has k and silhouette fields', () => {
    const results = silhouetteByK(data, { minK: 2, maxK: 3 });
    for (const entry of results) {
      expect(entry).toHaveProperty('k');
      expect(entry).toHaveProperty('silhouette');
      expect(typeof entry.k).toBe('number');
      expect(typeof entry.silhouette).toBe('number');
    }
  });

  it('returns entries sorted by k ascending', () => {
    const results = silhouetteByK(data, { minK: 2, maxK: 5 });
    for (let i = 1; i < results.length; i++) {
      expect(results[i].k).toBeGreaterThan(results[i - 1].k);
    }
  });
});

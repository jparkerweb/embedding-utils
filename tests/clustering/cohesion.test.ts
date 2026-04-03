import { describe, it, expect } from 'vitest';
import { centroidCohesion } from '../../src/clustering/metrics';
import type { Cluster, SimilarityMetric } from '../../src/types';

function makeCluster(members: number[][], centroid?: number[]): Cluster {
  const c = centroid ?? members.reduce(
    (acc, m) => acc.map((v, i) => v + m[i] / members.length),
    new Array(members[0].length).fill(0),
  );
  return { centroid: c, members, size: members.length, cohesion: 0 };
}

describe('centroidCohesion', () => {
  it('returns 1.0 for a single-member cluster', () => {
    const cluster = makeCluster([[1, 0, 0]]);
    expect(centroidCohesion(cluster, 'cosine')).toBe(1.0);
  });

  it('returns high cohesion for tight cluster (cosine)', () => {
    const members = [
      [1, 0, 0],
      [0.99, 0.01, 0],
      [0.98, 0.02, 0],
    ];
    const cluster = makeCluster(members);
    expect(centroidCohesion(cluster, 'cosine')).toBeGreaterThan(0.99);
  });

  it('returns low cohesion for loose cluster (cosine)', () => {
    const members = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    const cluster = makeCluster(members);
    // Members point in orthogonal directions; cosine to centroid is moderate
    expect(centroidCohesion(cluster, 'cosine')).toBeLessThan(0.7);
  });

  it('works with euclidean metric', () => {
    const members = [[1, 0], [1.1, 0.1]];
    const cluster = makeCluster(members);
    const score = centroidCohesion(cluster, 'euclidean');
    expect(typeof score).toBe('number');
    expect(Number.isFinite(score)).toBe(true);
  });

  it('works with dot metric', () => {
    const members = [[1, 0], [0.9, 0.1]];
    const cluster = makeCluster(members);
    const score = centroidCohesion(cluster, 'dot');
    expect(typeof score).toBe('number');
    expect(Number.isFinite(score)).toBe(true);
  });

  it('works with manhattan metric', () => {
    const members = [[1, 0], [0.9, 0.1]];
    const cluster = makeCluster(members);
    const score = centroidCohesion(cluster, 'manhattan');
    expect(typeof score).toBe('number');
    expect(Number.isFinite(score)).toBe(true);
  });

  it('returns 1.0 for all identical members (cosine)', () => {
    const members = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]];
    const cluster = makeCluster(members);
    expect(centroidCohesion(cluster, 'cosine')).toBeCloseTo(1.0, 5);
  });
});

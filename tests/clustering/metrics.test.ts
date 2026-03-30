import { describe, it, expect } from 'vitest';
import { cohesionScore, silhouetteScore } from '../../src/clustering/metrics';
import type { Cluster } from '../../src/types';

function makeCluster(members: number[][], cohesion = 0): Cluster {
  const dims = members[0].length;
  const centroid = new Array<number>(dims).fill(0);
  for (const m of members) {
    for (let i = 0; i < dims; i++) centroid[i] += m[i];
  }
  for (let i = 0; i < dims; i++) centroid[i] /= members.length;
  return { centroid, members, size: members.length, cohesion };
}

describe('cohesionScore', () => {
  it('returns 1.0 for a single-member cluster', () => {
    const cluster = makeCluster([[1, 0, 0]]);
    expect(cohesionScore(cluster)).toBe(1.0);
  });

  it('returns 1.0 for identical members', () => {
    const cluster = makeCluster([
      [1, 0, 0],
      [1, 0, 0],
      [1, 0, 0],
    ]);
    expect(cohesionScore(cluster)).toBeCloseTo(1.0, 5);
  });

  it('returns known numeric value for orthogonal vectors', () => {
    const cluster = makeCluster([
      [1, 0, 0],
      [0, 1, 0],
    ]);
    // cosine similarity of orthogonal vectors = 0
    expect(cohesionScore(cluster)).toBeCloseTo(0, 5);
  });

  it('returns intermediate value for partially similar vectors', () => {
    const cluster = makeCluster([
      [1, 0, 0],
      [0.7, 0.7, 0],
    ]);
    const score = cohesionScore(cluster);
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1);
  });

  it('works with euclidean metric', () => {
    const cluster = makeCluster([
      [1, 0, 0],
      [1, 0, 0],
    ]);
    expect(cohesionScore(cluster, 'euclidean')).toBeCloseTo(1.0, 5);
  });

  it('works with manhattan metric', () => {
    const cluster = makeCluster([
      [1, 0, 0],
      [0, 1, 0],
    ]);
    const score = cohesionScore(cluster, 'manhattan');
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1);
  });

  it('works with dot metric', () => {
    const cluster = makeCluster([
      [1, 0, 0],
      [1, 0, 0],
    ]);
    expect(cohesionScore(cluster, 'dot')).toBeCloseTo(1.0, 5);
  });
});

describe('silhouetteScore', () => {
  it('returns 0 for a single cluster', () => {
    const clusters = [
      makeCluster([
        [1, 0, 0],
        [1, 0, 0],
      ]),
    ];
    expect(silhouetteScore(clusters)).toBe(0);
  });

  it('returns score near 1.0 for well-separated clusters', () => {
    const clusters = [
      makeCluster([
        [1, 0, 0],
        [0.99, 0.01, 0],
        [0.98, 0.02, 0],
      ]),
      makeCluster([
        [0, 1, 0],
        [0.01, 0.99, 0],
        [0.02, 0.98, 0],
      ]),
    ];
    const score = silhouetteScore(clusters);
    expect(score).toBeGreaterThan(0.5);
  });

  it('returns lower score for overlapping clusters', () => {
    const wellSeparated = [
      makeCluster([
        [1, 0, 0],
        [0.99, 0.01, 0],
      ]),
      makeCluster([
        [0, 1, 0],
        [0.01, 0.99, 0],
      ]),
    ];
    const overlapping = [
      makeCluster([
        [0.6, 0.4, 0],
        [0.55, 0.45, 0],
      ]),
      makeCluster([
        [0.4, 0.6, 0],
        [0.45, 0.55, 0],
      ]),
    ];
    expect(silhouetteScore(overlapping)).toBeLessThan(
      silhouetteScore(wellSeparated),
    );
  });

  it('works with euclidean metric', () => {
    const clusters = [
      makeCluster([
        [1, 0, 0],
        [0.99, 0.01, 0],
      ]),
      makeCluster([
        [0, 1, 0],
        [0.01, 0.99, 0],
      ]),
    ];
    const score = silhouetteScore(clusters, 'euclidean');
    expect(score).toBeGreaterThan(0);
  });

  it('works with manhattan metric', () => {
    const clusters = [
      makeCluster([
        [1, 0, 0],
        [0.99, 0.01, 0],
      ]),
      makeCluster([
        [0, 1, 0],
        [0.01, 0.99, 0],
      ]),
    ];
    const score = silhouetteScore(clusters, 'manhattan');
    expect(score).toBeGreaterThan(0);
  });

  it('validates against known dataset', () => {
    // 3 well-defined clusters along axes
    const clusters = [
      makeCluster([
        [1, 0, 0],
        [0.95, 0.05, 0],
      ]),
      makeCluster([
        [0, 1, 0],
        [0.05, 0.95, 0],
      ]),
      makeCluster([
        [0, 0, 1],
        [0, 0.05, 0.95],
      ]),
    ];
    const score = silhouetteScore(clusters);
    // Well-separated clusters should give a positive silhouette score
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThanOrEqual(1);
  });
});

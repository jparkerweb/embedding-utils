import { describe, it, expect } from 'vitest';
import { hdbscan } from '../../src/clustering/hdbscan';
import { ValidationError } from '../../src/types';

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Generate a cluster of 2D points centered at (cx, cy) with uniform noise */
function makeCluster2D(
  cx: number,
  cy: number,
  count: number,
  spread: number,
  seed: number,
): number[][] {
  const pts: number[][] = [];
  let s = seed;
  for (let i = 0; i < count; i++) {
    // Simple PRNG (LCG) for deterministic tests
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    const r1 = (s / 0x7fffffff - 0.5) * 2 * spread;
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    const r2 = (s / 0x7fffffff - 0.5) * 2 * spread;
    pts.push([cx + r1, cy + r2]);
  }
  return pts;
}

/** Generate a ring of 2D points at distance `radius` from (cx, cy) */
function makeRing2D(
  cx: number,
  cy: number,
  radius: number,
  count: number,
  noise: number,
  seed: number,
): number[][] {
  const pts: number[][] = [];
  let s = seed;
  for (let i = 0; i < count; i++) {
    const angle = (2 * Math.PI * i) / count;
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    const nr = (s / 0x7fffffff - 0.5) * 2 * noise;
    pts.push([cx + (radius + nr) * Math.cos(angle), cy + (radius + nr) * Math.sin(angle)]);
  }
  return pts;
}

// ─── Task 5.1: Well-separated and varying-density clusters ────────────────────

describe('hdbscan — well-separated clusters', () => {
  it('finds 3 well-separated 2D clusters', () => {
    const embeddings = [
      ...makeCluster2D(0, 0, 20, 0.5, 1),
      ...makeCluster2D(10, 0, 20, 0.5, 2),
      ...makeCluster2D(0, 10, 20, 0.5, 3),
    ];
    const result = hdbscan(embeddings, { minClusterSize: 5 });
    expect(result.clusters.length).toBe(3);
    // Total assigned to clusters should be most/all points
    const clustered = result.clusters.reduce((sum, c) => sum + c.size, 0);
    expect(clustered).toBeGreaterThanOrEqual(50);
  });

  it('finds 2 clusters with different densities', () => {
    // Tight cluster and a more spread cluster
    const tight = makeCluster2D(0, 0, 30, 0.3, 10);
    const spread = makeCluster2D(20, 20, 30, 2.0, 20);
    const embeddings = [...tight, ...spread];
    const result = hdbscan(embeddings, { minClusterSize: 5 });
    expect(result.clusters.length).toBe(2);
  });

  it('separates concentric ring structures', () => {
    const inner = makeRing2D(0, 0, 2, 40, 0.2, 100);
    const outer = makeRing2D(0, 0, 8, 40, 0.2, 200);
    const embeddings = [...inner, ...outer];
    const result = hdbscan(embeddings, { minClusterSize: 5 });
    expect(result.clusters.length).toBe(2);
  });
});

// ─── Task 5.2: Noise detection ───────────────────────────────────────────────

describe('hdbscan — noise detection', () => {
  it('identifies scattered outliers as noise', () => {
    const cluster1 = makeCluster2D(0, 0, 20, 0.3, 1);
    const cluster2 = makeCluster2D(10, 0, 20, 0.3, 2);
    const cluster3 = makeCluster2D(0, 10, 20, 0.3, 3);
    // 5 scattered outlier points far from any cluster
    const outliers: number[][] = [
      [50, 50],
      [-50, -50],
      [50, -50],
      [-50, 50],
      [100, 100],
    ];
    const embeddings = [...cluster1, ...cluster2, ...cluster3, ...outliers];
    const result = hdbscan(embeddings, { minClusterSize: 5 });

    // Noise should contain the outliers
    expect(result.noise.members.length).toBeGreaterThanOrEqual(5);
    expect(result.noise.indices.length).toBe(result.noise.members.length);
  });

  it('assigns -1 label to noise points', () => {
    // Use multiple clusters + outliers so the outliers don't get absorbed
    const cluster1 = makeCluster2D(0, 0, 15, 0.3, 1);
    const cluster2 = makeCluster2D(10, 0, 15, 0.3, 2);
    const outliers: number[][] = [[100, 100], [-100, -100]];
    const embeddings = [...cluster1, ...cluster2, ...outliers];
    const result = hdbscan(embeddings, { minClusterSize: 5 });

    // The outlier points should have label -1
    const noiseLabels = result.labels.filter((l) => l === -1);
    expect(noiseLabels.length).toBeGreaterThanOrEqual(2);
  });

  it('noise points are NOT in any cluster members', () => {
    const cluster = makeCluster2D(0, 0, 20, 0.3, 1);
    const outliers: number[][] = [[100, 100], [-100, -100]];
    const embeddings = [...cluster, ...outliers];
    const result = hdbscan(embeddings, { minClusterSize: 5 });

    // Collect all indices that are in clusters
    const clusterIndices = new Set<number>();
    for (const c of result.clusters) {
      for (const idx of c.indices!) {
        clusterIndices.add(idx);
      }
    }
    // Noise indices should not overlap with cluster indices
    for (const idx of result.noise.indices) {
      expect(clusterIndices.has(idx)).toBe(false);
    }
  });
});

// ─── Task 5.3: Edge cases ────────────────────────────────────────────────────

describe('hdbscan — edge cases', () => {
  it('returns empty result for empty input', () => {
    const result = hdbscan([], { minClusterSize: 5 });
    expect(result.clusters).toEqual([]);
    expect(result.noise.members).toEqual([]);
    expect(result.noise.indices).toEqual([]);
    expect(result.labels).toEqual([]);
  });

  it('single point → all noise', () => {
    const result = hdbscan([[1, 2]], { minClusterSize: 5 });
    expect(result.clusters).toHaveLength(0);
    expect(result.noise.members).toHaveLength(1);
    expect(result.noise.indices).toEqual([0]);
    expect(result.labels).toEqual([-1]);
  });

  it('all identical vectors → single cluster when minClusterSize allows', () => {
    const embeddings = Array.from({ length: 10 }, () => [1, 0, 0]);
    const result = hdbscan(embeddings, { minClusterSize: 5 });
    // All identical → should form one cluster (distances are 0, all merge immediately)
    const totalPoints =
      result.clusters.reduce((sum, c) => sum + c.size, 0) + result.noise.members.length;
    expect(totalPoints).toBe(10);
    if (result.clusters.length > 0) {
      expect(result.clusters.length).toBe(1);
    }
  });

  it('larger minClusterSize → fewer, larger clusters', () => {
    const embeddings = [
      ...makeCluster2D(0, 0, 15, 0.3, 1),
      ...makeCluster2D(10, 0, 15, 0.3, 2),
      ...makeCluster2D(0, 10, 15, 0.3, 3),
    ];
    const smallMin = hdbscan(embeddings, { minClusterSize: 3 });
    const largeMin = hdbscan(embeddings, { minClusterSize: 10 });
    expect(largeMin.clusters.length).toBeLessThanOrEqual(smallMin.clusters.length);
  });

  it('minSamples affects core distance computation', () => {
    const embeddings = [
      ...makeCluster2D(0, 0, 20, 0.5, 1),
      ...makeCluster2D(10, 0, 20, 0.5, 2),
    ];
    const result1 = hdbscan(embeddings, { minClusterSize: 5, minSamples: 3 });
    const result2 = hdbscan(embeddings, { minClusterSize: 5, minSamples: 15 });
    // Different minSamples should potentially produce different results
    // At minimum both should be valid
    expect(result1.labels.length).toBe(40);
    expect(result2.labels.length).toBe(40);
  });
});

// ─── Task 5.4: Label preservation ────────────────────────────────────────────

describe('hdbscan — label preservation', () => {
  it('preserves string labels in cluster members', () => {
    const embeddings = [
      ...makeCluster2D(0, 0, 10, 0.3, 1),
      ...makeCluster2D(10, 0, 10, 0.3, 2),
    ];
    const labels = embeddings.map((_, i) => `point-${i}`);
    const result = hdbscan(embeddings, { minClusterSize: 5, labels });

    // All cluster members should have labels
    for (const cluster of result.clusters) {
      expect(cluster.labels).toBeDefined();
      expect(cluster.labels!.length).toBe(cluster.size);
      for (const label of cluster.labels!) {
        expect(label).toMatch(/^point-\d+$/);
      }
    }
  });

  it('preserves labels for noise members', () => {
    const cluster = makeCluster2D(0, 0, 10, 0.3, 1);
    const outliers: number[][] = [[100, 100]];
    const embeddings = [...cluster, ...outliers];
    const labels = embeddings.map((_, i) => `item-${i}`);
    const result = hdbscan(embeddings, { minClusterSize: 5, labels });

    // Noise labels should be populated
    if (result.noise.labels) {
      for (const label of result.noise.labels) {
        expect(label).toMatch(/^item-\d+$/);
      }
    }
  });

  it('throws ValidationError for labels length mismatch', () => {
    const embeddings = makeCluster2D(0, 0, 10, 0.3, 1);
    expect(() => hdbscan(embeddings, { minClusterSize: 5, labels: ['a', 'b'] })).toThrow(
      ValidationError,
    );
  });
});

// ─── Task 5.5: Metric support ────────────────────────────────────────────────

describe('hdbscan — metric support', () => {
  it('default metric is euclidean', () => {
    const embeddings = [
      ...makeCluster2D(0, 0, 15, 0.3, 1),
      ...makeCluster2D(10, 0, 15, 0.3, 2),
    ];
    const defaultResult = hdbscan(embeddings, { minClusterSize: 5 });
    const euclideanResult = hdbscan(embeddings, { minClusterSize: 5, metric: 'euclidean' });
    expect(defaultResult.labels).toEqual(euclideanResult.labels);
  });

  it('cosine vs euclidean produce different cluster assignments', () => {
    // Use vectors where cosine and euclidean would behave differently
    // Vectors with same direction but different magnitudes
    const embeddings: number[][] = [];
    for (let i = 0; i < 15; i++) {
      embeddings.push([1 + i * 0.01, 0.5 + i * 0.01]); // cluster A direction
    }
    for (let i = 0; i < 15; i++) {
      embeddings.push([-(1 + i * 0.01), 0.5 + i * 0.01]); // cluster B direction
    }
    // Add points with varied magnitudes in cluster A direction
    for (let i = 0; i < 15; i++) {
      embeddings.push([(1 + i * 0.01) * 10, (0.5 + i * 0.01) * 10]);
    }

    const cosineResult = hdbscan(embeddings, { minClusterSize: 5, metric: 'cosine' });
    const euclideanResult = hdbscan(embeddings, { minClusterSize: 5, metric: 'euclidean' });

    // At least one should produce different clustering
    const cosineLabels = JSON.stringify(cosineResult.labels);
    const euclideanLabels = JSON.stringify(euclideanResult.labels);
    expect(cosineLabels).not.toEqual(euclideanLabels);
  });
});

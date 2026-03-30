import { describe, it, expect } from 'vitest';
import { assignToCluster, mergeClusters } from '../../src/clustering/operations';
import type { Cluster } from '../../src/types';

function makeCluster(
  members: number[][],
  labels?: string[],
): Cluster {
  const dims = members[0].length;
  const centroid = new Array<number>(dims).fill(0);
  for (const m of members) {
    for (let i = 0; i < dims; i++) centroid[i] += m[i];
  }
  for (let i = 0; i < dims; i++) centroid[i] /= members.length;
  return { centroid, members, labels, size: members.length, cohesion: 0 };
}

describe('assignToCluster', () => {
  const clusters: Cluster[] = [
    makeCluster([[1, 0, 0], [0.95, 0.05, 0]]),
    makeCluster([[0, 1, 0], [0.05, 0.95, 0]]),
  ];

  it('assigns embedding to most similar cluster', () => {
    const result = assignToCluster([0.9, 0.1, 0], clusters);
    expect(result.clusterIndex).toBe(0);
    expect(result.similarity).toBeGreaterThan(0.9);
  });

  it('assigns to second cluster when more similar', () => {
    const result = assignToCluster([0.1, 0.9, 0], clusters);
    expect(result.clusterIndex).toBe(1);
    expect(result.similarity).toBeGreaterThan(0.9);
  });

  it('returns similarity score', () => {
    const result = assignToCluster([1, 0, 0], clusters);
    expect(typeof result.similarity).toBe('number');
    expect(result.similarity).toBeGreaterThan(0);
  });

  it('returns -1 index if below threshold', () => {
    const result = assignToCluster([0, 0, 1], clusters, { threshold: 0.9 });
    expect(result.clusterIndex).toBe(-1);
    expect(typeof result.similarity).toBe('number');
  });

  it('defaults to threshold 0 (accepts any match)', () => {
    const result = assignToCluster([0.5, 0.5, 0.7], clusters);
    // With threshold 0, should always find a match
    expect(result.clusterIndex).toBeGreaterThanOrEqual(0);
  });
});

describe('mergeClusters', () => {
  it('combines members from both clusters', () => {
    const a = makeCluster([[1, 0, 0], [0.9, 0.1, 0]]);
    const b = makeCluster([[0.8, 0.2, 0]]);
    const merged = mergeClusters(a, b);
    expect(merged.members).toHaveLength(3);
    expect(merged.size).toBe(3);
  });

  it('recomputes centroid from combined members', () => {
    const a = makeCluster([[1, 0, 0]]);
    const b = makeCluster([[0, 1, 0]]);
    const merged = mergeClusters(a, b);
    expect(merged.centroid[0]).toBeCloseTo(0.5, 5);
    expect(merged.centroid[1]).toBeCloseTo(0.5, 5);
    expect(merged.centroid[2]).toBeCloseTo(0, 5);
  });

  it('updates cohesion', () => {
    const a = makeCluster([[1, 0, 0], [0.99, 0.01, 0]]);
    const b = makeCluster([[0, 1, 0], [0.01, 0.99, 0]]);
    const merged = mergeClusters(a, b);
    expect(typeof merged.cohesion).toBe('number');
    // Merged disparate clusters should have lower cohesion
    expect(merged.cohesion).toBeLessThan(1);
  });

  it('concatenates labels if present', () => {
    const a = makeCluster([[1, 0, 0]], ['a1']);
    const b = makeCluster([[0, 1, 0]], ['b1']);
    const merged = mergeClusters(a, b);
    expect(merged.labels).toEqual(['a1', 'b1']);
  });

  it('handles one cluster with labels and one without', () => {
    const a = makeCluster([[1, 0, 0]], ['a1']);
    const b = makeCluster([[0, 1, 0]]);
    const merged = mergeClusters(a, b);
    expect(merged.labels).toEqual(['a1']);
  });

  it('omits labels when neither cluster has them', () => {
    const a = makeCluster([[1, 0, 0]]);
    const b = makeCluster([[0, 1, 0]]);
    const merged = mergeClusters(a, b);
    expect(merged.labels).toBeUndefined();
  });
});

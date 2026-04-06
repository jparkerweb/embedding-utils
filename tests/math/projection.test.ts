import { describe, it, expect } from 'vitest';
import { createRandomProjection } from '../../src/math/projection';
import { ValidationError } from '../../src/types';

function euclideanDist(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function randomVector(dims: number, rng: () => number): Float32Array {
  const v = new Float32Array(dims);
  for (let i = 0; i < dims; i++) v[i] = rng() * 2 - 1;
  return v;
}

// Simple seeded RNG for test data generation
function simpleRng(seed: number): () => number {
  let state = seed | 0 || 1;
  return () => {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
}

describe('createRandomProjection', () => {
  it('preserves pairwise distances within JL bounds', () => {
    const sourceDims = 512;
    const targetDims = 128;
    const n = 100;
    const rng = simpleRng(12345);

    const vectors = Array.from({ length: n }, () => randomVector(sourceDims, rng));
    const projector = createRandomProjection(sourceDims, targetDims, { seed: 42 });
    const projected = projector.projectBatch(vectors);

    // JL bound: epsilon ≈ sqrt(8 * ln(n) / targetDims)
    const epsilon = Math.sqrt((8 * Math.log(n)) / targetDims);

    // Check a sample of pairwise distances
    let withinBounds = 0;
    let totalPairs = 0;
    for (let i = 0; i < 50; i++) {
      for (let j = i + 1; j < 50; j++) {
        const origDist = euclideanDist(vectors[i], vectors[j]);
        const projDist = euclideanDist(projected[i], projected[j]);
        const ratio = projDist / origDist;
        if (ratio >= 1 - epsilon && ratio <= 1 + epsilon) {
          withinBounds++;
        }
        totalPairs++;
      }
    }
    // At least 80% of pairs should be within JL bounds (probabilistic guarantee)
    expect(withinBounds / totalPairs).toBeGreaterThan(0.8);
  });

  it('deterministic with same seed → identical output', () => {
    const v = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const p1 = createRandomProjection(8, 4, { seed: 99 });
    const p2 = createRandomProjection(8, 4, { seed: 99 });
    const r1 = p1.project(v);
    const r2 = p2.project(v);
    expect(Array.from(r1)).toEqual(Array.from(r2));
  });

  it('different seeds → different output', () => {
    const v = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const p1 = createRandomProjection(8, 4, { seed: 1 });
    const p2 = createRandomProjection(8, 4, { seed: 2 });
    const r1 = p1.project(v);
    const r2 = p2.project(v);
    expect(Array.from(r1)).not.toEqual(Array.from(r2));
  });

  it('projectBatch produces same results as individual project calls', () => {
    const vectors = [
      new Float32Array([1, 0, 0, 0]),
      new Float32Array([0, 1, 0, 0]),
      new Float32Array([0, 0, 1, 0]),
    ];
    const projector = createRandomProjection(4, 2, { seed: 42 });
    const batch = projector.projectBatch(vectors);
    const individual = vectors.map((v) => projector.project(v));
    for (let i = 0; i < vectors.length; i++) {
      expect(Array.from(batch[i])).toEqual(Array.from(individual[i]));
    }
  });

  it('sourceDims and targetDims properties are correct', () => {
    const projector = createRandomProjection(256, 64, { seed: 1 });
    expect(projector.sourceDims).toBe(256);
    expect(projector.targetDims).toBe(64);
  });

  it('throws ValidationError when targetDims > sourceDims', () => {
    expect(() => createRandomProjection(4, 8)).toThrow(ValidationError);
  });
});

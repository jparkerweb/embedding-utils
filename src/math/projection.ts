/**
 * Johnson-Lindenstrauss random projection for dimensionality reduction.
 *
 * Generates a Gaussian random matrix that approximately preserves pairwise distances
 * when projecting high-dimensional vectors to a lower-dimensional space.
 */

import { ValidationError } from '../types';
import type { Vector } from '../types';
import { toFloat32 } from '../internal/vector-utils';
import { createRng } from '../internal/random';

/**
 * A random projector that reduces vector dimensionality while approximately
 * preserving pairwise distances (Johnson-Lindenstrauss lemma).
 */
export interface RandomProjector {
  /** Source (input) dimensionality. */
  readonly sourceDims: number;
  /** Target (output) dimensionality. */
  readonly targetDims: number;
  /** Project a single vector. */
  project(vector: Vector): Float32Array;
  /** Project a batch of vectors. */
  projectBatch(vectors: Vector[]): Float32Array[];
}

/**
 * Box-Muller transform: generate a standard normal value from two uniform [0,1) values.
 */
function gaussianRandom(rng: () => number): number {
  let u1 = rng();
  // Avoid log(0)
  while (u1 === 0) u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Creates a random projection from sourceDims to targetDims dimensions.
 *
 * The projection matrix entries are drawn from N(0, 1/targetDims), which
 * ensures that pairwise distances are approximately preserved per the
 * Johnson-Lindenstrauss lemma.
 *
 * @param sourceDims - Number of dimensions in the input vectors
 * @param targetDims - Number of dimensions in the output vectors
 * @param options - Optional seed for deterministic projection
 * @returns A RandomProjector object
 * @throws {ValidationError} If targetDims > sourceDims or dimensions are invalid
 */
export function createRandomProjection(
  sourceDims: number,
  targetDims: number,
  options?: { seed?: number },
): RandomProjector {
  if (targetDims > sourceDims) {
    throw new ValidationError(
      `targetDims (${targetDims}) must not exceed sourceDims (${sourceDims})`,
    );
  }
  if (sourceDims <= 0 || targetDims <= 0) {
    throw new ValidationError('Dimensions must be positive integers');
  }

  const seed = options?.seed ?? 42;
  const rng = createRng(seed);
  const scale = 1 / Math.sqrt(targetDims);

  // Generate projection matrix (targetDims x sourceDims) stored row-major in a flat Float32Array
  const matrix = new Float32Array(targetDims * sourceDims);
  for (let i = 0; i < matrix.length; i++) {
    matrix[i] = gaussianRandom(rng) * scale;
  }

  function project(vector: Vector): Float32Array {
    const v = toFloat32(vector);
    const result = new Float32Array(targetDims);
    for (let row = 0; row < targetDims; row++) {
      let sum = 0;
      const offset = row * sourceDims;
      for (let col = 0; col < sourceDims; col++) {
        sum += matrix[offset + col] * v[col];
      }
      result[row] = sum;
    }
    return result;
  }

  return {
    sourceDims,
    targetDims,
    project,
    projectBatch(vectors: Vector[]): Float32Array[] {
      return vectors.map(project);
    },
  };
}

import { ValidationError } from '../types';

/**
 * Truncates embedding(s) to a target number of dimensions (Matryoshka-style).
 * @param input - A single embedding vector or array of embedding vectors
 * @param targetDims - The desired number of dimensions (must be > 0 and <= vector length)
 * @returns Truncated vector(s) matching the input shape
 * @throws {ValidationError} If targetDims is <= 0 or exceeds vector length
 * @example
 * truncateDimensions([1, 2, 3, 4], 2); // [1, 2]
 * truncateDimensions([[1, 2, 3], [4, 5, 6]], 2); // [[1, 2], [4, 5]]
 */
export function truncateDimensions(embedding: number[], targetDims: number): number[];
export function truncateDimensions(embeddings: number[][], targetDims: number): number[][];
export function truncateDimensions(
  input: number[] | number[][],
  targetDims: number
): number[] | number[][] {
  if (targetDims <= 0) {
    throw new ValidationError('targetDims must be greater than 0');
  }

  if (Array.isArray(input[0])) {
    const batch = input as number[][];
    return batch.map((v) => truncateSingle(v, targetDims));
  }

  return truncateSingle(input as number[], targetDims);
}

function truncateSingle(v: number[], targetDims: number): number[] {
  if (targetDims > v.length) {
    throw new ValidationError(
      `targetDims (${targetDims}) exceeds vector length (${v.length})`
    );
  }
  if (targetDims === v.length) return v;
  return v.slice(0, targetDims);
}

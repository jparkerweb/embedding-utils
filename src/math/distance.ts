import { ValidationError } from '../types';

function validateVectorPair(a: number[], b: number[]): void {
  if (a.length === 0 || b.length === 0) {
    throw new ValidationError('Vectors must be non-empty');
  }
  if (a.length !== b.length) {
    throw new ValidationError(`Dimension mismatch: ${a.length} vs ${b.length}`);
  }
}

/**
 * Computes the Euclidean (L2) distance between two vectors.
 * @param a - First vector
 * @param b - Second vector
 * @returns The Euclidean distance (always >= 0)
 * @throws {ValidationError} If vectors are empty or have different dimensions
 * @example
 * euclideanDistance([0, 0], [3, 4]); // 5
 */
export function euclideanDistance(a: number[], b: number[]): number {
  validateVectorPair(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/**
 * Computes the Manhattan (L1) distance between two vectors.
 * @param a - First vector
 * @param b - Second vector
 * @returns The Manhattan distance (always >= 0)
 * @throws {ValidationError} If vectors are empty or have different dimensions
 * @example
 * manhattanDistance([0, 0], [3, 4]); // 7
 */
export function manhattanDistance(a: number[], b: number[]): number {
  validateVectorPair(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum;
}

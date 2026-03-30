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
 * Computes the dot product of two vectors.
 * @param a - First vector
 * @param b - Second vector
 * @returns The dot product scalar value
 * @throws {ValidationError} If vectors are empty or have different dimensions
 * @example
 * dotProduct([1, 2, 3], [4, 5, 6]); // 32
 */
export function dotProduct(a: number[], b: number[]): number {
  validateVectorPair(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * Computes the cosine similarity between two vectors.
 * @param a - First vector
 * @param b - Second vector
 * @returns Similarity score between -1 and 1 (0 if either vector is zero)
 * @throws {ValidationError} If vectors are empty or have different dimensions
 * @example
 * cosineSimilarity([1, 0], [0, 1]); // 0
 * cosineSimilarity([1, 0], [1, 0]); // 1
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  validateVectorPair(a, b);

  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }

  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  if (denom === 0) return 0;
  return dot / denom;
}

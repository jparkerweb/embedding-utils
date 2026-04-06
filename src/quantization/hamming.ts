import { ValidationError } from '../types';

// Popcount lookup table for bytes 0-255
const POPCOUNT_TABLE = new Uint8Array(256);
for (let i = 0; i < 256; i++) {
  POPCOUNT_TABLE[i] = POPCOUNT_TABLE[i >> 1] + (i & 1);
}

/**
 * Computes the Hamming distance between two binary vectors.
 *
 * XORs each byte pair and counts the number of differing bits using
 * a lookup table for fast popcount.
 *
 * @param a - First binary vector (Uint8Array)
 * @param b - Second binary vector (Uint8Array, must be same length as `a`)
 * @returns Number of differing bits
 * @throws {ValidationError} If vectors have different lengths
 */
export function hammingDistance(a: Uint8Array, b: Uint8Array): number {
  if (a.length !== b.length) {
    throw new ValidationError(
      `Vector length mismatch: ${a.length} vs ${b.length}`,
    );
  }
  let distance = 0;
  for (let i = 0; i < a.length; i++) {
    distance += POPCOUNT_TABLE[a[i] ^ b[i]];
  }
  return distance;
}

/**
 * Computes the Hamming similarity between two binary vectors.
 *
 * Returns `1 - (hammingDistance(a, b) / dimensions)` where `dimensions`
 * is the total number of bits being compared.
 *
 * @param a - First binary vector (Uint8Array)
 * @param b - Second binary vector (Uint8Array, must be same length as `a`)
 * @param dimensions - Total number of bits (used as denominator)
 * @returns Similarity in range [0, 1]
 * @throws {ValidationError} If vectors have different lengths
 */
export function hammingSimilarity(
  a: Uint8Array,
  b: Uint8Array,
  dimensions: number,
): number {
  return 1 - hammingDistance(a, b) / dimensions;
}

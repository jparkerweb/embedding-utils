// ─────────────────────────────────────────────────────────────────────────────
// Vector Operations
//
// Element-wise operations for manipulating embedding vectors. These are the
// low-level building blocks for combining, transforming, and analyzing vectors.
// All functions return new arrays (no mutation) and validate inputs.
// ─────────────────────────────────────────────────────────────────────────────

import { ValidationError } from '../types';
import { validateVectorPair } from '../internal/validation';

/** @internal */
function validateVector(v: number[]): void {
  if (v.length === 0) {
    throw new ValidationError('Vector must be non-empty');
  }
}

/**
 * Computes the magnitude (L2 norm / Euclidean length) of a vector.
 *
 * **Formula:** `sqrt(v[0]^2 + v[1]^2 + ... + v[n]^2)`
 *
 * **When to use:** Sanity-check that a model is returning valid vectors.
 * A magnitude of 0 means a zero vector (something went wrong). A magnitude
 * near 1.0 means the vector is already L2-normalized (most embedding models
 * do this). Useful for verifying normalization before using dot product as
 * a similarity metric.
 *
 * @param v - Input vector (must be non-empty)
 * @returns The magnitude (always >= 0)
 * @throws {ValidationError} If the vector is empty
 *
 * @example
 * magnitude([3, 4]);       // 5
 * magnitude([0, 0, 0]);    // 0  (zero vector)
 * magnitude([0.6, 0.8]);   // 1  (already normalized)
 */
export function magnitude(v: number[]): number {
  validateVector(v);
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    sum += v[i] * v[i];
  }
  return Math.sqrt(sum);
}

/**
 * Normalizes a vector to unit length (L2 normalization).
 *
 * Divides each element by the vector's magnitude so the result has
 * magnitude 1.0. If the input is a zero vector, returns a zero vector
 * (avoids division by zero).
 *
 * **When to use:** Call this before storing embeddings in a database that
 * only supports dot product search (e.g., some Postgres pgvector configs).
 * After normalization, dot product equals cosine similarity, so you get
 * cosine-quality results with dot-product speed.
 *
 * Most embedding models (including all local ONNX models) return
 * pre-normalized vectors, so explicit normalization is only needed when
 * working with raw or custom vectors.
 *
 * @param v - Input vector (must be non-empty)
 * @returns A new unit vector with magnitude 1.0 (or zero vector if input is zero)
 * @throws {ValidationError} If the vector is empty
 *
 * @example
 * normalize([3, 4]);    // [0.6, 0.8]  (magnitude = 1.0)
 * normalize([0, 0]);    // [0, 0]       (zero vector stays zero)
 */
export function normalize(v: number[]): number[] {
  validateVector(v);
  const mag = magnitude(v);
  if (mag === 0) return new Array(v.length).fill(0);
  const result = new Array<number>(v.length);
  for (let i = 0; i < v.length; i++) {
    result[i] = v[i] / mag;
  }
  return result;
}

/**
 * Adds two vectors element-wise.
 *
 * Returns a new vector where `result[i] = a[i] + b[i]`.
 *
 * **When to use:** Vector arithmetic for analogy-style operations
 * ("king - man + woman ≈ queen"), combining embeddings with equal weight,
 * or accumulating embedding sums before dividing by count for averaging.
 *
 * @param a - First vector (must have the same length as `b`)
 * @param b - Second vector (must have the same length as `a`)
 * @returns A new vector representing the element-wise sum
 * @throws {ValidationError} If vectors are empty or have different dimensions
 *
 * @example
 * add([1, 2], [3, 4]); // [4, 6]
 */
export function add(a: number[], b: number[]): number[] {
  validateVectorPair(a, b);
  const result = new Array<number>(a.length);
  for (let i = 0; i < a.length; i++) {
    result[i] = a[i] + b[i];
  }
  return result;
}

/**
 * Subtracts the second vector from the first element-wise.
 *
 * Returns a new vector where `result[i] = a[i] - b[i]`.
 *
 * **When to use:** Analogy-style vector arithmetic ("king - man + woman"),
 * computing difference vectors for drift detection, or biasing search
 * results by subtracting an unwanted concept's embedding.
 *
 * @param a - First vector (must have the same length as `b`)
 * @param b - Second vector (must have the same length as `a`)
 * @returns A new vector representing the element-wise difference
 * @throws {ValidationError} If vectors are empty or have different dimensions
 *
 * @example
 * subtract([5, 3], [1, 2]); // [4, 1]
 */
export function subtract(a: number[], b: number[]): number[] {
  validateVectorPair(a, b);
  const result = new Array<number>(a.length);
  for (let i = 0; i < a.length; i++) {
    result[i] = a[i] - b[i];
  }
  return result;
}

/**
 * Scales a vector by a scalar value.
 *
 * Returns a new vector where `result[i] = v[i] * scalar`.
 *
 * **When to use:** Weight an embedding before combining with others.
 * For example, scale a title embedding by 2x before averaging it with the
 * body embedding, since titles carry more semantic signal per token.
 *
 * @param v - Input vector (must be non-empty)
 * @param scalar - The scalar multiplier (can be negative to invert direction)
 * @returns A new vector with each element multiplied by the scalar
 * @throws {ValidationError} If the vector is empty
 *
 * @example
 * scale([1, 2, 3], 2);    // [2, 4, 6]
 * scale([1, 2, 3], -1);   // [-1, -2, -3]  (inverts direction)
 * scale([1, 2, 3], 0.5);  // [0.5, 1, 1.5]
 */
export function scale(v: number[], scalar: number): number[] {
  validateVector(v);
  const result = new Array<number>(v.length);
  for (let i = 0; i < v.length; i++) {
    result[i] = v[i] * scalar;
  }
  return result;
}

/**
 * Checks whether a vector is L2-normalized (magnitude ≈ 1.0).
 *
 * @param vector - Input vector (must be non-empty)
 * @param tolerance - Maximum allowed deviation from magnitude 1.0 (default: 1e-6)
 * @returns `true` if the vector's magnitude is within `tolerance` of 1.0
 * @throws {ValidationError} If the vector is empty
 *
 * @example
 * isNormalized([0.6, 0.8]);   // true  (magnitude = 1.0)
 * isNormalized([1, 1, 1]);    // false (magnitude ≈ 1.73)
 */
export function isNormalized(vector: number[], tolerance = 1e-6): boolean {
  validateVector(vector);
  let sum = 0;
  for (let i = 0; i < vector.length; i++) {
    sum += vector[i] * vector[i];
  }
  return Math.abs(Math.sqrt(sum) - 1.0) <= tolerance;
}

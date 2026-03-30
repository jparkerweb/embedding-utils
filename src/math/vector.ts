import { ValidationError } from '../types';

function validateVector(v: number[]): void {
  if (v.length === 0) {
    throw new ValidationError('Vector must be non-empty');
  }
}

function validateVectorPair(a: number[], b: number[]): void {
  if (a.length === 0 || b.length === 0) {
    throw new ValidationError('Vectors must be non-empty');
  }
  if (a.length !== b.length) {
    throw new ValidationError(`Dimension mismatch: ${a.length} vs ${b.length}`);
  }
}

/**
 * Computes the magnitude (L2 norm) of a vector.
 * @param v - Input vector
 * @returns The magnitude (always >= 0)
 * @throws {ValidationError} If the vector is empty
 * @example
 * magnitude([3, 4]); // 5
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
 * @param v - Input vector
 * @returns A new unit vector (or zero vector if input is zero)
 * @throws {ValidationError} If the vector is empty
 * @example
 * normalize([3, 4]); // [0.6, 0.8]
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
 * @param a - First vector
 * @param b - Second vector
 * @returns A new vector representing the element-wise sum
 * @throws {ValidationError} If vectors are empty or have different dimensions
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
 * @param a - First vector
 * @param b - Second vector
 * @returns A new vector representing the element-wise difference
 * @throws {ValidationError} If vectors are empty or have different dimensions
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
 * @param v - Input vector
 * @param scalar - The scalar multiplier
 * @returns A new vector with each element multiplied by the scalar
 * @throws {ValidationError} If the vector is empty
 * @example
 * scale([1, 2, 3], 2); // [2, 4, 6]
 */
export function scale(v: number[], scalar: number): number[] {
  validateVector(v);
  const result = new Array<number>(v.length);
  for (let i = 0; i < v.length; i++) {
    result[i] = v[i] * scalar;
  }
  return result;
}

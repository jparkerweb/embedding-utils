import { ValidationError, DimensionMismatchError } from '../types';
import type { Vector } from '../types';

/**
 * Validates that a value is a non-empty array of finite numbers or a non-empty Float32Array.
 * @internal
 */
export function validateVector(v: unknown, name = 'Vector'): asserts v is Vector {
  if (v == null) {
    throw new ValidationError(`${name} must not be null or undefined`);
  }
  if (v instanceof Float32Array) {
    if (v.length === 0) {
      throw new ValidationError(`${name} must be non-empty`);
    }
    for (let i = 0; i < v.length; i++) {
      if (!Number.isFinite(v[i])) {
        throw new ValidationError(`${name}[${i}] must be a finite number, got ${v[i]}`);
      }
    }
    return;
  }
  if (!Array.isArray(v)) {
    throw new ValidationError(`${name} must be an array`);
  }
  if (v.length === 0) {
    throw new ValidationError(`${name} must be non-empty`);
  }
  for (let i = 0; i < v.length; i++) {
    if (typeof v[i] !== 'number' || !Number.isFinite(v[i])) {
      throw new ValidationError(`${name}[${i}] must be a finite number, got ${v[i]}`);
    }
  }
}

/**
 * Validates that two vectors are valid and have matching dimensions.
 * @internal
 */
export function validateVectorPair(
  a: Vector,
  b: Vector,
  nameA = 'Vector A',
  nameB = 'Vector B',
): void {
  if (a.length === 0 || b.length === 0) {
    throw new ValidationError('Vectors must be non-empty');
  }
  if (a.length !== b.length) {
    throw new DimensionMismatchError(
      `Dimension mismatch: ${nameA} has ${a.length} dimensions, ${nameB} has ${b.length}`,
    );
  }
}

/**
 * Validates that an array of embeddings is non-empty and all have matching dimensions.
 * @internal
 */
export function validateEmbeddings(embeddings: Vector[], name = 'Embeddings'): void {
  if (embeddings.length === 0) {
    throw new ValidationError(`${name} array must be non-empty`);
  }
  const dim = embeddings[0].length;
  for (let i = 1; i < embeddings.length; i++) {
    if (embeddings[i].length !== dim) {
      throw new DimensionMismatchError(
        `Dimension mismatch: expected ${dim}, got ${embeddings[i].length} at index ${i}`,
      );
    }
  }
}

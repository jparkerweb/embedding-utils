/**
 * Central conversion helpers for the Vector type.
 * @internal
 */

import type { Vector } from '../types';

/**
 * Converts a Vector (number[] or Float32Array) to Float32Array.
 * If the input is already a Float32Array, returns it unchanged (same reference).
 * If the input is a number[], creates a new Float32Array from it.
 */
export function toFloat32(v: Vector): Float32Array {
  if (v instanceof Float32Array) return v;
  return new Float32Array(v);
}

/**
 * Type guard that checks whether a value is a valid Vector
 * (either a number[] or a Float32Array).
 */
export function isVector(v: unknown): v is Vector {
  if (v instanceof Float32Array) return true;
  if (!Array.isArray(v)) return false;
  return v.every((el) => typeof el === 'number');
}

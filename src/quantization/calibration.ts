/**
 * Calibrated scalar quantization for improved quantization precision.
 *
 * Instead of mapping from a fixed [-1, 1] range, calibration learns per-dimension
 * min/max from sample data for a tighter mapping that preserves more information.
 */

import { ValidationError } from '../types';
import type { Vector } from '../types';
import { toFloat32 } from '../internal/vector-utils';

/**
 * Per-dimension calibration data learned from sample embeddings.
 */
export interface QuantizationCalibration {
  /** Per-dimension minimum values. */
  min: Float32Array;
  /** Per-dimension maximum values. */
  max: Float32Array;
  /** Number of dimensions. */
  dimensions: number;
}

/**
 * Learns per-dimension min and max across a set of sample embeddings.
 *
 * @param embeddings - Sample embeddings to calibrate against
 * @returns QuantizationCalibration with per-dimension ranges
 * @throws {ValidationError} If embeddings array is empty
 */
export function calibrate(embeddings: Vector[]): QuantizationCalibration {
  if (embeddings.length === 0) {
    throw new ValidationError('Cannot calibrate with empty embeddings array');
  }

  const first = toFloat32(embeddings[0]);
  const dims = first.length;
  const min = new Float32Array(dims);
  const max = new Float32Array(dims);

  // Initialize with first embedding
  for (let d = 0; d < dims; d++) {
    min[d] = first[d];
    max[d] = first[d];
  }

  // Scan remaining embeddings
  for (let i = 1; i < embeddings.length; i++) {
    const v = toFloat32(embeddings[i]);
    for (let d = 0; d < dims; d++) {
      if (v[d] < min[d]) min[d] = v[d];
      if (v[d] > max[d]) max[d] = v[d];
    }
  }

  return { min, max, dimensions: dims };
}

/**
 * Quantizes an embedding to uint8 using calibration data for tighter value mapping.
 *
 * Maps each dimension from [min[d], max[d]] to [0, 255] instead of the fixed [-1, 1] range.
 *
 * @param embedding - The embedding vector to quantize
 * @param calibration - Calibration data from {@link calibrate}
 * @returns Uint8Array with calibrated quantized values
 * @throws {ValidationError} If embedding dimensions don't match calibration
 */
export function calibratedQuantize(
  embedding: Vector,
  calibration: QuantizationCalibration,
): Uint8Array {
  const v = toFloat32(embedding);
  if (v.length !== calibration.dimensions) {
    throw new ValidationError(
      `Embedding dimensions (${v.length}) do not match calibration dimensions (${calibration.dimensions})`,
    );
  }

  const result = new Uint8Array(v.length);
  for (let d = 0; d < v.length; d++) {
    const range = calibration.max[d] - calibration.min[d];
    if (range === 0) {
      result[d] = 128; // midpoint for constant dimensions
    } else {
      const normalized = (v[d] - calibration.min[d]) / range;
      result[d] = Math.round(Math.min(Math.max(normalized, 0), 1) * 255);
    }
  }
  return result;
}

/**
 * Dequantizes a calibrated uint8 array back to Float32Array.
 *
 * @param data - Calibrated quantized data
 * @param calibration - The same calibration used during quantization
 * @returns Reconstructed Float32Array embedding
 * @throws {ValidationError} If data dimensions don't match calibration
 */
export function calibratedDequantize(
  data: Uint8Array,
  calibration: QuantizationCalibration,
): Float32Array {
  if (data.length !== calibration.dimensions) {
    throw new ValidationError(
      `Data dimensions (${data.length}) do not match calibration dimensions (${calibration.dimensions})`,
    );
  }

  const result = new Float32Array(data.length);
  for (let d = 0; d < data.length; d++) {
    const range = calibration.max[d] - calibration.min[d];
    if (range === 0) {
      result[d] = calibration.min[d];
    } else {
      result[d] = calibration.min[d] + (data[d] / 255) * range;
    }
  }
  return result;
}

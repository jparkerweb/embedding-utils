import { ValidationError } from '../types';
import type { Vector } from '../types';

type QuantizationType = 'fp16' | 'int8' | 'uint8' | 'binary';

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

// Float32 → Float16 truncation via DataView
function float32ToFloat16(value: number): number {
  const buf = new ArrayBuffer(4);
  const view = new DataView(buf);
  view.setFloat32(0, value);
  const bits = view.getUint32(0);

  const sign = (bits >> 31) & 0x1;
  const exp = (bits >> 23) & 0xff;
  const frac = bits & 0x7fffff;

  let hSign = sign;
  let hExp: number;
  let hFrac: number;

  if (exp === 0) {
    // Zero or subnormal — becomes zero in fp16
    hExp = 0;
    hFrac = 0;
  } else if (exp === 0xff) {
    // Inf or NaN
    hExp = 0x1f;
    hFrac = frac ? 0x200 : 0; // preserve NaN vs Inf
  } else {
    const unbiasedExp = exp - 127;
    if (unbiasedExp < -14) {
      // Too small for fp16 — subnormal or zero
      hExp = 0;
      hFrac = 0;
    } else if (unbiasedExp > 15) {
      // Too large — clamp to Inf
      hExp = 0x1f;
      hFrac = 0;
    } else {
      hExp = unbiasedExp + 15;
      hFrac = frac >> 13; // truncate mantissa from 23 to 10 bits
    }
  }

  // Pack fp16 bits, then convert back to float32 for storage
  const fp16Bits = (hSign << 15) | (hExp << 10) | hFrac;

  // Reconstruct float32 from fp16 bits
  const rSign = (fp16Bits >> 15) & 0x1;
  const rExp = (fp16Bits >> 10) & 0x1f;
  const rFrac = fp16Bits & 0x3ff;

  let result: number;
  if (rExp === 0) {
    if (rFrac === 0) {
      result = 0;
    } else {
      // Subnormal fp16
      result = (rFrac / 1024) * Math.pow(2, -14);
    }
  } else if (rExp === 0x1f) {
    result = rFrac === 0 ? Infinity : NaN;
  } else {
    result = Math.pow(2, rExp - 15) * (1 + rFrac / 1024);
  }

  return rSign ? -result : result;
}

/**
 * Quantizes an embedding to a lower-precision typed array.
 * @param embedding - The embedding vector to quantize
 * @param type - Quantization type: 'fp16', 'int8', 'uint8', or 'binary'
 * @returns Quantized typed array (Float32Array for fp16, Int8Array for int8, Uint8Array for uint8/binary)
 * @example
 * quantize([0.5, -0.3, 0.8], 'int8'); // Int8Array [64, -38, 102]
 */
export function quantize(embedding: Vector, type: 'fp16'): Float32Array;
export function quantize(embedding: Vector, type: 'int8'): Int8Array;
export function quantize(embedding: Vector, type: 'uint8'): Uint8Array;
export function quantize(embedding: Vector, type: 'binary'): Uint8Array;
export function quantize(
  embedding: Vector,
  type: QuantizationType,
): Float32Array | Int8Array | Uint8Array {
  switch (type) {
    case 'fp16': {
      const result = new Float32Array(embedding.length);
      for (let i = 0; i < embedding.length; i++) {
        result[i] = float32ToFloat16(embedding[i]);
      }
      return result;
    }

    case 'int8': {
      const result = new Int8Array(embedding.length);
      for (let i = 0; i < embedding.length; i++) {
        result[i] = Math.round(clamp(embedding[i], -1, 1) * 127);
      }
      return result;
    }

    case 'uint8': {
      const result = new Uint8Array(embedding.length);
      for (let i = 0; i < embedding.length; i++) {
        result[i] = Math.round(clamp(embedding[i], 0, 1) * 255);
      }
      return result;
    }

    case 'binary': {
      const byteCount = Math.ceil(embedding.length / 8);
      const result = new Uint8Array(byteCount);
      for (let i = 0; i < embedding.length; i++) {
        if (embedding[i] > 0) {
          const byteIdx = Math.floor(i / 8);
          const bitIdx = 7 - (i % 8); // MSB first
          result[byteIdx] |= 1 << bitIdx;
        }
      }
      return result;
    }
  }
}

/**
 * Dequantizes a typed array back to a regular number array.
 * @param data - Quantized typed array
 * @param type - Quantization type used during quantization
 * @returns Reconstructed embedding vector (approximate for lossy types)
 * @param originalDimension - For binary type, the original vector dimension count.
 *   Binary quantization pads to full bytes, so this truncates the result to
 *   the correct length. Ignored for other quantization types.
 * @example
 * dequantize(new Int8Array([64, -38, 102]), 'int8'); // [0.504, -0.299, 0.803]
 */
export function dequantize(data: Float32Array, type: 'fp16', originalDimension?: number): Float32Array;
export function dequantize(data: Int8Array, type: 'int8', originalDimension?: number): Float32Array;
export function dequantize(data: Uint8Array, type: 'uint8', originalDimension?: number): Float32Array;
export function dequantize(data: Uint8Array, type: 'binary', originalDimension?: number): Float32Array;
export function dequantize(
  data: Float32Array | Int8Array | Uint8Array,
  type: QuantizationType,
  originalDimension?: number,
): Float32Array {
  switch (type) {
    case 'fp16': {
      // Already stored as float32 — return a copy
      return new Float32Array(data);
    }

    case 'int8': {
      const result = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) {
        result[i] = (data as Int8Array)[i] / 127;
      }
      return result;
    }

    case 'uint8': {
      const result = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) {
        result[i] = data[i] / 255;
      }
      return result;
    }

    case 'binary': {
      const byteCount = data.length;
      const totalBits = byteCount * 8;
      const dim = originalDimension !== undefined ? originalDimension : totalBits;
      const result = new Float32Array(dim);
      for (let i = 0; i < dim; i++) {
        const byteIdx = Math.floor(i / 8);
        const bitIdx = 7 - (i % 8); // MSB first
        const bit = (data[byteIdx] >> bitIdx) & 1;
        result[i] = bit === 1 ? 1 : -1;
      }
      return result;
    }
  }
}

/**
 * Returns metadata about a quantization type.
 * @param type - Quantization type: 'fp16', 'int8', 'uint8', or 'binary'
 * @returns Object with bits per value, value range, and description
 * @throws {Error} If the quantization type is unknown
 * @example
 * getQuantizationInfo('int8'); // { bits: 8, range: [-128, 127], description: '...' }
 */
export function getQuantizationInfo(type: string): {
  bits: number;
  range: [number, number];
  description: string;
} {
  switch (type) {
    case 'fp16':
      return {
        bits: 16,
        range: [-65504, 65504],
        description: 'Half-precision floating point (IEEE 754 float16)',
      };
    case 'int8':
      return {
        bits: 8,
        range: [-128, 127],
        description: 'Signed 8-bit integer, maps [-1,1] to [-128,127]',
      };
    case 'uint8':
      return {
        bits: 8,
        range: [0, 255],
        description: 'Unsigned 8-bit integer, maps [0,1] to [0,255]',
      };
    case 'binary':
      return {
        bits: 1,
        range: [-1, 1],
        description: 'Binary quantization, 1 bit per dimension (positive=1, else=-1)',
      };
    default:
      throw new ValidationError(`Unknown quantization type: ${type}`);
  }
}

import { describe, it, expect } from 'vitest';
import { quantize, dequantize, getQuantizationInfo } from '../../src/quantization/quantize';

describe('quantize / dequantize — fp16', () => {
  it('roundtrip preserves values within Float16 precision', () => {
    const embedding = [0.5, -0.25, 0.125, 1.0, -1.0];
    const quantized = quantize(embedding, 'fp16');
    const restored = dequantize(quantized, 'fp16');
    for (let i = 0; i < embedding.length; i++) {
      expect(restored[i]).toBeCloseTo(embedding[i], 3);
    }
  });

  it('returns Float32Array (fp16 stored in f32 container)', () => {
    const quantized = quantize([0.1, 0.2], 'fp16');
    expect(quantized).toBeInstanceOf(Float32Array);
  });

  it('handles zero vector', () => {
    const quantized = quantize([0, 0, 0], 'fp16');
    const restored = dequantize(quantized, 'fp16');
    expect(restored).toEqual([0, 0, 0]);
  });

  it('handles Infinity', () => {
    const quantized = quantize([Infinity, -Infinity], 'fp16');
    const restored = dequantize(quantized, 'fp16');
    expect(restored[0]).toBe(Infinity);
    expect(restored[1]).toBe(-Infinity);
  });

  it('handles NaN', () => {
    const quantized = quantize([NaN], 'fp16');
    const restored = dequantize(quantized, 'fp16');
    expect(restored[0]).toBeNaN();
  });

  it('handles very large values (overflow to Inf)', () => {
    const quantized = quantize([100000], 'fp16');
    const restored = dequantize(quantized, 'fp16');
    expect(restored[0]).toBe(Infinity);
  });

  it('handles very small subnormal values', () => {
    // Value small enough to be subnormal in fp16 but not zero in fp32
    const tiny = 1e-8;
    const quantized = quantize([tiny], 'fp16');
    const restored = dequantize(quantized, 'fp16');
    // Should become 0 (too small for fp16)
    expect(restored[0]).toBe(0);
  });
});

describe('quantize / dequantize — int8', () => {
  it('maps [-1,1] to [-128,127] and roundtrips within error', () => {
    const embedding = [1.0, -1.0, 0.0, 0.5, -0.5];
    const quantized = quantize(embedding, 'int8');
    expect(quantized).toBeInstanceOf(Int8Array);
    const restored = dequantize(quantized, 'int8');
    for (let i = 0; i < embedding.length; i++) {
      expect(restored[i]).toBeCloseTo(embedding[i], 1);
    }
  });

  it('clamps values outside [-1,1]', () => {
    const embedding = [2.0, -3.0];
    const quantized = quantize(embedding, 'int8');
    // clamp(-3,-1,1)=-1, round(-1*127)=-127; clamp(2,-1,1)=1, round(1*127)=127
    expect(quantized[0]).toBe(127);
    expect(quantized[1]).toBe(-127);
  });

  it('handles zero vector', () => {
    const quantized = quantize([0, 0, 0], 'int8');
    const restored = dequantize(quantized, 'int8');
    for (const v of restored) {
      expect(v).toBeCloseTo(0, 1);
    }
  });

  it('handles all-ones vector', () => {
    const quantized = quantize([1, 1, 1], 'int8');
    const restored = dequantize(quantized, 'int8');
    for (const v of restored) {
      expect(v).toBeCloseTo(1.0, 1);
    }
  });
});

describe('quantize / dequantize — uint8', () => {
  it('maps [0,1] to [0,255] and roundtrips within error', () => {
    const embedding = [0.0, 0.5, 1.0, 0.25, 0.75];
    const quantized = quantize(embedding, 'uint8');
    expect(quantized).toBeInstanceOf(Uint8Array);
    const restored = dequantize(quantized, 'uint8');
    for (let i = 0; i < embedding.length; i++) {
      expect(restored[i]).toBeCloseTo(embedding[i], 1);
    }
  });

  it('clamps values outside [0,1]', () => {
    const embedding = [-0.5, 1.5];
    const quantized = quantize(embedding, 'uint8');
    expect(quantized[0]).toBe(0);
    expect(quantized[1]).toBe(255);
  });

  it('handles zero vector', () => {
    const quantized = quantize([0, 0, 0], 'uint8');
    const restored = dequantize(quantized, 'uint8');
    for (const v of restored) {
      expect(v).toBeCloseTo(0, 1);
    }
  });

  it('handles all-ones vector', () => {
    const quantized = quantize([1, 1, 1], 'uint8');
    const restored = dequantize(quantized, 'uint8');
    for (const v of restored) {
      expect(v).toBeCloseTo(1.0, 1);
    }
  });
});

describe('quantize / dequantize — binary', () => {
  it('positive → 1, negative/zero → 0 packed bits', () => {
    // 8 values to fill one byte: [+, -, +, 0, +, -, -, +]
    const embedding = [0.5, -0.3, 0.1, 0.0, 1.0, -1.0, -0.5, 0.9];
    const quantized = quantize(embedding, 'binary');
    expect(quantized).toBeInstanceOf(Uint8Array);
    // Bits: 1,0,1,0,1,0,0,1 = 0b10101001 = 169
    expect(quantized[0]).toBe(0b10101001);
  });

  it('dequantize binary returns 1/-1 values', () => {
    const embedding = [0.5, -0.3, 0.1, 0.0, 1.0, -1.0, -0.5, 0.9];
    const quantized = quantize(embedding, 'binary');
    const restored = dequantize(quantized, 'binary');
    expect(restored).toEqual([1, -1, 1, -1, 1, -1, -1, 1]);
  });

  it('handles non-multiple-of-8 length', () => {
    const embedding = [0.5, -0.3, 0.1];
    const quantized = quantize(embedding, 'binary');
    expect(quantized).toBeInstanceOf(Uint8Array);
    // Should pack into 1 byte with padding
    expect(quantized.length).toBe(1);
  });

  it('handles zero vector', () => {
    const quantized = quantize([0, 0, 0, 0], 'binary');
    const restored = dequantize(quantized, 'binary');
    // All zeros → all -1
    expect(restored[0]).toBe(-1);
    expect(restored[1]).toBe(-1);
    expect(restored[2]).toBe(-1);
    expect(restored[3]).toBe(-1);
  });
});

describe('getQuantizationInfo', () => {
  it('returns info for fp16', () => {
    const info = getQuantizationInfo('fp16');
    expect(info.bits).toBe(16);
    expect(info.range).toEqual([-65504, 65504]);
    expect(typeof info.description).toBe('string');
  });

  it('returns info for int8', () => {
    const info = getQuantizationInfo('int8');
    expect(info.bits).toBe(8);
    expect(info.range).toEqual([-128, 127]);
    expect(typeof info.description).toBe('string');
  });

  it('returns info for uint8', () => {
    const info = getQuantizationInfo('uint8');
    expect(info.bits).toBe(8);
    expect(info.range).toEqual([0, 255]);
    expect(typeof info.description).toBe('string');
  });

  it('returns info for binary', () => {
    const info = getQuantizationInfo('binary');
    expect(info.bits).toBe(1);
    expect(info.range).toEqual([0, 1]);
    expect(typeof info.description).toBe('string');
  });

  it('throws for unknown type', () => {
    expect(() => getQuantizationInfo('unknown')).toThrow('Unknown quantization type');
  });
});

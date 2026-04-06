import { describe, it, expect } from 'vitest';
import { calibrate, calibratedQuantize, calibratedDequantize } from '../../src/quantization/calibration';
import { quantize, dequantize } from '../../src/quantization/quantize';
import { ValidationError } from '../../src/types';

describe('calibrate', () => {
  it('learns correct per-dimension min/max from sample data', () => {
    const embeddings = [
      new Float32Array([0.1, -0.5, 0.3]),
      new Float32Array([0.4, 0.2, -0.1]),
      new Float32Array([0.2, 0.8, 0.5]),
    ];
    const cal = calibrate(embeddings);

    expect(cal.dimensions).toBe(3);
    expect(cal.min[0]).toBeCloseTo(0.1);
    expect(cal.max[0]).toBeCloseTo(0.4);
    expect(cal.min[1]).toBeCloseTo(-0.5);
    expect(cal.max[1]).toBeCloseTo(0.8);
    expect(cal.min[2]).toBeCloseTo(-0.1);
    expect(cal.max[2]).toBeCloseTo(0.5);
  });

  it('calibrated quantize produces tighter value ranges than fixed [-1,1] mapping', () => {
    // Embeddings in a narrow range [0.3, 0.7] — calibrated should use full uint8 range
    const embeddings = [
      new Float32Array([0.3, 0.3]),
      new Float32Array([0.7, 0.7]),
      new Float32Array([0.5, 0.5]),
    ];
    const cal = calibrate(embeddings);
    const calibrated = calibratedQuantize(new Float32Array([0.5, 0.5]), cal);

    // With calibration: 0.5 maps to (0.5-0.3)/(0.7-0.3)*255 = 0.5*255 ≈ 128
    expect(calibrated[0]).toBeCloseTo(128, 0);

    // Without calibration (uint8 maps [0,1] to [0,255]): 0.5 maps to 128
    // But for values like 0.3, calibrated gives 0 (full range), uncalibrated gives 77
    const calLow = calibratedQuantize(new Float32Array([0.3, 0.3]), cal);
    const uncalLow = quantize(new Float32Array([0.3, 0.3]), 'uint8');
    expect(calLow[0]).toBe(0); // Full range utilization
    expect(uncalLow[0]).toBe(77); // Only partial range used
  });

  it('roundtrip: calibrate → quantize → dequantize has lower error than uncalibrated', () => {
    // Embeddings in narrow range [0.2, 0.6]
    const embeddings: Float32Array[] = [];
    for (let i = 0; i < 50; i++) {
      const v = new Float32Array(4);
      for (let d = 0; d < 4; d++) v[d] = 0.2 + (i / 50) * 0.4;
      embeddings.push(v);
    }

    const cal = calibrate(embeddings);
    const testVec = new Float32Array([0.25, 0.35, 0.45, 0.55]);

    // Calibrated roundtrip
    const calQuantized = calibratedQuantize(testVec, cal);
    const calRestored = calibratedDequantize(calQuantized, cal);

    // Uncalibrated roundtrip (uint8 maps [0,1] to [0,255])
    const uncalQuantized = quantize(testVec, 'uint8');
    const uncalRestored = dequantize(uncalQuantized, 'uint8');

    let calError = 0;
    let uncalError = 0;
    for (let d = 0; d < 4; d++) {
      calError += Math.abs(testVec[d] - calRestored[d]);
      uncalError += Math.abs(testVec[d] - uncalRestored[d]);
    }

    expect(calError).toBeLessThan(uncalError);
  });

  it('calibration with single embedding', () => {
    const cal = calibrate([new Float32Array([0.5, -0.3])]);
    expect(cal.dimensions).toBe(2);
    expect(cal.min[0]).toBeCloseTo(0.5);
    expect(cal.max[0]).toBeCloseTo(0.5);
    // Constant dimension → quantizes to midpoint 128
    const q = calibratedQuantize(new Float32Array([0.5, -0.3]), cal);
    expect(q[0]).toBe(128);
  });

  it('calibration dimensions mismatch → throws ValidationError', () => {
    const cal = calibrate([new Float32Array([0.1, 0.2, 0.3])]);
    expect(() => calibratedQuantize(new Float32Array([0.1, 0.2]), cal)).toThrow(ValidationError);
  });

  it('empty embeddings array → throws ValidationError', () => {
    expect(() => calibrate([])).toThrow(ValidationError);
  });
});

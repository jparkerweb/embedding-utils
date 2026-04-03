type QuantizationType = 'fp16' | 'int8' | 'uint8' | 'binary';

/**
 * Estimates memory savings from quantizing embeddings.
 *
 * @param dimensions - Number of dimensions per embedding vector
 * @param count - Number of embedding vectors
 * @param quantizationType - Target quantization type
 * @returns Object with original/quantized byte sizes, absolute savings, and ratio
 * @example
 * estimateMemorySavings(384, 10000, 'int8');
 * // { originalBytes: 15360000, quantizedBytes: 3840000, savings: 11520000, ratio: 0.25 }
 */
export function estimateMemorySavings(
  dimensions: number,
  count: number,
  quantizationType: QuantizationType,
): {
  originalBytes: number;
  quantizedBytes: number;
  savings: number;
  ratio: number;
} {
  const originalBytes = count * dimensions * 4; // float32

  let quantizedBytes: number;
  switch (quantizationType) {
    case 'fp16':
      quantizedBytes = count * dimensions * 4; // stored as float32 (fp16 truncated values)
      break;
    case 'int8':
      quantizedBytes = count * dimensions;
      break;
    case 'uint8':
      quantizedBytes = count * dimensions;
      break;
    case 'binary':
      quantizedBytes = count * Math.ceil(dimensions / 8);
      break;
  }

  const savings = originalBytes - quantizedBytes;
  const ratio = quantizedBytes / originalBytes;

  return { originalBytes, quantizedBytes, savings, ratio };
}

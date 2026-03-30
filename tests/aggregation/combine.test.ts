import { describe, it, expect, vi } from 'vitest';
import { combineEmbeddings } from '../../src/aggregation/combine';
import type { EmbeddingProvider } from '../../src/types';
import { ValidationError } from '../../src/types';

function createMockProvider(embeddings: number[][]): EmbeddingProvider {
  return {
    name: 'mock',
    dimensions: embeddings[0]?.length ?? 0,
    embed: vi.fn().mockResolvedValue({
      embeddings,
      model: 'mock-model',
      dimensions: embeddings[0]?.length ?? 0,
    }),
  };
}

describe('combineEmbeddings', () => {
  it('should combine multiple texts into single averaged embedding', async () => {
    const provider = createMockProvider([
      [1, 2, 3],
      [3, 4, 5],
    ]);

    const result = await combineEmbeddings(['hello', 'world'], provider);

    expect(result).toEqual([2, 3, 4]);
    expect(provider.embed).toHaveBeenCalledWith(['hello', 'world'], undefined);
  });

  it('should use custom aggregation function', async () => {
    const provider = createMockProvider([
      [1, 2, 3],
      [4, 5, 6],
    ]);

    // Use max pooling as custom aggregation
    const maxPool = (embeddings: number[][]): number[] => {
      const dim = embeddings[0].length;
      const result = new Array(dim);
      for (let j = 0; j < dim; j++) {
        result[j] = Math.max(...embeddings.map((e) => e[j]));
      }
      return result;
    };

    const result = await combineEmbeddings(['a', 'b'], provider, {
      aggregate: maxPool,
    });

    expect(result).toEqual([4, 5, 6]);
  });

  it('should use provider embed method', async () => {
    const provider = createMockProvider([[0.5, 0.5]]);

    await combineEmbeddings(['test'], provider);

    expect(provider.embed).toHaveBeenCalledTimes(1);
  });

  it('should return a single number[]', async () => {
    const provider = createMockProvider([[1, 2], [3, 4]]);

    const result = await combineEmbeddings(['a', 'b'], provider);

    expect(Array.isArray(result)).toBe(true);
    expect(typeof result[0]).toBe('number');
  });

  it('should throw ValidationError on empty input', async () => {
    const provider = createMockProvider([]);

    await expect(combineEmbeddings([], provider)).rejects.toThrow(ValidationError);
  });

  it('should pass embedOptions to provider', async () => {
    const provider = createMockProvider([[0.1, 0.2]]);

    await combineEmbeddings(['test'], provider, {
      embedOptions: { inputType: 'document', dimensions: 2 },
    });

    expect(provider.embed).toHaveBeenCalledWith(['test'], {
      inputType: 'document',
      dimensions: 2,
    });
  });
});

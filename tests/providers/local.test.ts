import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createLocalProvider } from '../../src/providers/local';

// Mock @huggingface/transformers
const mockPipelineFn = vi.fn();
const mockPipeline = vi.fn();

vi.mock('@huggingface/transformers', () => ({
  pipeline: mockPipeline,
}));

describe('createLocalProvider', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockPipelineFn.mockReset();
    mockPipeline.mockReset();
    mockPipeline.mockResolvedValue(mockPipelineFn);
  });

  it('should have name "local"', () => {
    const provider = createLocalProvider();
    expect(provider.name).toBe('local');
  });

  it('should have dimensions null (determined at runtime)', () => {
    const provider = createLocalProvider();
    expect(provider.dimensions).toBeNull();
  });

  it('should embed a single string input', async () => {
    mockPipelineFn.mockResolvedValue({ tolist: () => [[0.1, 0.2, 0.3]] });
    const provider = createLocalProvider();

    const result = await provider.embed('hello');

    expect(result.embeddings).toEqual([[0.1, 0.2, 0.3]]);
    expect(result.dimensions).toBe(3);
    expect(result.model).toBe('Xenova/all-MiniLM-L12-v2');
    expect(mockPipeline).toHaveBeenCalledWith(
      'feature-extraction',
      'Xenova/all-MiniLM-L12-v2',
      expect.any(Object),
    );
  });

  it('should embed batch string[] input', async () => {
    mockPipelineFn.mockResolvedValue({
      tolist: () => [
        [0.1, 0.2],
        [0.3, 0.4],
      ],
    });
    const provider = createLocalProvider();

    const result = await provider.embed(['hello', 'world']);

    expect(result.embeddings).toEqual([
      [0.1, 0.2],
      [0.3, 0.4],
    ]);
    expect(result.dimensions).toBe(2);
  });

  it('should prepend documentPrefix when inputType is "document"', async () => {
    mockPipelineFn.mockResolvedValue({ tolist: () => [[0.1, 0.2]] });
    const provider = createLocalProvider({
      documentPrefix: 'passage: ',
    });

    await provider.embed('hello', { inputType: 'document' });

    expect(mockPipelineFn).toHaveBeenCalledWith(
      ['passage: hello'],
      expect.any(Object),
    );
  });

  it('should prepend queryPrefix when inputType is "query"', async () => {
    mockPipelineFn.mockResolvedValue({ tolist: () => [[0.1, 0.2]] });
    const provider = createLocalProvider({
      queryPrefix: 'query: ',
    });

    await provider.embed('hello', { inputType: 'query' });

    expect(mockPipelineFn).toHaveBeenCalledWith(
      ['query: hello'],
      expect.any(Object),
    );
  });

  it('should use default model Xenova/all-MiniLM-L12-v2', async () => {
    mockPipelineFn.mockResolvedValue({ tolist: () => [[0.1]] });
    const provider = createLocalProvider();

    await provider.embed('test');

    expect(mockPipeline).toHaveBeenCalledWith(
      'feature-extraction',
      'Xenova/all-MiniLM-L12-v2',
      expect.any(Object),
    );
  });

  it('should use custom model from config', async () => {
    mockPipelineFn.mockResolvedValue({ tolist: () => [[0.1]] });
    const provider = createLocalProvider({ model: 'Xenova/bge-small-en-v1.5' });

    await provider.embed('test');

    expect(mockPipeline).toHaveBeenCalledWith(
      'feature-extraction',
      'Xenova/bge-small-en-v1.5',
      expect.any(Object),
    );
  });

  it('should return EmbeddingResult shape', async () => {
    mockPipelineFn.mockResolvedValue({ tolist: () => [[0.1, 0.2, 0.3]] });
    const provider = createLocalProvider();

    const result = await provider.embed('test');

    expect(result).toHaveProperty('embeddings');
    expect(result).toHaveProperty('model');
    expect(result).toHaveProperty('dimensions');
    expect(Array.isArray(result.embeddings)).toBe(true);
    expect(typeof result.model).toBe('string');
    expect(typeof result.dimensions).toBe('number');
  });

  it('should truncate dimensions when dimensions option is set', async () => {
    mockPipelineFn.mockResolvedValue({
      tolist: () => [[0.1, 0.2, 0.3, 0.4, 0.5]],
    });
    const provider = createLocalProvider();

    const result = await provider.embed('test', { dimensions: 3 });

    expect(result.embeddings).toEqual([[0.1, 0.2, 0.3]]);
    expect(result.dimensions).toBe(3);
  });

  it('should cache pipeline instance (reuse across calls)', async () => {
    mockPipelineFn.mockResolvedValue({ tolist: () => [[0.1]] });
    const provider = createLocalProvider();

    await provider.embed('first');
    await provider.embed('second');

    // Pipeline created only once
    expect(mockPipeline).toHaveBeenCalledTimes(1);
  });

  it('should cache embedding results (same input returns cached)', async () => {
    mockPipelineFn.mockResolvedValue({ tolist: () => [[0.1, 0.2]] });
    const provider = createLocalProvider();

    const result1 = await provider.embed('hello');
    const result2 = await provider.embed('hello');

    expect(result1.embeddings).toEqual(result2.embeddings);
    // Pipeline fn called only once for same input
    expect(mockPipelineFn).toHaveBeenCalledTimes(1);
  });

  it('should handle AbortSignal cancellation', async () => {
    const controller = new AbortController();
    controller.abort();

    mockPipelineFn.mockResolvedValue({ tolist: () => [[0.1]] });
    const provider = createLocalProvider();

    await expect(
      provider.embed('test', { signal: controller.signal }),
    ).rejects.toThrow(/abort/i);
  });
});

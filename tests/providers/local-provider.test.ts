import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createLocalProvider, disposeLocalPipelines } from '../../src/providers/local';

// Mocked transformers loader: a spy-able pipeline factory plus a mutable env
// object so we can assert config wiring without downloading any model.
const mockPipelineFn = vi.fn();
const mockPipeline = vi.fn();
const mockEnv: Record<string, unknown> = {};

vi.mock('@huggingface/transformers', () => ({
  pipeline: mockPipeline,
  env: mockEnv,
}));

describe('createLocalProvider config wiring + per-input cache', () => {
  beforeEach(async () => {
    // Empty the process-level pipeline registry so each test constructs fresh.
    await disposeLocalPipelines();
    vi.clearAllMocks();
    mockPipelineFn.mockReset();
    mockPipeline.mockReset();
    mockPipeline.mockResolvedValue(mockPipelineFn);
    // Reset env between tests.
    for (const key of Object.keys(mockEnv)) {
      delete mockEnv[key];
    }
  });

  describe('config wiring', () => {
    it('does NOT pass device into pipeline options when device is "webgpu"', async () => {
      mockPipelineFn.mockResolvedValue({ tolist: () => [[0.1, 0.2]] });
      const provider = createLocalProvider({ device: 'webgpu' });

      await provider.embed('hello');

      const opts = mockPipeline.mock.calls[0][2] as Record<string, unknown>;
      expect(opts).not.toHaveProperty('device');
    });

    it('DOES pass a non-webgpu device into pipeline options', async () => {
      mockPipelineFn.mockResolvedValue({ tolist: () => [[0.1, 0.2]] });
      const provider = createLocalProvider({ device: 'cpu' });

      await provider.embed('hello');

      const opts = mockPipeline.mock.calls[0][2] as Record<string, unknown>;
      expect(opts.device).toBe('cpu');
    });

    it('maps precision (incl. "q4") to dtype', async () => {
      mockPipelineFn.mockResolvedValue({ tolist: () => [[0.1, 0.2]] });
      const provider = createLocalProvider({ precision: 'q4' });

      await provider.embed('hello');

      const opts = mockPipeline.mock.calls[0][2] as Record<string, unknown>;
      expect(opts.dtype).toBe('q4');
    });

    it('applies modelPath, cacheDir, and allowRemoteModels to env', async () => {
      mockPipelineFn.mockResolvedValue({ tolist: () => [[0.1, 0.2]] });
      const provider = createLocalProvider({
        modelPath: '/models/local',
        cacheDir: '/tmp/hf-cache',
        allowRemoteModels: false,
      });

      await provider.embed('hello');

      expect(mockEnv.localModelPath).toBe('/models/local');
      expect(mockEnv.cacheDir).toBe('/tmp/hf-cache');
      expect(mockEnv.allowRemoteModels).toBe(false);
    });

    it('defaults allowRemoteModels to true when not configured', async () => {
      mockPipelineFn.mockResolvedValue({ tolist: () => [[0.1, 0.2]] });
      const provider = createLocalProvider();

      await provider.embed('hello');

      expect(mockEnv.allowRemoteModels).toBe(true);
    });
  });

  describe('per-input cache', () => {
    it('runs the pipeline only on new texts across overlapping batches', async () => {
      // First batch: ['a','b'] -> two embeddings. Second batch ['b','c'] should
      // only run the pipeline on 'c' since 'b' is already cached.
      mockPipelineFn
        .mockResolvedValueOnce({
          tolist: () => [
            [1, 0],
            [0, 1],
          ],
        })
        .mockResolvedValueOnce({ tolist: () => [[1, 1]] });

      const provider = createLocalProvider();

      const first = await provider.embed(['a', 'b']);
      const second = await provider.embed(['b', 'c']);

      // The pipeline fn was invoked once per batch (not per text).
      expect(mockPipelineFn).toHaveBeenCalledTimes(2);

      // Second call only received the cache miss 'c'.
      expect(mockPipelineFn.mock.calls[1][0]).toEqual(['c']);

      // 'b' vector is consistent across both batches.
      const bFromFirst = first.embeddings[1];
      const bFromSecond = second.embeddings[0];
      expect(Array.from(bFromSecond)).toEqual(Array.from(bFromFirst));
      expect(Array.from(bFromSecond)).toEqual([0, 1]);

      // 'c' resolved correctly and order is preserved.
      expect(Array.from(second.embeddings[1])).toEqual([1, 1]);
    });

    it('does not call the pipeline at all when every text is cached', async () => {
      mockPipelineFn.mockResolvedValue({ tolist: () => [[0.5, 0.5]] });
      const provider = createLocalProvider();

      await provider.embed('repeat');
      await provider.embed('repeat');

      expect(mockPipelineFn).toHaveBeenCalledTimes(1);
    });
  });

  describe('empty input', () => {
    it('returns { embeddings: [], dimensions: 0 } without touching the pipeline', async () => {
      const provider = createLocalProvider();

      const result = await provider.embed([]);

      expect(result.embeddings).toEqual([]);
      expect(result.dimensions).toBe(0);
      expect(result.model).toBe('Xenova/all-MiniLM-L12-v2');
      expect(mockPipeline).not.toHaveBeenCalled();
      expect(mockPipelineFn).not.toHaveBeenCalled();
    });
  });
});

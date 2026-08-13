import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createLocalProvider, disposeLocalPipelines } from '../../src/providers/local';

// Mocked transformers loader so no model is downloaded. Each pipeline gets a
// dispose() spy so we can assert disposeLocalPipelines() releases sessions.
const mockPipeline = vi.fn();

vi.mock('@huggingface/transformers', () => ({
  pipeline: mockPipeline,
  env: {},
}));

/** Builds a fresh mock pipeline fn returning a fixed vector, with dispose(). */
function makePipelineFn(vector: number[]) {
  const fn = vi.fn(async (texts: string[]) => ({
    tolist: () => texts.map(() => vector),
  })) as ReturnType<typeof vi.fn> & { dispose: ReturnType<typeof vi.fn> };
  fn.dispose = vi.fn(async () => undefined);
  return fn;
}

describe('createLocalProvider pipeline reuse (process-level registry)', () => {
  beforeEach(async () => {
    await disposeLocalPipelines();
    vi.clearAllMocks();
    mockPipeline.mockReset();
  });

  it('shares one pipeline across providers with identical construction config', async () => {
    mockPipeline.mockResolvedValue(makePipelineFn([0.1, 0.2]));

    const a = createLocalProvider({ model: 'Xenova/all-MiniLM-L6-v2' });
    const b = createLocalProvider({ model: 'Xenova/all-MiniLM-L6-v2' });

    await a.embed('hello');
    await b.embed('world');

    expect(mockPipeline).toHaveBeenCalledTimes(1);
  });

  it('produces identical vectors from a shared pipeline', async () => {
    mockPipeline.mockResolvedValue(makePipelineFn([0.25, 0.75]));

    const a = createLocalProvider({ model: 'Xenova/all-MiniLM-L6-v2' });
    const b = createLocalProvider({ model: 'Xenova/all-MiniLM-L6-v2' });

    const resA = await a.embed('same text');
    const resB = await b.embed('same text');

    expect(Array.from(resA.embeddings[0])).toEqual(Array.from(resB.embeddings[0]));
  });

  it('shares across providers differing only in pooling/prefixes (per-inference args)', async () => {
    mockPipeline.mockResolvedValue(makePipelineFn([1, 0]));

    const meanProvider = createLocalProvider({ model: 'Xenova/all-MiniLM-L6-v2', pooling: 'mean' });
    const clsProvider = createLocalProvider({
      model: 'Xenova/all-MiniLM-L6-v2',
      pooling: 'cls',
      documentPrefix: 'passage: ',
    });

    await meanProvider.embed('a');
    await clsProvider.embed('b');

    expect(mockPipeline).toHaveBeenCalledTimes(1);
  });

  it.each([
    ['model', { model: 'Xenova/all-MiniLM-L6-v2' }, { model: 'Xenova/bge-small-en-v1.5' }],
    ['precision', { precision: 'fp32' as const }, { precision: 'q8' as const }],
    ['device', { device: 'cpu' }, { device: 'dml' }],
    ['modelPath', { modelPath: '/models/a' }, { modelPath: '/models/b' }],
    ['cacheDir', { cacheDir: '/cache/a' }, { cacheDir: '/cache/b' }],
    ['allowRemoteModels', { allowRemoteModels: true }, { allowRemoteModels: false }],
  ])('constructs separate pipelines when %s differs', async (_field, cfgA, cfgB) => {
    mockPipeline.mockResolvedValue(makePipelineFn([0.5]));

    await createLocalProvider(cfgA).embed('x');
    await createLocalProvider(cfgB).embed('y');

    expect(mockPipeline).toHaveBeenCalledTimes(2);
  });

  it('reuse: false opts out of sharing (private session per provider)', async () => {
    mockPipeline.mockResolvedValue(makePipelineFn([0.5]));

    await createLocalProvider({ reuse: false }).embed('x');
    await createLocalProvider({ reuse: false }).embed('y');
    await createLocalProvider().embed('z'); // shared path, separate from both

    expect(mockPipeline).toHaveBeenCalledTimes(3);
  });

  it('a private (reuse: false) provider still reuses its own pipeline across calls', async () => {
    mockPipeline.mockResolvedValue(makePipelineFn([0.5]));
    const provider = createLocalProvider({ reuse: false });

    await provider.embed('first');
    await provider.embed('second');

    expect(mockPipeline).toHaveBeenCalledTimes(1);
  });

  it('disposeLocalPipelines() disposes shared sessions and resets the registry', async () => {
    const pipelineFn = makePipelineFn([0.1]);
    mockPipeline.mockResolvedValue(pipelineFn);

    await createLocalProvider().embed('x');
    await disposeLocalPipelines();

    expect(pipelineFn.dispose).toHaveBeenCalledTimes(1);

    // Next provider constructs a fresh pipeline.
    mockPipeline.mockResolvedValue(makePipelineFn([0.2]));
    await createLocalProvider().embed('y');
    expect(mockPipeline).toHaveBeenCalledTimes(2);
  });

  it('retries construction after a failure instead of caching the rejection', async () => {
    mockPipeline
      .mockRejectedValueOnce(new Error('flaky download'))
      .mockResolvedValueOnce(makePipelineFn([0.9]));

    await expect(createLocalProvider().embed('x')).rejects.toThrow('flaky download');

    const result = await createLocalProvider().embed('x');
    expect(Array.from(result.embeddings[0])).toEqual(Array.from(new Float32Array([0.9])));
    expect(mockPipeline).toHaveBeenCalledTimes(2);
  });
});

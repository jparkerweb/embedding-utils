import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createTokenizer, disposeLocalTokenizers } from '../../src/text/tokenizer';

// Mocked transformers loader so no tokenizer files are read from disk.
const mockFromPretrained = vi.fn();

vi.mock('@huggingface/transformers', () => ({
  AutoTokenizer: {
    from_pretrained: mockFromPretrained,
  },
  env: {},
}));

/** A stub tokenizer instance: token count = word count. */
function makeTokenizerInstance() {
  return (text: string) => ({
    input_ids: { size: text.split(/\s+/).filter(Boolean).length },
  });
}

const MODEL = 'Xenova/all-MiniLM-L6-v2';

describe('createTokenizer reuse (process-level registry)', () => {
  beforeEach(() => {
    disposeLocalTokenizers();
    vi.clearAllMocks();
    mockFromPretrained.mockReset();
    mockFromPretrained.mockResolvedValue(makeTokenizerInstance());
  });

  it('loads the underlying tokenizer once across createTokenizer calls with the same config', async () => {
    const a = createTokenizer(MODEL);
    const b = createTokenizer(MODEL);

    await a.load();
    await b.load();

    expect(mockFromPretrained).toHaveBeenCalledTimes(1);
    expect(a.count('hello world')).toBe(2);
    expect(b.count('hello world')).toBe(2);
  });

  it('loads separately when modelPath/cacheDir differ', async () => {
    const a = createTokenizer(MODEL, { modelPath: '/models/a' });
    const b = createTokenizer(MODEL, { modelPath: '/models/b' });

    await a.load();
    await b.load();

    expect(mockFromPretrained).toHaveBeenCalledTimes(2);
  });

  it('reuse: false opts out of sharing', async () => {
    const a = createTokenizer(MODEL, { reuse: false });
    const b = createTokenizer(MODEL, { reuse: false });

    await a.load();
    await b.load();

    expect(mockFromPretrained).toHaveBeenCalledTimes(2);
  });

  it('disposeLocalTokenizers() resets the registry; existing tokenizers keep working', async () => {
    const a = createTokenizer(MODEL);
    await a.load();

    disposeLocalTokenizers();

    // Existing instance still counts fine.
    expect(a.count('one two three')).toBe(3);

    // New tokenizer re-loads.
    const b = createTokenizer(MODEL);
    await b.load();
    expect(mockFromPretrained).toHaveBeenCalledTimes(2);
  });

  it('retries load after a failure instead of caching the rejection', async () => {
    mockFromPretrained
      .mockRejectedValueOnce(new Error('corrupt tokenizer.json'))
      .mockResolvedValueOnce(makeTokenizerInstance());

    const failing = createTokenizer(MODEL);
    await expect(failing.load()).rejects.toThrow('corrupt tokenizer.json');

    const retry = createTokenizer(MODEL);
    await retry.load();
    expect(retry.count('a b')).toBe(2);
    expect(mockFromPretrained).toHaveBeenCalledTimes(2);
  });
});

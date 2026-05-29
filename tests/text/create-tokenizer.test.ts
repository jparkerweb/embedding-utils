import { describe, it, expect, beforeAll } from 'vitest';
import { createTokenizer } from '../../src/text/tokenizer';
import { EmbeddingUtilsError } from '../../src/types';

const MODEL = 'Xenova/all-MiniLM-L6-v2';
const FIXED_STRINGS = [
  'hello world',
  'The quick brown fox jumps over the lazy dog.',
  'embedding-utils tokenizer parity check',
  '',
];

// Real model load can hit the network/disk on a cold cache.
const LOAD_TIMEOUT = 120_000;

describe('createTokenizer (real load)', () => {
  it('throws EmbeddingUtilsError when count() is called before load()', () => {
    const tok = createTokenizer(MODEL);
    expect(() => tok.count('hello')).toThrow(EmbeddingUtilsError);
    expect(() => tok.countBatch(['hello'])).toThrow(EmbeddingUtilsError);
  });

  it('populates maxTokens (256) and modelId from the registry', () => {
    const tok = createTokenizer(MODEL);
    expect(tok.maxTokens).toBe(256);
    expect(tok.modelId).toBe(MODEL);
  });

  describe('counts against transformers AutoTokenizer', () => {
    // Reference counts computed directly via AutoTokenizer for parity.
    let tok: ReturnType<typeof createTokenizer>;
    let reference: number[];

    beforeAll(async () => {
      const { AutoTokenizer } = await import('@huggingface/transformers');
      const autoTok = await AutoTokenizer.from_pretrained(MODEL);
      reference = FIXED_STRINGS.map((s) => (autoTok(s) as { input_ids: { size: number } }).input_ids.size);

      tok = createTokenizer(MODEL);
      await tok.load();
    }, LOAD_TIMEOUT);

    it('count() matches transformers input_ids.size for fixed strings', () => {
      FIXED_STRINGS.forEach((s, i) => {
        expect(tok.count(s)).toBe(reference[i]);
      });
    });

    it('countBatch() equals element-wise count()', () => {
      const batch = tok.countBatch(FIXED_STRINGS);
      const elementWise = FIXED_STRINGS.map((s) => tok.count(s));
      expect(batch).toEqual(elementWise);
      expect(batch).toEqual(reference);
    });

    it('load() is idempotent (second call is a no-op)', async () => {
      await tok.load();
      expect(tok.count('hello world')).toBe(reference[0]);
    });
  });
});

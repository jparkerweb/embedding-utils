import { describe, it, expect } from 'vitest';
import { getTokenizerInfo } from '../../src/text/tokenizer';

describe('getTokenizerInfo', () => {
  it('returns info for a known model', () => {
    const info = getTokenizerInfo('Xenova/all-MiniLM-L12-v2');
    expect(info).toBeDefined();
    expect(info!.maxTokens).toBe(256);
    expect(info!.modelId).toBe('Xenova/all-MiniLM-L12-v2');
  });

  it('returns undefined for an unknown model', () => {
    const info = getTokenizerInfo('nonexistent/model');
    expect(info).toBeUndefined();
  });

  it('returns correct info for BGE model', () => {
    const info = getTokenizerInfo('Xenova/bge-small-en-v1.5');
    expect(info).toBeDefined();
    expect(info!.maxTokens).toBe(512);
    expect(info!.modelId).toBe('Xenova/bge-small-en-v1.5');
  });
});

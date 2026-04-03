import { describe, it, expect } from 'vitest';
import { chunkByTokenCount, chunkBySentence } from '../../src/text/chunking';

describe('chunkByTokenCount', () => {
  it('splits long text into chunks', () => {
    const words = Array.from({ length: 100 }, (_, i) => `word${i}`);
    const text = words.join(' ');
    const chunks = chunkByTokenCount(text, 20);
    expect(chunks.length).toBeGreaterThan(1);
    for (const chunk of chunks) {
      // Each chunk should have roughly maxTokens/1.3 ≈ 15 words or fewer
      const wordCount = chunk.split(' ').length;
      expect(wordCount).toBeLessThanOrEqual(16);
    }
  });

  it('respects overlap', () => {
    const words = Array.from({ length: 30 }, (_, i) => `w${i}`);
    const text = words.join(' ');
    const chunks = chunkByTokenCount(text, 13, { overlap: 5 });
    expect(chunks.length).toBeGreaterThan(1);
    // With overlap, later chunks should share some words with previous chunks
    if (chunks.length >= 2) {
      const firstWords = chunks[0].split(' ');
      const secondWords = chunks[1].split(' ');
      // Some words from end of chunk 0 should appear at start of chunk 1
      const lastFew = firstWords.slice(-4);
      const firstFew = secondWords.slice(0, 4);
      const overlap = lastFew.filter((w) => firstFew.includes(w));
      expect(overlap.length).toBeGreaterThan(0);
    }
  });

  it('never splits mid-word', () => {
    const text = 'superlongword anotherlongword yetanother';
    const chunks = chunkByTokenCount(text, 2);
    for (const chunk of chunks) {
      // Every chunk should contain only complete words
      const words = chunk.split(' ');
      for (const word of words) {
        expect(word).not.toContain(' ');
        expect(word.length).toBeGreaterThan(0);
      }
    }
  });

  it('returns empty array for empty text', () => {
    expect(chunkByTokenCount('', 10)).toEqual([]);
  });

  it('returns empty array for whitespace-only text', () => {
    expect(chunkByTokenCount('   ', 10)).toEqual([]);
  });

  it('returns single chunk for short text', () => {
    const chunks = chunkByTokenCount('hello world', 100);
    expect(chunks).toEqual(['hello world']);
  });

  it('uses custom separator', () => {
    const chunks = chunkByTokenCount('one two three four five six', 4, { separator: '-' });
    for (const chunk of chunks) {
      expect(chunk).toContain('-');
    }
  });
});

describe('chunkBySentence', () => {
  it('splits on sentence boundaries', () => {
    const text = 'Hello world. How are you? I am fine! Great.';
    const chunks = chunkBySentence(text);
    expect(chunks.length).toBe(4);
  });

  it('groups sentences under maxTokens', () => {
    const text = 'Short. Also short. This one is a bit longer sentence. Another one.';
    const chunks = chunkBySentence(text, { maxTokens: 10 });
    expect(chunks.length).toBeGreaterThan(1);
  });

  it('returns empty array for empty text', () => {
    expect(chunkBySentence('')).toEqual([]);
  });

  it('returns single sentence as one chunk', () => {
    const chunks = chunkBySentence('Just one sentence.');
    expect(chunks).toHaveLength(1);
  });

  it('handles text with only whitespace', () => {
    expect(chunkBySentence('   ')).toEqual([]);
  });
});
